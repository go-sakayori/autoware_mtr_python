from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2

from awml_pred.datatype import WaymoAgent

if TYPE_CHECKING:
    from awml_pred.dataclass import EvaluationData
    from awml_pred.typing import NDArray

all_gpus = tf.config.experimental.list_physical_devices("GPU")
if all_gpus:
    try:
        for cur_gpu in all_gpus:
            tf.config.experimental.set_memory_growth(cur_gpu, enable=True)
    except RuntimeError as ex:
        warnings.warn(ex, stacklevel=1)

__all__ = ("waymo_evaluation",)


def _default_metrics_config(eval_second: int, eval_topk: int = 6) -> str:
    assert eval_second in (3, 5, 8)
    config = motion_metrics_pb2.MotionMetricsConfig()
    config_text = """
    track_steps_per_second: 10
    prediction_steps_per_second: 2
    track_history_samples: 10
    speed_lower_bound: 1.4
    speed_upper_bound: 11.0
    speed_scale_lower: 0.5
    speed_scale_upper: 1.0
    step_configurations {
    measurement_step: 5
    lateral_miss_threshold: 1.0
    longitudinal_miss_threshold: 2.0
    }
    """
    config_text += f"""
    max_predictions: {eval_topk}
    """
    if eval_second == 3:  # noqa
        config_text += """
        track_future_samples: 30
        """
    elif eval_second == 5:  # noqa
        config_text += """
        track_future_samples: 50
        step_configurations {
        measurement_step: 9
        lateral_miss_threshold: 1.8
        longitudinal_miss_threshold: 3.6
        }
        """
    else:
        config_text += """
        track_future_samples: 80
        step_configurations {
        measurement_step: 9
        lateral_miss_threshold: 1.8
        longitudinal_miss_threshold: 3.6
        }
        step_configurations {
        measurement_step: 15
        lateral_miss_threshold: 3.0
        longitudinal_miss_threshold: 6.0
        }
        """

    text_format.Parse(config_text, config)
    return config


def _format_for_waymo(
    eval_data: list[EvaluationData],
    eval_topk: int = -1,
    eval_interval: int = 5,
) -> tuple[NDArray, NDArray, dict, dict]:
    """Format EvaluationData for waymo evaluation.

    Args:
    ----
        eval_data (list[EvaluationData]): List of EvaluationData instances.
        eval_topk (int, optional): The number of top-k modes to be evaluated. Defaults to -1.
        eval_interval (int, optional): Timestamp interval to evaluate. Defaults to 5.

    Returns:
    -------
        tuple[NDArray, NDArray, dict, dict]: _description_

    """
    num_scenario = len(eval_data)
    max_num_agent = max(data.num_agent for data in eval_data)

    # TODO: validate using data[0]
    num_scenario_frame = eval_data[0].num_scenario_frame
    num_future = eval_data[0].num_future
    num_mode = eval_data[0].num_mode

    num_eval_frame = num_future // eval_interval

    eval_topk = min(eval_topk, num_mode) if eval_topk != -1 else num_mode
    assert eval_topk > 1, f"top k must be greater than 1, but {eval_topk}"

    batch_pred_trajs = np.zeros((num_scenario, max_num_agent, eval_topk, 1, num_eval_frame, 2))
    batch_pred_scores = np.zeros((num_scenario, max_num_agent, eval_topk))
    gt_trajs = np.zeros((num_scenario, max_num_agent, num_scenario_frame, 7))
    gt_is_valid = np.zeros((num_scenario, max_num_agent, num_scenario_frame), dtype=np.uint8)
    pred_gt_indices = np.zeros((num_scenario, max_num_agent, 1), dtype=np.int64)
    pred_gt_indices_mask = np.zeros((num_scenario, max_num_agent, 1), dtype=bool)
    agent_types = np.zeros((num_scenario, max_num_agent), dtype=np.int64)
    agent_ids = np.zeros((num_scenario, max_num_agent), dtype=np.uint64)
    scenario_ids = np.zeros(num_scenario, dtype=object)

    agent_type_count: dict[str, int] = {mem.name: 0 for mem in WaymoAgent}

    for scene_idx, data in enumerate(eval_data):
        scenario_ids[scene_idx] = data.scenario_id

        prediction = data.prediction
        ground_truth = data.ground_truth

        sort_indices = np.argsort(-prediction.score, axis=1)
        cur_pred_score: NDArray = np.take_along_axis(prediction.score, sort_indices, axis=1)
        cur_pred_score = np.divide(cur_pred_score, cur_pred_score.sum(), where=cur_pred_score != 0)
        cur_pred_xy: NDArray = np.take_along_axis(prediction.xy, sort_indices[..., None, None], axis=1)  # (N, M, T, 2)

        cur_num_agent = prediction.num_agent
        batch_pred_scores[scene_idx, :cur_num_agent] = cur_pred_score[:, :eval_topk]
        batch_pred_trajs[scene_idx, :cur_num_agent] = cur_pred_xy[:, :eval_topk, None, 4::eval_interval]
        gt_trajs[scene_idx, :cur_num_agent] = np.concatenate(
            [
                data.ground_truth.xy,
                data.ground_truth.size[..., :2],
                data.ground_truth.yaw[..., None],
                data.ground_truth.vxy,
            ],
            axis=-1,
        )
        gt_is_valid[scene_idx, :cur_num_agent] = ground_truth.is_valid
        pred_gt_indices[scene_idx, :cur_num_agent, 0] = np.arange(cur_num_agent, dtype=np.int64)
        pred_gt_indices_mask[scene_idx, :cur_num_agent, 0] = True
        # TODO: use enum directly
        agent_types[scene_idx, :cur_num_agent] = np.array([WaymoAgent.from_str(t) for t in ground_truth.types])
        agent_ids[scene_idx, :cur_num_agent] = ground_truth.ids

        for name in ground_truth.types:
            agent_type_count[name] += 1

    gt_info = {
        "scenario_id": scenario_ids.tolist(),
        "object_id": agent_ids.tolist(),
        "object_type": agent_types.tolist(),
        "gt_is_valid": gt_is_valid,
        "gt_trajectory": gt_trajs,
        "pred_gt_indices": pred_gt_indices,
        "pred_gt_indices_mask": pred_gt_indices_mask,
    }
    return batch_pred_scores, batch_pred_trajs, gt_info, agent_type_count


def waymo_evaluation(
    eval_data: list[EvaluationData],
    eval_topk: int = 6,
    eval_second: int = 8,
) -> tuple[dict, str]:
    """Evaluate with waymo metrics.

    Args:
    ----
        eval_data (list[EvaluationData]): List of evaluation data.
        eval_topk (int, optional): Number of top-k modes to evaluate. Defaults to 6.
        eval_second (int, optional): Time length to evaluate in [s]. Defaults to 8.

    Returns:
    -------
        tuple[dict, str]: Evaluation result data and text.

    """
    pred_score, pred_trajectory, gt_infos, object_type_cnt_dict = _format_for_waymo(
        eval_data,
        eval_topk=eval_topk,
    )
    eval_config = _default_metrics_config(eval_second=eval_second, eval_topk=eval_topk)

    pred_score = tf.convert_to_tensor(pred_score, np.float32)
    pred_trajs = tf.convert_to_tensor(pred_trajectory, np.float32)
    gt_trajs = tf.convert_to_tensor(gt_infos["gt_trajectory"], np.float32)
    gt_is_valid = tf.convert_to_tensor(gt_infos["gt_is_valid"], bool)
    pred_gt_indices = tf.convert_to_tensor(gt_infos["pred_gt_indices"], tf.int64)
    pred_gt_indices_mask = tf.convert_to_tensor(gt_infos["pred_gt_indices_mask"], bool)
    object_type = tf.convert_to_tensor(gt_infos["object_type"], tf.int64)

    metric_results = py_metrics_ops.motion_metrics(
        config=eval_config.SerializeToString(),
        prediction_trajectory=pred_trajs,  # (B, num_pred_groups, top_k, num_agents_per_group, num_pred_steps)
        prediction_score=pred_score,  # (B, num_pred_groups, top_k)
        ground_truth_trajectory=gt_trajs,  # (B, num_total_agents, num_gt_steps, 7)
        ground_truth_is_valid=gt_is_valid,  # (B, num_total_agents, num_gt_steps)
        prediction_ground_truth_indices=pred_gt_indices,  # (B, num_pred_groups, num_agents_per_group)
        prediction_ground_truth_indices_mask=pred_gt_indices_mask,  # (B, num_pred_groups, num_agents_per_group)
        object_type=object_type,  # (B, num_total_agents)
    )

    metric_names = config_util.get_breakdown_names_from_motion_config(eval_config)

    result_dict = {}
    avg_results = {}
    for i, m in enumerate(["minADE", "minFDE", "MissRate", "OverlapRate", "mAP"]):
        avg_results.update({f"{m} - VEHICLE": [0.0, 0], f"{m} - PEDESTRIAN": [0.0, 0], f"{m} - CYCLIST": [0.0, 0]})
        for j, n in enumerate(metric_names):
            cur_name = n.split("_")[1]
            avg_results[f"{m} - {cur_name}"][0] += float(metric_results[i][j])
            avg_results[f"{m} - {cur_name}"][1] += 1
            result_dict[f"{m} - {n}\t"] = float(metric_results[i][j])

    for key in avg_results:
        avg_results[key] = avg_results[key][0] / avg_results[key][1]

    result_dict["-------------------------------------------------------------"] = 0
    result_dict.update(avg_results)

    final_avg_results = {}
    result_format_list = [
        ["Waymo", "mAP", "minADE", "minFDE", "MissRate", "\n"],
        ["VEHICLE", None, None, None, None, "\n"],
        ["PEDESTRIAN", None, None, None, None, "\n"],
        ["CYCLIST", None, None, None, None, "\n"],
        ["Avg", None, None, None, None, "\n"],
    ]
    name_to_row = {"VEHICLE": 1, "PEDESTRIAN": 2, "CYCLIST": 3, "Avg": 4}
    name_to_col = {"mAP": 1, "minADE": 2, "minFDE": 3, "MissRate": 4}

    for cur_metric_name in ["minADE", "minFDE", "MissRate", "mAP"]:
        final_avg_results[cur_metric_name] = 0
        for cur_name in ["VEHICLE", "PEDESTRIAN", "CYCLIST"]:
            cur_score = avg_results[f"{cur_metric_name} - {cur_name}"]
            final_avg_results[cur_metric_name] += cur_score
            result_format_list[name_to_row[cur_name]][name_to_col[cur_metric_name]] = f"{cur_score:.4f}"

        final_avg_results[cur_metric_name] /= 3
        result_format_list[4][name_to_col[cur_metric_name]] = f"{final_avg_results[cur_metric_name]:.4f}"

    result_format_str = " ".join([x.rjust(12) for items in result_format_list for x in items])

    result_dict["--------------------------------------------------------------"] = 0
    result_dict.update(final_avg_results)
    result_dict["---------------------------------------------------------------"] = 0
    result_dict.update(object_type_cnt_dict)
    result_dict[
        "-----Note that this evaluation may have marginal differences with the official Waymo evaluation server-----"
    ] = 0

    return result_dict, result_format_str
