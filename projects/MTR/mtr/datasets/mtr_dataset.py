from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Sequence

import numpy as np
import torch

from awml_pred.common import DATASETS
from awml_pred.dataclass import EvaluationData, GroundTruth, Prediction
from awml_pred.datasets import BaseDataset
from awml_pred.datasets.utils import merge_batch_by_padding_2nd_dim
from awml_pred.ops import rotate_points_along_z

if TYPE_CHECKING:
    from awml_pred.dataclass import AWMLAgentScenario
    from awml_pred.typing import NDArray, NDArrayF32, NDArrayI32, NDArrayStr, Tensor

__all__ = ("MTRDataset",)


@DATASETS.register()
class MTRDataset(BaseDataset):
    def postprocess(self, info: dict) -> dict:
        """Run process after transformation.

        Args:
        ----
            info (dict): Source info.

        Returns:
        -------
            dict: Result info.

        """
        scenario: AWMLAgentScenario = info["scenario"]

        # TODO(ktro2828): update key names
        ret_info = {}
        ret_info["scenario_id"] = info["scenario_id"]
        ret_info["timestamps"] = scenario.timestamps
        ret_info["obj_trajs"] = info["agent_past"]
        ret_info["obj_trajs_mask"] = info["agent_past_mask"]
        ret_info["track_index_to_predict"] = scenario.target_indices
        ret_info["obj_trajs_pos"] = info["agent_past_pos"]
        ret_info["obj_trajs_last_pos"] = info["agent_last_pos"]
        ret_info["obj_types"] = scenario.types
        ret_info["obj_ids"] = scenario.ids
        ret_info["center_objects_world"] = (
            scenario.get_current().as_array()
            if self.predict_all_agents
            else scenario.get_current(scenario.target_indices).as_array()
        )
        ret_info["center_objects_type"] = scenario.types if self.predict_all_agents else scenario.target_types
        ret_info["center_objects_id"] = scenario.ids if self.predict_all_agents else scenario.target_ids
        ret_info["obj_trajs_future_state"] = info["agent_future"]
        ret_info["obj_trajs_future_mask"] = info["agent_future_mask"]
        ret_info["center_gt_trajs"] = info["gt_trajs"]
        ret_info["center_gt_trajs_mask"] = info["gt_trajs_mask"]
        ret_info["center_gt_final_valid_idx"] = info["gt_target_index"]
        ret_info["center_gt_trajs_src"] = (
            scenario.trajectory if self.predict_all_agents else scenario.get_target(as_array=True)
        )
        ret_info["map_polylines"] = info["polylines"]
        ret_info["map_polylines_mask"] = info["polylines_mask"]
        ret_info["map_polylines_center"] = info["polylines_center"]
        ret_info["intention_points"] = info["intention_points"]

        return ret_info

    def collate_batch_input(
        self,
        batch_list: list[dict[str, Any]],
    ) -> dict[str, Tensor]:
        """Collate model inputs from list of stacked batch data.

        Args:
        ----
            batch_list (list[dict[str, Any]]): List of stacked batch data.

        Returns:
        -------
            dict[str, Tensor]: Model inputs.

        """
        input_keys = (
            (
                "obj_trajs",
                "obj_trajs_mask",
                "map_polylines",
                "map_polylines_mask",
                "map_polylines_center",
                "obj_trajs_last_pos",
                "track_index_to_predict",
                "intention_points",
                "center_gt_trajs",
                "center_gt_trajs_mask",
                "center_gt_final_valid_idx",
                "obj_trajs_future_state",
                "obj_trajs_future_mask",
            )
            if self.training
            else (
                "obj_trajs",
                "obj_trajs_mask",
                "map_polylines",
                "map_polylines_mask",
                "map_polylines_center",
                "obj_trajs_last_pos",
                "track_index_to_predict",
                "intention_points",
            )
        )

        batch_size = len(batch_list)
        batch_input = {
            key: [torch.from_numpy(batch_list[bs_idx][key]) for bs_idx in range(batch_size)] for key in input_keys
        }

        for key, items in batch_input.items():
            if key in (
                "obj_trajs",
                "obj_trajs_mask",
                "map_polylines",
                "map_polylines_mask",
                "map_polylines_center",
                "obj_trajs_pos",
                "obj_trajs_last_pos",
                "obj_trajs_future_state",
                "obj_trajs_future_mask",
            ):
                batch_input[key] = merge_batch_by_padding_2nd_dim(items)
            else:
                batch_input[key] = torch.cat(items, dim=0)

        return batch_input

    def collate_batch_meta(self, batch_list: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate metadata from list of stacked batch data.

        Args:
        ----
        batch_list (list[dict[str, Any]]): List of stacked batch data.

        Returns:
        -------
            dict[str, Any]: Input meta data.
                batch_size (int): Batch size(=B).
                batch_sample_count (list[int]): The number of targets in each batch.
                input_dict (dict):
                    center_objects_world: Target objects coordinates in the shape of (B, D).
                    center_objects_type: Sequence of target agent types in the shape of (B,).
                    center_objects_id: Sequence of target agent ids in the shape of (B,).
                    center_gt_trajs_src: Target agent trajectory (B, T, D).

        """
        meta_keys: Final[Sequence[str]] = (
            "scenario_id",
            "timestamps",
            "center_objects_type",
            "center_objects_id",
            "center_objects_world",
            "center_gt_trajs_src",
            "track_index_to_predict",
        )

        batch_size = len(batch_list)
        batch_sample_count = [len(x["track_index_to_predict"]) for x in batch_list]

        batch_input_meta = {}
        for key in meta_keys:
            if key == "timestamps":
                batch_input_meta[key] = np.stack(
                    [batch_list[idx][key] for idx in range(batch_size)],
                    axis=0,
                )
            else:
                batch_input_meta[key] = np.concatenate(
                    [batch_list[idx][key] for idx in range(batch_size)],
                    axis=0,
                )

        return {
            "batch_size": batch_size,
            "batch_sample_count": batch_sample_count,
            "batch_input_meta": batch_input_meta,
        }

    def generate_prediction(
        self,
        pred_scores: Tensor,
        pred_trajs: Tensor,
        batch_meta: dict[str, Any],
    ) -> list[EvaluationData]:
        """Generate prediction output.

        Args:
        ----
            pred_scores: Predicted scores in the shape of (B, M)
            pred_trajs: (B, M, T, 7).
            batch_meta (dict[str, NDArray]): Metadata as dict.
                batch_size (int): Batch size(=B).
                batch_sample_count (list[int]): The number of targets in each batch.
                input_dict (dict):
                    center_objects_world: Target objects coordinates in the shape of (B, D).
                    center_objects_type: Sequence of target agent types in the shape of (B,).
                    center_objects_id: Sequence of target agent ids in the shape of (B,).
                    center_gt_trajs_src: Target agent trajectory (B, T, D).

        Returns:
        -------
            list[EvaluationData]: EvaluationData instance.

        """
        batch_size: int = batch_meta["batch_size"]
        batch_input_meta: dict[str, Any] = batch_meta["batch_input_meta"]

        center_objects_world: NDArray = batch_input_meta["center_objects_world"]

        num_all_batch, num_mode, num_future, num_feat = pred_trajs.shape

        pred_num_feat: Final[int] = 7
        assert num_feat == pred_num_feat, "Expected predicted feature dim is 7"

        # transform from target centric coordinates into world coordinates
        pred_trajs_: NDArray = rotate_points_along_z(
            points=pred_trajs.view(num_all_batch, num_mode * num_future, num_feat).cpu().numpy(),
            angle=center_objects_world[:, 6],
        ).reshape(num_all_batch, num_mode, num_future, num_feat)
        pred_trajs_[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2]

        pred_scores_: NDArray = pred_scores.cpu().numpy()

        batch_sample_count: list[int] = batch_meta["batch_sample_count"]
        scenario_ids: NDArrayStr = batch_input_meta["scenario_id"]
        timestamps: NDArrayF32 = batch_input_meta["timestamps"]
        agent_ids: NDArrayI32 = batch_input_meta["center_objects_id"]
        agent_types: NDArrayStr = batch_input_meta["center_objects_type"]
        gt_trajs: NDArrayF32 = batch_input_meta["center_gt_trajs_src"]

        cur_start_idx = 0
        eval_data: list[EvaluationData] = []
        for bs_idx in range(batch_size):
            cur_scenario_id = scenario_ids[bs_idx]
            cur_timestamps = timestamps[bs_idx]
            num_sample = batch_sample_count[bs_idx]
            batch_indices = np.arange(cur_start_idx, cur_start_idx + num_sample)

            prediction = Prediction(
                score=pred_scores_[batch_indices],
                xy=pred_trajs_[batch_indices, ..., 0:2],
            )

            batch_gt_xy = gt_trajs[batch_indices, ..., 0:2]
            batch_gt_size = gt_trajs[batch_indices, ..., 3:5]
            batch_gt_yaw = gt_trajs[batch_indices, ..., 6]
            batch_gt_vxy = gt_trajs[batch_indices, ..., 7:9]
            batch_gt_is_valid = gt_trajs[batch_indices, ..., 9]

            ground_truth = GroundTruth(
                types=agent_types[batch_indices],
                ids=agent_ids[batch_indices],
                xy=batch_gt_xy,
                size=batch_gt_size,
                yaw=batch_gt_yaw,
                vxy=batch_gt_vxy,
                is_valid=batch_gt_is_valid,
            )

            eval_data.append(
                EvaluationData(
                    scenario_id=cur_scenario_id,
                    timestamps=cur_timestamps,
                    prediction=prediction,
                    ground_truth=ground_truth,
                ),
            )
            cur_start_idx += num_sample

        assert cur_start_idx == num_all_batch
        assert len(eval_data) == batch_meta["batch_size"]

        return eval_data
