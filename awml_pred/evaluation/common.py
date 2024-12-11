from __future__ import annotations

import math
from collections import OrderedDict
from typing import TYPE_CHECKING, Sequence

import numpy as np
from tabulate import tabulate

if TYPE_CHECKING:
    from awml_pred.dataclass import EvaluationData
    from awml_pred.typing import NDArrayFloat

LOW_PROB_THRESHOLD_FOR_METRICS = 0.05

__all__ = ("common_evaluation", "evaluation_result_as_table")


def get_ade(forecasted_trajectory: NDArrayFloat, gt_trajectory: NDArrayFloat) -> float:
    """Compute Average Displacement Error.

    Args:
    ----
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)

    Returns:
    -------
        ade: Average Displacement Error

    """
    pred_len = forecasted_trajectory.shape[0]
    return float(
        sum(
            math.sqrt(
                (forecasted_trajectory[i, 0] - gt_trajectory[i, 0]) ** 2
                + (forecasted_trajectory[i, 1] - gt_trajectory[i, 1]) ** 2,
            )
            for i in range(pred_len)
        )
        / pred_len,
    )


def get_fde(forecasted_trajectory: NDArrayFloat, gt_trajectory: NDArrayFloat) -> float:
    """Compute Final Displacement Error.

    Args:
    ----
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)

    Returns:
    -------
        fde: Final Displacement Error

    """
    return math.sqrt(
        (forecasted_trajectory[-1, 0] - gt_trajectory[-1, 0]) ** 2
        + (forecasted_trajectory[-1, 1] - gt_trajectory[-1, 1]) ** 2,
    )


def get_displacement_errors_and_miss_rate(
    forecasted_trajectories: dict[int, list[NDArrayFloat]],
    gt_trajectories: dict[int, NDArrayFloat],
    max_guesses: int,
    horizon: int,
    miss_threshold: float,
    forecasted_probabilities: dict[int, list[float]] | None = None,
) -> dict[str, float]:
    """Compute min fde and ade for each sample.

    Note: Both min_fde and min_ade values correspond to the trajectory which has minimum fde.
    The Brier Score is defined here:
        Brier, G. W. Verification of forecasts expressed in terms of probability. Monthly weather review, 1950.
        https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml

    Args:
    ----
        forecasted_trajectories: Predicted top-k trajectory dict with key as seq_id and value as list of trajectories.
                Each element of the list is of shape (pred_len x 2).
        gt_trajectories: Ground Truth Trajectory dict with key as seq_id and values as trajectory of
                shape (pred_len x 2)
        max_guesses: Number of guesses allowed
        horizon: Prediction horizon
        miss_threshold: Distance threshold for the last predicted coordinate
        forecasted_probabilities: Probabilities associated with forecasted trajectories.

    Returns:
    -------
        metric_results: Metric values for minADE, minFDE, MR, p-minADE, p-minFDE, p-MR, brier-minADE, brier-minFDE

    """
    metric_results: dict[str, float] = OrderedDict()
    min_ade, prob_min_ade, brier_min_ade = [], [], []
    min_fde, prob_min_fde, brier_min_fde = [], [], []
    n_misses, prob_n_misses = [], []
    for k, v in gt_trajectories.items():
        curr_min_ade = float("inf")
        curr_min_fde = float("inf")
        min_idx = 0
        max_num_traj = min(max_guesses, len(forecasted_trajectories[k]))

        # If probabilities available, use the most likely trajectories, else use the first few
        if forecasted_probabilities is not None:
            sorted_idx = np.argsort([-x for x in forecasted_probabilities[k]], kind="stable")
            pruned_probabilities = [forecasted_probabilities[k][t] for t in sorted_idx[:max_num_traj]]
            # Normalize
            prob_sum = sum(pruned_probabilities)
            pruned_probabilities = [p / prob_sum for p in pruned_probabilities]
        else:
            sorted_idx = np.arange(len(forecasted_trajectories[k]))
        pruned_trajectories = [forecasted_trajectories[k][t] for t in sorted_idx[:max_num_traj]]

        for j in range(len(pruned_trajectories)):
            fde = get_fde(pruned_trajectories[j][:horizon], v[:horizon])
            if fde < curr_min_fde:
                min_idx = j
                curr_min_fde = fde
        curr_min_ade = get_ade(pruned_trajectories[min_idx][:horizon], v[:horizon])
        min_ade.append(curr_min_ade)
        min_fde.append(curr_min_fde)
        n_misses.append(curr_min_fde > miss_threshold)

        if forecasted_probabilities is not None:
            prob_n_misses.append(1.0 if curr_min_fde > miss_threshold else (1.0 - pruned_probabilities[min_idx]))
            prob_min_ade.append(
                min(
                    -np.log(pruned_probabilities[min_idx]),
                    -np.log(LOW_PROB_THRESHOLD_FOR_METRICS),
                )
                + curr_min_ade,
            )
            brier_min_ade.append((1 - pruned_probabilities[min_idx]) ** 2 + curr_min_ade)
            prob_min_fde.append(
                min(
                    -np.log(pruned_probabilities[min_idx]),
                    -np.log(LOW_PROB_THRESHOLD_FOR_METRICS),
                )
                + curr_min_fde,
            )
            brier_min_fde.append((1 - pruned_probabilities[min_idx]) ** 2 + curr_min_fde)

    metric_results["minADE"] = sum(min_ade) / len(min_ade) if len(min_ade) != 0 else np.nan
    metric_results["minFDE"] = sum(min_fde) / len(min_fde) if len(min_fde) != 0 else np.nan
    metric_results["MR"] = sum(n_misses) / len(n_misses) if len(n_misses) != 0 else np.nan
    if forecasted_probabilities is not None:
        metric_results["p-minADE"] = sum(prob_min_ade) / len(prob_min_ade) if len(prob_min_ade) != 0 else np.nan
        metric_results["p-minFDE"] = sum(prob_min_fde) / len(prob_min_fde) if len(prob_min_fde) != 0 else np.nan
        metric_results["p-MR"] = sum(prob_n_misses) / len(prob_n_misses) if len(prob_n_misses) != 0 else np.nan
        metric_results["brier-minADE"] = sum(brier_min_ade) / len(brier_min_ade) if len(brier_min_ade) != 0 else np.nan
        metric_results["brier-minFDE"] = sum(brier_min_fde) / len(brier_min_fde) if len(brier_min_fde) != 0 else np.nan
    return metric_results


def average_result(results: dict[str, dict]) -> dict[str, dict]:
    """Calculate average score for each metric and return a dict including average score.

    Args:
    ----
        results (dict[str, dict]): Evaluation scores for each metric and type.

    Returns:
    -------
        dict[str, dict]: A dict including average score.

    """
    output = results.copy()
    for metric, scores in results.items():
        output[metric]["Avg"] = np.nanmean([*scores.values()])
    return output


def evaluation_result_as_table(results: dict[str, dict], types: Sequence[str], fmt: str = "github") -> str:
    """Convert a dict of evaluation results to a table string.

    Args:
    ----
        results (dict[str, dict]): Dict of evaluation results.
        types (Sequence[str]): Type names to be evaluated.
        fmt (str, optional): Name of format of table.

    Returns:
    -------
        str: Table string.

    """
    metrics = results.keys()

    headers = ("Type", *metrics)
    type_names = ("Avg", *types)

    data = []
    for t in type_names:
        current = [t]
        for metric in metrics:
            current.append(results[metric][t])
        data.append(current)
    return tabulate(data, headers=headers, tablefmt=fmt)


def common_evaluation(
    eval_data: Sequence[EvaluationData],
    types: Sequence[str],
    top_k_modes: int | Sequence[int] = (1, 6),
    miss_threshold: float = 2.0,
) -> dict[str, dict]:
    """Evaluate predicted trajectory.

    Args:
    ----
        eval_data (Sequence[EvaluationData]): Sequence of evaluation data.
        types (Sequence[str]): Sequence of agent types.
        top_k_modes (int, Sequence[int]): List of the number of top k modes. Defaults to (1, 6).
        miss_threshold (float): Distance threshold indicating missing. Defaults to 2.0.

    Returns:
    -------
        dict[str, dict]: Evaluation results dict formatted as `{<Metrics0>_<K>: {Type0: <Score0>, }, ...}`.

    """
    if isinstance(top_k_modes, int):
        top_k_modes = [top_k_modes]

    # {<Metrics0>_<K>: {Type0: <Score0>, }, ...}
    output: dict[str, dict] = {}
    for t in types:
        # {<Metrics0>_<K>: <Score0>, ...}

        pred_dict = {}
        gt_dict = {}
        prob_dict = {}
        for data in eval_data:
            # mask filters out  specific type trajectories
            mask = data.ground_truth.types == t

            prediction = data.prediction.apply_mask(mask)
            ground_truth = data.ground_truth.apply_mask(mask)

            current_time_idx = data.num_scenario_frame - data.num_future

            # separate trajectories into a single agent
            for i, agent_id in enumerate(ground_truth.ids):
                sequence_id = f"{data.scenario_id}_{agent_id}"
                is_valid = ground_truth.is_valid[i, current_time_idx:]
                pred_dict[sequence_id] = np.where(is_valid[None, :, None], prediction.xy[i], 0.0)
                gt_dict[sequence_id] = np.where(is_valid[..., None], ground_truth.xy[i, current_time_idx:], 0.0)
                prob_dict[sequence_id] = prediction.score[i]

        for top_k in top_k_modes:
            if data.num_mode < top_k:
                continue

            result_k = get_displacement_errors_and_miss_rate(
                pred_dict,
                gt_dict,
                max_guesses=top_k,
                horizon=data.num_future,
                miss_threshold=miss_threshold,
                forecasted_probabilities=prob_dict,
            )

            for metric, score in result_k.items():
                key = f"{metric}_{top_k}"
                if output.get(key) is None:
                    output[key] = {}
                output[f"{metric}_{top_k}"][t] = score

    return average_result(output)
