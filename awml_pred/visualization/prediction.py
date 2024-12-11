from __future__ import annotations

from typing import TYPE_CHECKING

from .color import ColorSemantics, get_semantics_color
from .render import render_agent, render_trajectory, setup_axes

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes

    from awml_pred.dataclass import EvaluationData

__all__ = ("render_prediction",)


def render_prediction(
    axes: Axes,
    eval_data: EvaluationData,
    color_semantics: ColorSemantics,
    *,
    num_mode: int | None = None,
    instance_colors: dict | None = None,
    render_box: bool = False,
    render_gt: bool = False,
) -> Axes:
    """Visualize prediction results.

    Args:
    ----
        axes (Axes): Visualization axes.
        eval_data (EvaluationData): EvaluationData object.
        color_semantics (ColorSemantics): Color semantics.
        num_mode (int | None, optional): The number of modes to be rendered.
        instance_colors (dict | None, optional): Container to store color for each instance.
            If color semantics is not `ColorSemantics.INSTANCE`, no required to specify this.
        render_box (bool, optional): Whether to render agent box.
        render_gt (bool, optional): Whether to render ground truth trajectory.

    Returns:
    -------
        Axes: Figure after predictions were rendered.

    """
    if color_semantics == ColorSemantics.INSTANCE:
        legend_type = None
    elif color_semantics == ColorSemantics.CONTEXT:
        legend_type = "CONTEXT"
    elif color_semantics == ColorSemantics.LABEL:
        legend_type = "LABEL"
    else:
        raise ValueError(f"Unexpected level: {color_semantics}")
    setup_axes(axes, legend_type=legend_type)

    current_time_index = eval_data.num_scenario_frame - eval_data.num_future
    for (agent_type, agent_id, gt_xy, gt_size, gt_yaw, _, is_valid), (pred_score, pred_xy, *_) in zip(
        eval_data.ground_truth,
        eval_data.prediction,
        strict=True,
    ):
        if color_semantics == ColorSemantics.INSTANCE:
            assert instance_colors is not None
            if agent_id in instance_colors:
                color = instance_colors[agent_id]
            else:
                color = get_semantics_color(color_semantics)
                instance_colors[agent_id] = color
        else:
            # TODO: Do not use .replace("TYPE_", "")
            type_name = "FOCAL_AGENT" if color_semantics == ColorSemantics.CONTEXT else agent_type.replace("TYPE_", "")
            color = get_semantics_color(color_semantics, type_name)

        if render_box:
            axes = render_agent(
                axes,
                position=gt_xy[current_time_index],
                heading=gt_yaw[current_time_index],
                bbox_size=gt_size[current_time_index, :2],
                color=color,
            )

        colors = [color] * eval_data.num_mode if num_mode is None else [color] * min(num_mode, eval_data.num_mode)
        axes = render_trajectory(axes, trajectory=pred_xy[:num_mode], confidence=pred_score[:num_mode], colors=colors)
        if render_gt:
            gt_xy_valid = gt_xy[is_valid]
            axes = render_trajectory(axes, trajectory=gt_xy_valid[None, current_time_index:], colors=["red"])
    return axes
