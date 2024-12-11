from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Polygon, Rectangle

from awml_pred.visualization import ContextColor, LabelColor

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from awml_pred.typing import NDArray

__all__ = ("render_trajectory", "render_polyline", "render_agent", "render_polygon", "setup_axes")


def render_trajectory(
    axes: Axes,
    trajectory: NDArray,
    confidence: NDArray | None = None,
    colors: list[str] | None = None,
    style: str = "-",
    marker: str = "o",
    line_width: float = 0.7,
    marker_size: float = 0.7,
) -> Axes:
    """Render trajectory.

    The shape information represents as follows.
    * M : Number of modes
    * T : Number of time steps
    * D : Number of state dimensions

    Args:
    ----
        axes (Axes): `Axes` instance.
        trajectory (NDArray): Trajectory, in shape `(M, T, D(>2))`.
        confidence (NDArray | None, optional): Scores for each mode, in shape `(M,)`. Defaults to None.
        colors (list[str] | None, optional): List of colors for each mode.
            If None, an unique color is assigned. Defaults to None.
        style (str, optional): Line style. Defaults to `"-"`.
        marker (str, optional): Marker type. Defaults to `"o"`.
        line_width (float, optional): Trajectory line width. Defaults to 0.7.
        marker_size (float, optional): Trajectory marker size. Defaults to 0.7.

    Returns:
    -------
        Axes: Axes instance updated by visualization result.

    """
    if colors is None:
        colors = [None] * len(trajectory)

    if confidence is None:
        for traj, color in zip(trajectory, colors, strict=True):
            axes.plot(traj[:, 0], traj[:, 1], marker=marker, linewidth=line_width, markersize=marker_size, color=color)
    else:
        for traj, conf, color in zip(trajectory, confidence, colors, strict=True):
            axes.plot(
                traj[:, 0],
                traj[:, 1],
                marker=marker,
                alpha=conf,
                linestyle=style,
                linewidth=line_width,
                markersize=marker_size,
                color=color,
            )
    return axes


def render_polyline(
    axes: Axes,
    polyline: NDArray,
    style: str = "-",
    line_width: float = 0.5,
    alpha: float = 1.0,
    color: str = "#E0E0E0",
) -> Axes:
    """Render polylines.

    Args:
    ----
        axes (Axes): Axes instance.
        polyline (NDArrayDouble): NdArray of polylines, in shape (Np, Nw, Dp).
        style (str, optional): Line style. Defaults to `-`.
        line_width (float, optional): Line width. Defaults to 1.0.
        alpha (float, optional): Alpha value. Defaults to 1.0.
        color (str, optional): Polyline color. Defaults to `r`.

    Returns:
    -------
        Axes: Axes instance updated by visualization result.

    """
    axes.plot(
        polyline[:, 0],
        polyline[:, 1],
        linestyle=style,
        linewidth=line_width,
        color=color,
        alpha=alpha,
    )

    return axes


def render_agent(
    axes: Axes,
    position: NDArray,
    heading: float,
    bbox_size: tuple[float, float],
    color: str,
) -> Axes:
    """Render agent with box.

    Args:
    ----
        axes (Axes): Axes instance.
        position (NDArray): Position of actor, (x, y)[m].
        heading (float): Heading of actor, yaw[rad].
        bbox_size (tuple[float, float]): Size of box, (length, width)[m].
        color (str): Color code in string.

    Returns:
    -------
        Axes: Axes instance updated by visualization result.

    """
    length, width = bbox_size
    hypotenuse = np.hypot(length, width)
    theta = np.arctan2(width, length)
    pivot_x = position[0] - 0.5 * hypotenuse * np.cos(heading + theta)
    pivot_y = position[1] - 0.5 * hypotenuse * np.sin(heading + theta)

    bbox = Rectangle((pivot_x, pivot_y), length, width, np.rad2deg(heading), color=color, fill=False)
    axes.add_patch(bbox)

    # Calculate triangle vertices for the front direction indicator
    fx = position[0] + 0.5 * length * np.cos(heading)
    fy = position[1] + 0.5 * length * np.sin(heading)
    lx = position[0] - 0.5 * width * np.cos(heading - np.pi / 2)
    ly = position[1] - 0.5 * width * np.sin(heading - np.pi / 2)
    rx = position[0] + 0.5 * width * np.cos(heading - np.pi / 2)
    ry = position[1] + 0.5 * width * np.sin(heading - np.pi / 2)

    triangle = Polygon(
        [[fx, fy], [lx, ly], [rx, ry]],
        closed=True,
        color=color,
        fill=False,
    )
    axes.add_patch(triangle)

    return axes


def render_polygon(axes: Axes, polygon: NDArray, alpha: float = 1.0, color: str = "#E0E0E0") -> Axes:
    """Render polygon.

    Args:
    ----
        axes (Axes): Axes instance.
        polygon (NDArray): Points of polygon, in shape (Np, D>=2).
        alpha (float, optional): Alpha value. Defaults to 1.0.
        color (str, optional): Color code in string. Defaults to "#E0E0E0".

    Returns:
    -------
        Axes: Axes instance updated by visualization result.

    """
    axes.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=alpha)
    return axes


class LegendType(str, Enum):
    """A enum to represent legend mode."""

    CONTEXT = "CONTEXT"
    LABEL = "LABEL"

    @classmethod
    def from_str(cls, name: str) -> LegendType:
        """Construct a enum member from a name.

        Args:
        ----
            name (str): Name of a member.

        Returns:
        -------
            LegendType: Constructed enum.

        """
        name = name.upper()
        assert name in cls.__members__, f"{name} not in enum members."
        return cls.__members__[name]

    def __eq__(self, __value: str | LegendType) -> bool:
        if isinstance(__value, str):
            return __value.upper() == self.value
        else:
            return super().__eq__(__value)


def setup_axes(
    axes: Axes,
    bounds: tuple[float, float, float, float] | None = None,
    plot_bounds_offset: float = 0.0,
    font_size: float | str = 5.0,
    legend_type: str | LegendType | None = "context",
) -> Axes:
    """Set up Axes instance.

    Args:
    ----
        axes (Axes): Axes instance.
        bounds (tuple[float, float, float, float]): Plot bounds ordering (xmin, ymin, xmax, ymax).
        plot_bounds_offset (float, optional): Offset value of plot bounds. Defaults to 0.0[m].
        font_size (float | str, optional): Font size of legend. Defaults to 5.0.
        legend_type (str | LegendType | None, optional): Type of legend, choose from (context, object).
            Defaults to context.

    Returns:
    -------
        Axes: Axes instance updated by setup configuration.

    """
    if bounds is not None:
        axes.set_xlim(bounds[0] - plot_bounds_offset, bounds[2] + plot_bounds_offset)
        axes.set_ylim(bounds[1] - plot_bounds_offset, bounds[3] + plot_bounds_offset)
    axes.set_aspect("equal", adjustable="box")

    axes.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    axes.margins(0, 0)
    axes.xaxis.set_major_locator(plt.NullLocator())
    axes.yaxis.set_major_locator(plt.NullLocator())

    if legend_type is not None:
        if legend_type == LegendType.CONTEXT:
            handles = [Patch(color=ctx.value, label=ctx.name) for ctx in ContextColor]
        elif legend_type == LegendType.LABEL:
            handles = [Patch(color=agt.value, label=agt.name) for agt in LabelColor]
        else:
            msg = f"Unexpected legend type: {legend_type}"
            raise ValueError(msg)

        axes.legend(handles=handles, bbox_to_anchor=(1.15, 0.01), loc="lower right", fontsize=font_size)

    return axes
