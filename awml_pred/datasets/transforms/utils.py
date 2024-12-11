from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from awml_pred.typing import NDArrayBool, NDArrayF32, NDArrayI64


__all__ = ["batch_polyline"]


def batch_polyline(
    polyline: NDArrayF32,
    num_point: int,
    break_distance: float,
) -> tuple[NDArrayF32, NDArrayBool]:
    """Generate batch polyline from (L, D) to (K, P, D).

    Args:
    ----
        polyline (NDArrayF32): Polyline array in shape (L, D).
        num_point (int): The number of points contained in output polyline, P.
        break_distance (float): Distance threshold to break points into 2 polyline groups.

    Returns:
    -------
        tuple[NDArrayF32, NDArrayBool]: Result polyline array and its mask.

    """
    break_idx: NDArrayI64 = _polyline_break_idx(polyline, break_distance)
    polyline_list: list[NDArrayF32] = np.array_split(polyline, break_idx, axis=0)

    ret_polyline_list: list[NDArrayF32] = []
    ret_polyline_mask_list: list[NDArrayBool] = []

    for points in polyline_list:
        num_pts = len(points)
        if num_pts <= 0:
            continue
        for idx in range(0, num_pts, num_point):
            _append_single_polyline(points[idx : idx + num_point], num_point, ret_polyline_list, ret_polyline_mask_list)

    ret_polyline: NDArrayF32 = np.stack(ret_polyline_list, axis=0)
    ret_polyline_mask: NDArrayBool = np.stack(ret_polyline_mask_list, axis=0)

    return ret_polyline, ret_polyline_mask


def _polyline_break_idx(polyline: NDArrayF32, break_distance: float) -> NDArrayI64:
    """Return indices to break a single polyline into multiple.

    Args:
    ----
        polyline (NDArrayF32): Polyline array in shape (L, D).
        break_distance (float): Distance threshold to break a polyline into two.

    Returns:
    -------
        NDArrayI64: Indices of polyline groups.

    """
    polyline_shifts = np.roll(polyline, shift=1, axis=0)
    buffer = np.concatenate((polyline[:, 0:2], polyline_shifts[:, 0:2]), axis=-1)
    buffer[0, 2:4] = buffer[0, 0:2]
    return (np.linalg.norm(buffer[:, 0:2] - buffer[:, 2:4], axis=-1) > break_distance).nonzero()[0]


def _append_single_polyline(
    new_polyline: NDArrayF32,
    num_point: int,
    ret_polyline_list: list[NDArrayF32],
    ret_polyline_mask_list: list[NDArrayBool],
) -> None:
    """Append a single polyline info to a `ret_*`.

    Args:
    ----
        new_polyline (NDArrayF32): Polyline array to be appended.
        num_point (int): Max number of points contained in a single polyline.
        ret_polyline_list (list[NDArrayF32]): A container to append new polyline.
        ret_polyline_mask_list (NDArrayBool): A container to append new polyline mask.

    """
    num_new_polyline, point_dim = new_polyline.shape

    cur_polyline: NDArrayF32 = np.zeros((num_point, point_dim), dtype=np.float32)
    cur_polyline_mask: NDArrayBool = np.zeros(num_point, dtype=np.bool_)
    cur_polyline[:num_new_polyline] = new_polyline
    cur_polyline_mask[:num_new_polyline] = True

    ret_polyline_list.append(cur_polyline)
    ret_polyline_mask_list.append(cur_polyline_mask)
