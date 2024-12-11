from numbers import Number
from typing import Final

import numpy as np
import torch

from awml_pred.typing import NDArray, Tensor


def _check_numpy_to_torch(x: Number | NDArray | Tensor) -> tuple[Tensor, bool]:
    """Check whether input object is `numpy.ndarray` or `torch.Tensor`.

    If `numpy.ndarray`, convert it to `torch.Tensor` and returns `False`.

    Args:
    ----
        x (Number | NDArray | Tensor): _description_

    Returns:
    -------
        tuple[Tensor, bool]: If `numpy.ndarray`, convert it to `torch.Tensor` and returns `False`.

    """
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    elif isinstance(x, Number):
        return torch.tensor([x]).float(), True
    return x, False


def rotate_points_along_z(points: NDArray | Tensor, angle: Number | NDArray | Tensor) -> NDArray | Tensor:
    """Rotate points array along z-axis.

    Args:
    ----
        points(NDArray | Tensor): (B, N, 3 + C)
        angle (Number | NDarray | Tensor): (B), angle along z-axis, angle increases x ==> y

    Returns:
    -------
        NDArray | Tensor: Rotated points array or tensor.

    """
    DeprecationWarning("`rotate_points_along_z` is deprecated, use `rotate_along_z` instead.")

    points, is_numpy = _check_numpy_to_torch(points)
    angle, _ = _check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])

    dim2d: Final[int] = 2
    if points.shape[-1] == dim2d:
        rot_matrix = torch.stack((cosa, sina, -sina, cosa), dim=1).view(-1, 2, 2).float()
        points_rot = torch.matmul(points, rot_matrix)
    else:
        ones = angle.new_ones(points.shape[0])
        rot_matrix = (
            torch.stack((cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), dim=1).view(-1, 3, 3).float()
        )
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def rotate_along_z(points: NDArray | Tensor, angle: Number | NDArray | Tensor) -> NDArray | Tensor:
    """Rotate input points along z-axis.

    Args:
    ----
        points (NDArray | Tensor): Matrix of coordinates.
        angle (Number | NDArray | Tensor): Scalar or matrix of angle in [rad].

    Returns:
    -------
        NDArray | Tensor: Rotated points array.

    """
    points, is_numpy = _check_numpy_to_torch(points)
    angle, _ = _check_numpy_to_torch(angle)

    cos = torch.cos(angle)
    sin = torch.sin(angle)

    dim: int = points.shape[-1]
    dim2d: Final[int] = 2
    dim3d: Final[int] = 3
    if dim == dim2d:
        rotation = torch.stack([cos, -sin, sin, cos], dim=1).reshape(-1, 2, 2)
    elif dim == dim3d:
        zeros = torch.zeros_like(angle)
        ones = torch.ones(angle)
        rotation = torch.stack([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], dim=1).reshape(-1, 3, 3)
    else:
        raise ValueError(f"Unexpected point dimensions: {dim}")

    output = torch.matmul(points, rotation)

    return output.numpy() if is_numpy else output
