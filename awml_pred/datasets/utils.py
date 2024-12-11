from __future__ import annotations

from typing import TYPE_CHECKING, Final, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
    from numbers import Number

    from awml_pred.typing import Tensor


def merge_batch_by_padding_2nd_dim(
    tensor_list: list[Tensor],
    *,
    return_pad_mask: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Merge batch data by padding 2nd dimensions.

    Args:
    ----
        tensor_list (list[Tensor]): Each tensor is in shape
            * 3D tensors: [(N0, D1, D2), (N1, D1, D2), ..., (Nn, D1, D2)]
            -> (N, D1, D2)
                * N = N0 + N1 + ... + Nn
            * 4D tensors: [(B0, N0, D2, D3), (B1, N1, D2, D3), ..., (Bn, Nn, D2, D3)]
            -> (B, N, D2, D3)
                * B = B0 + B1 + ... + Bn
                * N = max(N0, N1, ..., Nn)
        return_pad_mask (bool, optional): Whether to return pad mask. Defaults to False.

    Returns:
    -------
        Tensor | tuple[Tensor, Tensor]: Merged tensor and return pad mask if `return_pad_mask=True`.

    """
    assert all(t.ndim == tensor_list[0].ndim for t in tensor_list[1:]), "All tensors must have same dimension"
    assert all(t.ndim in (3, 4) for t in tensor_list), "All tensors must be 3D or 4D"
    is_3d_tensor = False
    ndim3d: Final[int] = 3
    if tensor_list[0].ndim == ndim3d:
        tensor_list = [x.unsqueeze(dim=-1) for x in tensor_list]
        is_3d_tensor = True
    max_feat0 = max([x.shape[1] for x in tensor_list])

    _, _, num_feat1, num_feat2 = tensor_list[0].shape

    ret_tensor_list = []
    ret_mask_list = []
    for cur_tensor in tensor_list:
        assert cur_tensor.shape[2] == num_feat1 and cur_tensor.shape[3] == num_feat2

        new_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], max_feat0, num_feat1, num_feat2)
        new_tensor[:, : cur_tensor.shape[1], :, :] = cur_tensor
        ret_tensor_list.append(new_tensor)

        new_mask_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], max_feat0)
        new_mask_tensor[:, : cur_tensor.shape[1]] = 1
        ret_mask_list.append(new_mask_tensor.bool())

    ret_tensor = torch.cat(ret_tensor_list, dim=0)  # (num_stacked_samples, num_feat0_maxt, num_feat1, num_feat2)
    ret_mask = torch.cat(ret_mask_list, dim=0)

    if is_3d_tensor:
        ret_tensor = ret_tensor.squeeze(dim=-1)

    if return_pad_mask:
        return ret_tensor, ret_mask
    return ret_tensor


def relative_timestamps(timestamps: Sequence[Number]) -> Sequence[Number]:
    """Return the relative timestamps from the first element.

    Args:
    ----
        timestamps (Sequence[Number]): Sequence of timestamps.

    Raises:
    ------
        TypeError: Expecting `NDArray`, `Tensor`, `list` or `tuple`.

    Returns:
    -------
        Sequence[Number]: Return the `NDArray` if the input is `NDArray` or `list` if the input is `list | tuple`.

    """
    if isinstance(timestamps, (np.ndarray, torch.Tensor)):
        return timestamps - timestamps[0]
    elif isinstance(timestamps, (list, tuple)):
        return [t - timestamps[0] for t in timestamps]
    else:
        raise TypeError(f"Unexpected type: {type(timestamps)}")
