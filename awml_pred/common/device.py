from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from awml_pred.typing import DeviceLike

__all__ = ("items2device", "item2device")


def items2device(items: dict[str, Any], device: DeviceLike) -> dict[str, Any]:
    """Allocate input item on the specified device.

    Args:
    ----
        items (dict[str, Any]): Tensor or list tensors.
        device (DeviceLike): Target device.

    Returns:
    -------
        dict[str, Any]: Items allocated on the device.

    """
    for key, item in items.items():
        items[key] = item2device(item, device)

    return items


def item2device(item: Any, device: DeviceLike) -> Any:
    """Allocate input item on the specified device.

    Args:
    ----
        item (Any): Tensor or list tensors.
        device (DeviceLike): Target device.

    Returns:
    -------
        Any: Item allocated on the device.

    """
    if isinstance(item, torch.Tensor):
        ret = item.to(device)
    elif isinstance(item, list):
        ret = [item2device(it, device) for it in item]
    elif item is None:
        ret = None
    else:
        msg = f"Expected item type is Tensor or list of Tensors, but got {type(item)}"
        raise TypeError(msg)
    return ret
