from __future__ import annotations

from typing import TYPE_CHECKING

from dataclasses import asdict
from enum import Enum
from typing import Any
import numpy as np

__all__ = ("custom_as_dict",)


def to_np_f32(x):
    """Convert an array like object to a numpy float32 array."""
    return np.array(x, dtype=np.float32)


def custom_as_dict(data: Any) -> dict:
    """Return dict. A wrapper of `dataclasses.as_dict` with custom dict factory.

    Args:
    ----
        data (Any): Any data.

    Returns:
    -------
        dict: Dict data.

    """
    return asdict(data, dict_factory=_custom_as_dict_factory)


def _custom_as_dict_factory(data: Any) -> dict:
    """Return dict. Custom factory used in `asdict` function of `dataclass` instance.

    This factory executes following operations for specific types:
        - Enum: executes `ret=str(EnumMember)`, which can be regenerated by `EnumMember=eval(ret)`.

    Args:
    ----
        data (dict): Some data.

    Returns:
    -------
        dict: Converted data.

    """

    def convert_value(obj: Any) -> Any:
        if isinstance(obj, Enum):
            return str(obj)  # enable to reconstruct by `eval()`
        return obj

    return {k: convert_value(v) for k, v in data}
