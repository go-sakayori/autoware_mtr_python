from __future__ import annotations

from enum import Enum
from typing import Iterator

__all__ = ("BoxSize", "get_box_size")


class BoxSize(Enum):
    """Box size of objects ordering `(length, width)[m]`."""

    EGO = (4.0, 2.0)

    # scene objects
    VEHICLE = (4.7, 2.0)
    BUS = (8.5, 2.6)
    PEDESTRIAN = (0.7, 0.7)
    CYCLIST = (2.0, 0.7)
    MOTORCYCLIST = (2.0, 0.8)
    UNKNOWN = (1.5, 1.5)

    @classmethod
    def from_str(cls, name: str) -> BoxSize:
        """Return box size of specified context in str.

        Args:
        ----
            name (str): Name of object type.

        Returns:
        -------
            BoxSize: Instance with specified name.

        """
        name = name.upper()
        assert name in cls.__members__, f"{name} is not in enum members of {cls.__name__}."
        return cls.__members__[name]

    def __iter__(self) -> Iterator[tuple[float, float]]:
        return iter(self.value)


def get_box_size(name: str) -> tuple[float, float]:
    """Return box size in `(length, width)[m]`.

    Valid name is following.
    * [vehicle, bus, pedestrian, cyclist, unknown]

    Args:
    ----
        name (str): _description_

    Returns:
    -------
        tuple[float, float]: _description_

    """
    return BoxSize.from_str(name).value
