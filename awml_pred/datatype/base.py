from __future__ import annotations

import logging
from abc import abstractmethod
from enum import Enum, IntEnum

from typing_extensions import Self

__all__ = ("BaseType", "ContextType", "LabelType")


class BaseType(IntEnum):
    """Base of types.

    All types must have the following enum format.
    * TYPE_NAME = TYPE_ID <int>
    """

    def as_str(self) -> str:
        """Return the type name.

        Returns
        -------
            str: Name in str.

        """
        return self.name

    @classmethod
    def from_str(cls, name: str) -> Self:
        """Construct from the name of member.

        Args:
        ----
            name (str): Name of an enum member.

        Returns:
        -------
            Self: Constructed member.

        """
        name = name.upper()
        assert name in cls.__members__, f"{name} is not in enum members of {cls.__name__}."
        return cls.__members__[name]

    @classmethod
    def from_id(cls, type_id: int) -> Self:
        """Construct from the value of member.

        Args:
        ----
            type_id (int): Value of enum member.

        Returns:
        -------
            Self: Constructed member.

        """
        for _, item in cls.__members__.items():
            if item.value == type_id:
                return item
        msg = f"{type_id} is not in enum ids."
        raise ValueError(msg)

    @classmethod
    def contains(cls, name: str) -> bool:
        """Check whether the input name is contained in members.

        Args:
        ----
            name (str): Name of enum member.

        Returns:
        -------
            bool: Whether it is contained.

        """
        return name.upper() in cls.__members__

    @classmethod
    def to_dict(cls) -> dict[str, int]:
        """Convert members to dict formatting as `{name: value}`.

        Returns
        -------
            dict[str, int]: Converted dict.

        """
        return {name: item.value for name, item in cls.__members__.items()}

    @classmethod
    def get_static_items(cls) -> list[Self]:
        """Return members that `is_dynamic()=False`.

        Returns
        -------
            list[Self]: List of static items.

        """
        return [item for item in cls if not item.is_dynamic()]

    @classmethod
    def get_dynamic_items(cls) -> list[Self]:
        """Return members that `is_dynamic()=True`.

        Returns
        -------
            list[Self]: List of dynamic items.

        """
        return [item for item in cls if item.is_dynamic()]

    @staticmethod
    @abstractmethod
    def to_context(*, as_str: bool = False) -> ContextType | str:
        """Convert the enum member to `ContextType`.

        Args:
        ----
            as_str (bool, optional): Whether to return as str. Defaults to False.

        Returns:
        -------
            ContextType | str: Converted object.

        """

    @abstractmethod
    def is_dynamic(self) -> bool:
        """Whether the item is dynamic.

        Returns
        -------
            bool: Return True if dynamic is.

        """


class ContextType(str, Enum):
    """Context types."""

    # Global context
    AGENT = "AGENT"
    POLYLINE = "POLYLINE"

    # Local context
    EGO = "EGO"
    FOCAL_AGENT = "FOCAL_AGENT"
    OTHER_AGENT = "OTHER_AGENT"

    LANE = "LANE"  # = CENTERLINE
    ROADLINE = "ROADLINE"
    ROADEDGE = "ROADEDGE"
    CROSSWALK = "CROSSWALK"
    SIGNAL = "SIGNAL"

    # Catch all contexts
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, name: str) -> ContextType:
        """Construct the member from name.

        Args:
        ----
            name (str): Name of member in str.

        Returns:
        -------
            ContextType: Constructed member.

        """
        name = name.upper()
        if name not in cls.__members__:
            msg = f"{name} is not in enum members of {cls.__name__}, UNKNOWN is used."
            logging.warning(msg)
            return cls.UNKNOWN
        else:
            return cls.__members__[name]


class LabelType(str, Enum):
    """Label types."""

    # Agents
    EGO = "EGO"
    VEHICLE = "VEHICLE"
    LARGE_VEHICLE = "LARGE_VEHICLE"
    PEDESTRIAN = "PEDESTRIAN"
    MOTORCYCLIST = "MOTORCYCLIST"
    CYCLIST = "CYCLIST"

    # Polylines
    LANE = "LANE"  # = CENTERLINE
    ROADLINE = "ROADLINE"
    ROADEDGE = "ROADEDGE"
    CROSSWALK = "CROSSWALK"
    SIGNAL = "SIGNAL"

    # Catch all labels
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, name: str) -> LabelType:
        """Construct the member from name.

        Args:
        ----
            name (str): Name of member in str.

        Returns:
        -------
            LabelType: Constructed member.

        """
        name = name.upper()
        if name not in cls.__members__:
            msg = f"{name} is not in enum members of {cls.__members__}, UNKNOWN is used."
            logging.warning(msg)
            return cls.UNKNOWN
        else:
            return cls.__members__[name]
