from .base import AgentType


class WaymoAgent(AgentType):
    """Agent types in Waymo."""

    TYPE_UNSET = 0
    TYPE_VEHICLE = 1
    TYPE_PEDESTRIAN = 2
    TYPE_CYCLIST = 3
    TYPE_OTHER = 4

    def is_dynamic(self) -> bool:
        """Whether the object is dynamic movers.

        Returns
        -------
            bool: True if any of (VEHICLE, PEDESTRIAN, MOTORCYCLIST, CYCLIST, BUS).

        """
        return self in (WaymoAgent.TYPE_VEHICLE, WaymoAgent.TYPE_PEDESTRIAN, WaymoAgent.TYPE_CYCLIST)
