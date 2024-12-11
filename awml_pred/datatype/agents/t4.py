from .base import AgentType


class T4Agent(AgentType):
    """Agent types in T4."""

    # Dynamic movers
    VEHICLE = 0
    PEDESTRIAN = 1
    CYCLIST = 2

    # Static objects
    STATIC = 3

    # Catch-all type for other/unknown objects
    UNKNOWN = 4

    def is_dynamic(self) -> bool:
        """Whether the object is dynamic movers.

        Returns
        -------
            bool: True if any of (VEHICLE, PEDESTRIAN, MOTORCYCLIST, CYCLIST, BUS).

        """
        return self in (T4Agent.VEHICLE, T4Agent.PEDESTRIAN, T4Agent.CYCLIST)
