from .base import AgentType


class ArgoverseAgent(AgentType):
    """Agent types in Argoverse."""

    # Dynamic movers
    VEHICLE = 0
    PEDESTRIAN = 1
    MOTORCYCLIST = 2
    CYCLIST = 3
    BUS = 4

    # Static objects
    STATIC = 5
    BACKGROUND = 6
    CONSTRUCTION = 7
    RIDERLESS_BICYCLE = 8

    # Catch-all type for other/unknown objects
    UNKNOWN = 9

    def is_dynamic(self) -> bool:
        """Whether the object is dynamic movers.

        Returns
        -------
            bool: True if any of (VEHICLE, PEDESTRIAN, MOTORCYCLIST, CYCLIST, BUS).

        """
        return self in (
            ArgoverseAgent.VEHICLE,
            ArgoverseAgent.PEDESTRIAN,
            ArgoverseAgent.MOTORCYCLIST,
            ArgoverseAgent.CYCLIST,
            ArgoverseAgent.BUS,
        )
