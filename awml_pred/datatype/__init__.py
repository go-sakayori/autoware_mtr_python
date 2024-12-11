from .agents import AgentType, ArgoverseAgent, T4Agent, WaymoAgent
from .base import ContextType, LabelType
from .dataset import DatasetName
from .polylines import (
    ArgoverseLane,
    ArgoversePolyline,
    ArgoverseRoadEdge,
    ArgoverseRoadLine,
    BoundaryType,
    LaneType,
    PolylineType,
    SignalType,
    T4Lane,
    T4Polyline,
    T4RoadEdge,
    T4RoadLine,
    WaymoLane,
    WaymoPolyline,
    WaymoRoadEdge,
    WaymoRoadLine,
    WaymoSignal,
)

__all__ = (
    "AgentType",
    "ArgoverseAgent",
    "ArgoverseLane",
    "ArgoversePolyline",
    "ArgoverseRoadEdge",
    "ArgoverseRoadLine",
    "BoundaryType",
    "ContextType",
    "LabelType",
    "LaneType",
    "PolylineType",
    "SignalType",
    "T4Agent",
    "T4Lane",
    "T4Polyline",
    "T4RoadEdge",
    "T4RoadLine",
    "WaymoLane",
    "WaymoAgent",
    "WaymoPolyline",
    "WaymoRoadEdge",
    "WaymoRoadLine",
    "WaymoSignal",
    "get_focal_agent_types",
    "DatasetName",
)


def get_focal_agent_types() -> list[str]:
    """Return the list of agent type names that can be focal agents.

    Returns
    -------
        list[str]: List of focal agent type names.

    """
    # TODO(ktro2828): refactor code
    waymo_focal_types = [
        WaymoAgent.TYPE_VEHICLE.name,
        WaymoAgent.TYPE_PEDESTRIAN.name,
        WaymoAgent.TYPE_CYCLIST.name,
    ]
    av_focal_types = [
        ArgoverseAgent.VEHICLE.name,
        ArgoverseAgent.PEDESTRIAN.name,
        ArgoverseAgent.MOTORCYCLIST.name,
        ArgoverseAgent.MOTORCYCLIST.name,
        ArgoverseAgent.CYCLIST.name,
        ArgoverseAgent.BUS.name,
    ]

    return waymo_focal_types + av_focal_types
