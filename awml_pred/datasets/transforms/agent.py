from awml_pred.common import TRANSFORMS
from awml_pred.dataclass import AgentTrack, AWMLAgentScenario, Trajectory
from awml_pred.ops import rotate_points_along_z

__all__ = ("TargetCentricAgent",)


@TRANSFORMS.register()
class TargetCentricAgent:
    """Transform agent trajectory from world coords to target centric coords.

    Required Keys:
    -------------
        scenario (AWMLAgentScenario): `AWMLAgentScenario` instance.
        predict_all_agents (bool): Whether to predict all agents.

    Updated Keys:
    ------------
        agent_past (Trajectory): (B, N, Tp, D)
        agent_future (Trajectory): (B, N, Tf, D)
    """

    def __init__(self, *, with_velocity: bool = True) -> None:
        """Construct instance.

        Args:
        ----
            with_velocity (bool, optional): Whether to transform including velocity. Defaults to True.

        """
        self.with_velocity = with_velocity

    def _do_transform(self, agent_track: AgentTrack, target_current: Trajectory) -> Trajectory:
        """Transform the agent track to target centric coordinates.

        Args:
        ----
            agent_track (AgentTrack): `AgentTrack` instance.
            target_current (Trajectory): `Trajectory` instance of targets.

        Returns:
        -------
            Trajectory: Transformed trajectory.

        """
        num_agent, num_time, num_dim = agent_track.shape
        num_target, _ = target_current.shape
        agent = agent_track.get_trajectory().as_array()
        batch_agent = Trajectory(agent.reshape(1, num_agent, num_time, num_dim).repeat(num_target, axis=0))
        batch_agent.xyz -= target_current.xyz[:, None, None]
        batch_agent.xy = rotate_points_along_z(batch_agent.xy.reshape(num_target, -1, 2), -target_current.yaw).reshape(
            num_target,
            num_agent,
            num_time,
            2,
        )
        batch_agent.yaw -= target_current.yaw[:, None, None]

        if self.with_velocity:
            batch_agent.vxy = rotate_points_along_z(
                points=batch_agent.vxy.reshape(num_target, -1, 2),
                angle=-target_current.yaw,
            ).reshape(num_target, num_agent, num_time, 2)

        return batch_agent

    def __call__(self, info: dict) -> dict:
        """Run transformation.

        Args:
        ----
            info (dict): Source info.

        Returns:
        -------
            dict: Output info.

        """
        scenario: AWMLAgentScenario = info["scenario"]
        predict_all_agents: bool = info["predict_all_agents"]
        target_current = (
            scenario.get_current() if predict_all_agents else scenario.get_current(at=scenario.target_indices)
        )
        batch_agent_past = self._do_transform(scenario.past_track, target_current)
        batch_agent_future = self._do_transform(scenario.future_track, target_current)

        batch_agent_past.waypoints[~batch_agent_past.is_valid] = 0
        batch_agent_future.waypoints[~batch_agent_future.is_valid] = 0

        info.update(
            {
                "agent_past": batch_agent_past,
                "agent_future": batch_agent_future,
            },
        )
        return info
