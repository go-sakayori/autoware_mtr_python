from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from awml_pred.common import TRANSFORMS
from awml_pred.datasets.transforms import TargetCentricAgent, TargetCentricPolyline

if TYPE_CHECKING:
    from awml_pred.dataclass import AWMLAgentScenario, Trajectory
    from awml_pred.typing import NDArrayBool, NDArrayF32

__all__ = ("MtrAgentEmbed", "MtrPolylineEmbed")


@TRANSFORMS.register()
class MtrAgentEmbed:
    """Embed agent state for MTR, which includes positions, sizes, timestamps,.

    headings, velocities and accelerations.

    Required Keys:
    -------------
        scenario (AWMLAgentScenario): `AWMLAgentScenario` instance.
        agent_past (Trajectory): `Trajectory` instance of past track in shape (B, N, T, D).
        predict_all_agents (bool): Whether to predict all agents.
        agent_types (list[str]): List of target agent types.

    Updated Keys:
    ------------
        agent_past (NDArray): (B, N, T, D')
        agent_past_mask (NDArray): (B, N, T)
        agent_past_pos (NDArray): (B, N, T, 3)
        agent_last_pos (NDArray): (B, N, 3)
    """

    def __init__(self, *, with_velocity: bool = True) -> None:
        """Construct instance.

        Args:
        ----
            with_velocity (bool, optional): Whether to transform including velocity. Defaults to True.

        """
        self.preprocess = TargetCentricAgent(with_velocity=with_velocity)

    @staticmethod
    def _load_current_position(
        scenario: AWMLAgentScenario,
        agent_past: Trajectory,
        *,
        predict_all_agents: bool = False,
    ) -> NDArrayF32:
        num_agent = scenario.num_agent if predict_all_agents else scenario.num_target
        agent_current_xyz = np.zeros((num_agent, len(scenario.types), 3), dtype=np.float32)
        for k in range(scenario.current_time_index + 1):
            is_cur_valid = agent_past.is_valid[..., k]
            agent_current_xyz[is_cur_valid] = agent_past.xyz[..., k, :][is_cur_valid]
        return agent_current_xyz

    def __call__(self, info: dict) -> dict:
        """Run transformation.

        Args:
        ----
            info (dict): Source info.

        Returns:
        -------
            dict: Output info.

        """
        info = self.preprocess(info)

        scenario: AWMLAgentScenario = info["scenario"]
        agent_past: Trajectory = info["agent_past"]
        predict_all_agents: bool = info["predict_all_agents"]

        # extract past positions and latest positions
        info["agent_past_pos"] = agent_past.xyz
        info["agent_last_pos"] = self._load_current_position(
            scenario,
            agent_past,
            predict_all_agents=predict_all_agents,
        )

        num_target, num_agent, num_time, _ = agent_past.shape
        target_indices = np.arange(num_agent) if predict_all_agents else scenario.target_indices

        # type onehot
        agent_types: list[str] = info["agent_types"]
        num_type = len(agent_types)
        type_onehot = np.zeros((num_target, num_agent, num_time, num_type + 2), dtype=np.float32)
        for i, target_type in enumerate(agent_types):
            type_onehot[:, scenario.types == target_type, :, i] = 1
        type_onehot[np.arange(num_target), target_indices, :, num_type] = 1
        type_onehot[:, scenario.ego_index, :, num_type + 1] = 1

        # time embedding
        time_embed = np.zeros((num_target, num_agent, num_time, num_time + 1), dtype=np.float32)
        time_embed[:, :, np.arange(num_time), np.arange(num_time)] = 1
        time_embed[:, :, :num_time, -1] = scenario.past_track.timestamps

        # heading embedding
        yaw_embed = np.zeros((num_target, num_agent, num_time, 2), dtype=np.float32)
        yaw_embed[..., 0] = np.sin(agent_past.yaw)
        yaw_embed[..., 1] = np.cos(agent_past.yaw)

        # accel
        # TODO: use accurate timestamp diff
        vel_diff = np.diff(agent_past.vxy, axis=2, prepend=agent_past.vxy[..., 0, :][:, :, None, :])
        accel = vel_diff / 0.1
        accel[:, :, 0, :] = accel[:, :, 1, :]

        past_embed = np.concatenate(
            (
                agent_past.xyz,
                agent_past.size,
                type_onehot,
                time_embed,
                yaw_embed,
                agent_past.vxy,
                accel,
            ),
            axis=-1,
            dtype=np.float32,
        )
        past_embed[~agent_past.is_valid] = 0

        info["agent_past"] = past_embed
        info["agent_past_mask"] = agent_past.is_valid

        return info


@TRANSFORMS.register()
class MtrPolylineEmbed:
    """Transform polylines from map coords to target centric coords.

    NOTE: current implementation returns different values compared with previous one.
        But test score is same.

    Required Keys:
    --------------
        scenario (AWMLAgentScenario): `AWMLAgentScenario` instance.
        static_map (AWMLStaticMap): `AWMLStaticMap` instance.
        predict_all_agents (bool): Whether to predict all agents.

    Updated Keys:
    -------------
        polylines (NDArrayF32): (B, K, P, D)
        polylines_mask (NDArrayBool): (B, K, P)
        polylines_center (NDArrayF32): (B, K, 3)
    """

    def __init__(
        self,
        num_polylines: int = 768,
        num_points: int = 20,
        break_distance: float = 1.0,
        center_offset: tuple[float, float] = (30.0, 0.0),
    ) -> None:
        """Construct instance.

        Args:
        ----
            num_polylines (int, optional): Max number of polylines can be contained. Defaults to 768.
            num_points (int, optional): Max number of points, which each polyline can contain. Defaults to 20.
            break_distance (float, optional): The distance threshold to separate polyline into two polylines.
                Defaults to 1.0.
            center_offset (tuple[float, float], optional): The offset position. Defaults to (30.0, 0.0).

        """
        self.preprocess = TargetCentricPolyline(
            num_polylines=num_polylines,
            num_points=num_points,
            break_distance=break_distance,
            center_offset=center_offset,
        )

    @staticmethod
    def _load_polyline_center(polyline: NDArrayF32, polyline_mask: NDArrayBool) -> NDArrayF32:
        tmp_sum = (polyline[..., :3] * polyline_mask[..., None]).sum(axis=-2, dtype=np.float32)
        polyline_center: NDArrayF32 = tmp_sum / np.clip(
            polyline_mask.sum(axis=-1, dtype=np.float32)[..., None],
            a_min=1.0,
            a_max=None,
        )
        return polyline_center.astype(np.float32)

    def __call__(self, info: dict) -> dict:
        """Run transformation.

        Args:
        ----
            info (dict): Source info.

        Returns:
        -------
            dict: Output info.

        """
        info = self.preprocess(info)

        info["polylines_center"] = self._load_polyline_center(info["polylines"], info["polylines_mask"])

        return info
