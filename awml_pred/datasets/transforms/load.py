from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

from awml_pred.common import TRANSFORMS, load_pkl
from awml_pred.dataclass import Trajectory

if TYPE_CHECKING:
    from awml_pred.dataclass import AWMLAgentScenario
    from awml_pred.typing import NDArrayBool, NDArrayF32, NDArrayI32


__all__ = ("LoadFutureAnnotation", "LoadIntentionPoint")


@TRANSFORMS.register()
class LoadFutureAnnotation:
    """Load future annotation state.

    If input future tensor shape is (N, T, D), GT tensors become same shape with future tensors.

    Required Keys:
    --------------
        scenario (AWMLAgentScenario): `AWMLAgentScenario` instance.
        agent_future (Trajectory | NDArrayF32): Agent future trajectory in shape (B, N, T, D) or (N, T, D).
        predict_all_agents (bool): Whether to predict all agents.

    Update Keys:
    ------------
        agent_future (Trajectory): (B, N, T, 6) or (N, T, 6) in the order of (x, y, vx, vy, cos, sin)
            if `self.with_yaw=True`, otherwise (B, N, T, 4) or (N, T, 4) in the order of (x, y, vx, vy).
        gt_trajs (NDArrayF32): GT trajectory in shape (B, T, 6) or (N, T, 6) if `self.with_yaw=True`,
            otherwise (B, T, 4) or (N, T, 4).
        gt_trajs_mask (NDArrayBool): GT trajectory mask in shape (B, T) or (N, T).
        gt_target_index (NDArrayI32): Indices of GTs associated with the future track in shape (B,) or (N,).
    """

    def __init__(self, *, with_yaw: bool = False) -> None:
        """Construct instance.

        Args:
        ----
            with_yaw (bool, optional): Indicates whether to contain yaw information into GT. Defaults to False.

        """
        self.with_yaw = with_yaw

    def _load_batch_gt(
        self,
        agent_future: NDArrayF32,
        agent_future_mask: NDArrayBool,
        target_index: NDArrayI32,
    ) -> tuple[NDArrayF32, NDArrayBool, NDArrayI32]:
        """Load ground truth trajectory, mask and indices.

        Args:
        ----
            agent_future (NDArrayF32): Future agent trajectory, in shape (B, N, T, D).
            agent_future_mask (NDArrayBool): Mask of future agent trajectory, in shape (B, N, T).
            target_index (NDArrayI32): Indices of target agents.

        Returns:
        -------
            tuple[NDArrayF32, NDArrayBool, NDArrayI32]: GT trajectory, mask and indices.

        """
        num_batch, *_ = agent_future.shape
        batch_idx = np.arange(num_batch)
        gt_trajs: NDArrayF32 = agent_future[batch_idx, target_index]
        gt_trajs_mask: NDArrayBool = agent_future_mask[batch_idx, target_index]
        gt_trajs[~gt_trajs_mask] = 0

        num_targets, num_future_times = gt_trajs_mask.shape
        gt_target_index: NDArrayI32 = np.zeros((num_targets), dtype=np.int32)
        for k in range(num_future_times):
            cur_valid_mask = gt_trajs_mask[:, k] > 0
            gt_target_index[cur_valid_mask] = k

        return gt_trajs, gt_trajs_mask, gt_target_index

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

        if info.get("agent_future") is None:
            agent_future: Trajectory = scenario.future_track.get_trajectory()
        else:
            agent_future: Trajectory | NDArrayF32 = info["agent_future"]

        items: list[NDArrayF32] = []
        if isinstance(agent_future, Trajectory):
            agent_future_mask: NDArrayBool = agent_future.is_valid
            agent_future.waypoints[~agent_future_mask] = 0
            items.extend((agent_future.xy, agent_future.vxy))
            if self.with_yaw:
                cos = np.cos(agent_future.yaw)[..., None]
                sin = np.sin(agent_future.yaw)[..., None]
                items.extend((cos, sin))
        else:
            agent_future_mask: NDArrayBool = info["agent_future_mask"]
            agent_future[~agent_future_mask] = 0
            items.extend((agent_future[..., Trajectory.XY_IDX], agent_future[..., Trajectory.VEL_IDX]))
            if self.with_yaw:
                cos = np.cos(agent_future[..., Trajectory.YAW_IDX])[..., None]
                sin = np.sin(agent_future[..., Trajectory.YAW_IDX])[..., None]
                items.extend((cos, sin))
        agent_future: NDArrayF32 = np.concatenate(items, axis=-1, dtype=np.float32)

        dim3d: Final[int] = 3
        dim4d: Final[int] = 4
        assert agent_future.ndim in (dim3d, dim4d), f"Expected 3 or 4d tensor, but got {agent_future.ndim}"

        target_indices = np.arange(scenario.num_agent) if predict_all_agents else scenario.target_indices
        if agent_future.ndim == dim4d:  # (B, N, T, D)
            gt_trajs, gt_trajs_mask, gt_target_index = self._load_batch_gt(
                agent_future,
                agent_future_mask,
                target_indices,
            )
        else:  # (N, T, D)
            gt_trajs = agent_future[target_indices].copy()
            gt_trajs_mask = agent_future_mask[target_indices].copy()
            gt_target_index = target_indices.copy()

        ret_info = info.copy()
        ret_info["agent_future"] = agent_future
        ret_info["agent_future_mask"] = agent_future_mask > 0
        ret_info["gt_trajs"] = gt_trajs
        ret_info["gt_trajs_mask"] = gt_trajs_mask > 0
        ret_info["gt_target_index"] = gt_target_index

        return ret_info


@TRANSFORMS.register()
class LoadIntentionPoint:
    """Load intention points from pkl file.

    Required Keys:
    --------------
        scenario (AWMLAgentScenario)
        predict_all_agents (bool): Whether to predict all agents.

    Update Keys:
    ------------
        intention_points (NDArrayF32): if `only_target=True` (B, M, 2), otherwise (N, M, 2).
    """

    def __init__(self, filepath: str) -> None:
        """Construct instance.

        Args:
        ----
            filepath (str): Pickle file path.
            only_target (bool, optional): Whether to load intention points for target agents.
                Defaults to True.

        """
        self.filepath = filepath
        self.intention_point_info = load_pkl(self.filepath)

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
        target_types = scenario.types if predict_all_agents else scenario.target_types
        intention_points = np.stack([self.intention_point_info[key] for key in target_types], axis=0).astype(np.float32)

        info["intention_points"] = intention_points

        return info
