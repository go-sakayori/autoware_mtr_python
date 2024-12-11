from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence, TypeVar, overload

import numpy as np
from typing_extensions import Self

from awml_pred.typing import ArrayLike, NDArray, NDArrayBool, NDArrayF32, NDArrayI32, NDArrayStr

from .agent import AgentTrack, State, Trajectory

__all__ = ["TargetInfo", "AWMLAgentScenario"]


AgentLike = TypeVar("AgentLike", AgentTrack, Trajectory, State, NDArrayF32)


@dataclass
class TargetInfo:
    """Represents target agent information in scenario.

    Attributes
    ----------
        indices (NDArray): Indices of targets of associated track.
        difficulties (NDArray): Target difficulties.

    """

    indices: NDArrayI32
    difficulties: NDArrayI32

    def __post_init__(self) -> None:
        assert len(self.indices) == len(
            self.difficulties,
        ), f"Number of items must be same, but got {len(self.indices)} and {len(self.difficulties)}"
        self.indices = np.asarray(self.indices, dtype=np.int32)
        self.difficulties = np.asarray(self.difficulties, dtype=np.int32)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Construct a instance from a dict data.

        Args:
        ----
            data (dict): Dict data of `TargetInfo`.

        Returns:
        -------
            TargetInfo: Constructed instance.

        """
        indices = data["indices"]
        difficulties = data["difficulties"]
        return cls(indices, difficulties)

    def __len__(self) -> int:
        return len(self.indices)


@dataclass
class AWMLAgentScenario:
    """Represents scenario information."""

    scenario_id: str
    ego_index: int
    current_time_index: int
    timestamps: NDArrayF32
    trajectory: NDArrayF32
    mask: NDArrayBool
    types: NDArrayStr
    ids: NDArrayI32
    past_track: AgentTrack
    future_track: AgentTrack
    target_info: TargetInfo

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Construct a instance from a dict data.

        Args:
        ----
            data (dict): Dict data of `AWMLAgentScenario`.

        Returns:
        -------
            AWMLAgentScenario: Constructed instance.

        """
        scenario_id: str = data["scenario_id"]
        ego_index: int = data["ego_index"]
        current_time_index: int = data["current_time_index"]
        timestamps: NDArrayF32 = np.asarray(data["timestamps"], dtype=np.float32)

        # all tracks
        trajectory: NDArrayF32 = np.asarray(data["trajectory"], dtype=np.float32)
        mask: NDArrayBool = trajectory[..., Trajectory.IS_VALID_IDX] > 0
        types: NDArrayStr = np.asarray(data["types"])
        ids: NDArrayI32 = np.asarray(data["ids"], dtype=np.int32)

        # target tracks
        target_info = TargetInfo.from_dict(data["target_info"])

        past_track = AgentTrack(
            scenario_id=scenario_id,
            timestamps=timestamps[: current_time_index + 1],
            trajectory=trajectory[:, : current_time_index + 1],
            types=types,
            ids=ids,
        )

        future_track = AgentTrack(
            scenario_id=scenario_id,
            timestamps=timestamps[current_time_index + 1 :],
            trajectory=trajectory[:, current_time_index + 1 :],
            types=types,
            ids=ids,
        )

        return cls(
            scenario_id,
            ego_index,
            current_time_index,
            timestamps,
            trajectory,
            mask,
            types,
            ids,
            past_track,
            future_track,
            target_info,
        )

    def filter(self, mask: NDArrayBool) -> AWMLAgentScenario:
        """Filter out scenario by the specified mask.

        Note that `mask` must have the shape of (NumAgent,).

        Args:
        ----
            mask (NDArrayBool): Mask array.

        Returns:
        -------
            AWMLAgentScenario: New instance applied the mask.

        """
        assert len(mask) == self.num_agent
        assert mask.ndim == 1

        valid_trajectory = self.trajectory[mask]
        valid_mask = self.mask[mask]
        valid_types = self.types[mask]
        valid_ids = self.ids[mask]

        valid_idx_cnt = mask.cumsum(axis=0)
        valid_ego_index = int(valid_idx_cnt[self.ego_index] - 1)

        target_mask = mask[self.target_indices]
        valid_target_indices = (valid_idx_cnt[self.target_indices] - 1)[target_mask]
        valid_target_difficulties = self.target_difficulties[mask[self.target_indices]]

        target_info = TargetInfo(valid_target_indices, valid_target_difficulties)

        valid_past_track = AgentTrack(
            scenario_id=self.scenario_id,
            timestamps=self.timestamps[: self.current_time_index + 1],
            trajectory=valid_trajectory[:, : self.current_time_index + 1],
            types=valid_types,
            ids=valid_ids,
        )

        valid_future_track = AgentTrack(
            scenario_id=self.scenario_id,
            timestamps=self.timestamps[self.current_time_index + 1 :],
            trajectory=valid_trajectory[:, self.current_time_index + 1 :],
            types=valid_types,
            ids=valid_ids,
        )

        return AWMLAgentScenario(
            scenario_id=self.scenario_id,
            ego_index=valid_ego_index,
            current_time_index=self.current_time_index,
            timestamps=self.timestamps,
            trajectory=valid_trajectory,
            mask=valid_mask,
            types=valid_types,
            ids=valid_ids,
            past_track=valid_past_track,
            future_track=valid_future_track,
            target_info=target_info,
        )

    def filter_by_type(self, types: Sequence[str]) -> AWMLAgentScenario:
        """Filter out agents by their types.

        This method filters out agents whose types are included in input types.

        Args:
        ----
            types (Sequence[str]): Sequence of type names.

        Returns:
        -------
            AWMLAgentScenario: New instance applied type mask.

        """
        mask: NDArrayBool = np.array([t in types for t in self.types], dtype=np.bool_)
        return self.filter(mask)

    def filter_by_past_mask(self) -> AWMLAgentScenario:
        """Apply past mask and return a new AWMLAgentScenario instance.

        This method filters out agents whose past time series tracks are all invalid.

        Returns
        -------
            AWMLAgentScenario: New instance applied past mask.

        """
        # mask agents if all past frames are invalid
        mask1: NDArrayBool = self.past_track.is_valid.any(axis=-1)

        # mask agents if the latest frame is invalid
        mask2: NDArrayBool = self.past_track.at(-1, axis=1).is_valid

        mask = np.bitwise_and(mask1, mask2)

        return self.filter(mask)

    @property
    def num_agent(self) -> int:
        """Return the number of all agents.

        Returns
        -------
            int: The number of all agents.

        """
        assert len(self.types) == len(self.ids)
        return len(self.types)

    @property
    def num_target(self) -> int:
        """Return the number of target agents.

        Returns
        -------
            int: The number of target agents.

        """
        return len(self.target_info)

    @property
    def target_types(self) -> NDArrayStr:
        """Return target agent type names.

        Returns
        -------
            NDArrayStr: Target agent type names.

        """
        return self.types[self.target_indices]

    @property
    def target_ids(self) -> NDArrayStr:
        """Return target agent ids.

        Returns
        -------
            NDArrayStr: Target agent ids.

        """
        return self.ids[self.target_indices]

    @property
    def target_indices(self) -> NDArrayI32:
        """Return indices of target agents.

        Returns
        -------
            NDArrayI32: Indices of target agents.

        """
        return self.target_info.indices

    @property
    def target_difficulties(self) -> NDArrayI32:
        """Return target difficulties.

        Returns
        -------
            NDArrayI32: Target difficulties.

        """
        return self.target_info.difficulties

    def as_dict(self) -> dict:
        """Convert the instance to a dict.

        Returns
        -------
            dict: Converted data.

        """
        return asdict(self)

    @overload
    def get_current(self) -> Trajectory:
        """Return the current trajectory of all agents.

        Returns
        -------
            Trajectory: `Trajectory` instance.

        """

    @overload
    def get_current(self, at: int) -> State:
        """Return the current state by specifying a single agent.

        Args:
        ----
            at (int): Index of an agent.

        Returns:
        -------
            State: `State` instance.

        """

    @overload
    def get_current(self, at: ArrayLike) -> Trajectory:
        """Return the current trajectory by specifying agents indices.

        Args:
        ----
            at (ArrayLike): Sequence of agents indices.

        Returns:
        -------
            Trajectory: `Trajectory` instance.

        """

    def get_current(self, at: int | ArrayLike | None = None) -> AgentLike:
        """Return the current state or trajectory."""
        current_trajectory = self.past_track.at(self.current_time_index, axis=1)
        if at is None:
            return current_trajectory
        elif isinstance(at, int) or (hasattr(at, "len") and len(at) == 1):
            return State(current_trajectory.waypoints[at].copy())
        else:
            return Trajectory(current_trajectory.waypoints[at].copy())

    @overload
    def get_ego(self) -> Trajectory:
        """Return the ego trajectory in shape (T, D).

        Returns
        -------
            Trajectory: `Trajectory` instance.

        """

    @overload
    def get_ego(self, at: int) -> State:
        """Return the state at the specified timestamp.

        Args:
        ----
            at (int): Timestamp index.

        Returns:
        -------
            State: `State` instance.

        """

    def get_ego(self, at: int | None = None) -> AgentLike:
        """Return the ego track, which has the shape `(T, D)`."""
        waypoints: NDArray = self.trajectory[self.ego_index].copy()
        return Trajectory(waypoints) if at is None else State(waypoints[at])  # at time t

    def get_ego_past(self, since: int | None = None) -> Trajectory:
        """Return the past trajectory of ego.

        Args:
        ----
            since (int | None, optional): If specified returns trajectory from `since` to the current timestamp.
                Otherwise returns whole time-length until the current timestamp. Defaults to None.

        Returns:
        -------
            Trajectory: `Trajectory` instance in shape (Tp, D).

        """
        waypoints = self.past_track.trajectory[self.ego_index].copy()
        return Trajectory(waypoints) if since is None else Trajectory(waypoints[since:])

    def get_ego_future(self, until: int | None = None) -> Trajectory:
        """Return the future trajectory of ego.

        Args:
        ----
            until (int | None, optional): If specified returns trajectory from the current timestamp to `until`.
                Otherwise returns whole time-length since the current timestamp. Defaults to None.

        Returns:
        -------
            Trajectory: `Trajectory` instance in shape (Tf, D).

        """
        waypoints = self.future_track.trajectory[self.ego_index].copy()
        if until is not None:
            return Trajectory(waypoints[:until])
        return Trajectory(waypoints)

    @overload
    def get_target(self, *, as_array: bool = False) -> AgentTrack | NDArray:
        """Return target track or trajectory as an array.

        Args:
        ----
            as_array (bool, optional): If `True` returns NDArray in shape (B, T, D). Defaults to False.

        Returns:
        -------
            AgentTrack | NDArray: If `as_array=False`, returns as `AgentTrack` otherwise `NDArray`.

        """

    @overload
    def get_target(self, at: int, axis: int = 1) -> Trajectory:
        """Return target trajectory at b-th agent or t-th time.

        Args:
        ----
            at (int): Index pointing the order of a target or timestamp.
            axis (int, optional): If `axis=0` returns b-th agent's trajectory in shape (T, D).
                If `axis=1` returns trajectory at t-th timestamp in shape (B, D).
                Defaults to 1.

        Returns:
        -------
            Trajectory: Trajectory instance.

        """

    def get_target(self, *, as_array: bool = False, at: int | None = None, axis: int = 1) -> AgentLike:
        """Return target tracks."""
        if at is not None:
            if axis == 0:  # at target n
                waypoints = self.trajectory[self.target_indices][at]
            elif axis == 1:  # at time t
                waypoints = self.trajectory[self.target_indices, at]
            else:
                msg = f"axis must be 0 or 1, but got {axis}"
                raise ValueError(msg)
            return Trajectory(waypoints.copy())
        if as_array:
            return self.trajectory[self.target_indices].copy()
        else:
            return AgentTrack(
                scenario_id=self.scenario_id,
                timestamps=self.timestamps.copy(),
                trajectory=self.trajectory[self.target_indices].copy(),
                types=self.types[self.target_indices].copy(),
                ids=self.ids[self.target_indices].copy(),
            )

    def get_target_past(self, since: int | None = None) -> Trajectory:
        """Return the past trajectory of targets.

        Args:
        ----
            since (int | None, optional): If specified returns trajectory from `since` to the current timestamp.
                Otherwise returns whole time-length until the current timestamp. Defaults to None.

        Returns:
        -------
            Trajectory: Trajectory instance in shape (B, Tp, D).

        """
        waypoints = self.past_track.trajectory[self.target_indices].copy()
        return Trajectory(waypoints) if since is None else Trajectory(waypoints[since:])

    def get_target_future(self, until: int | None = None) -> Trajectory:
        """Return the future trajectory of targets.

        Args:
        ----
            until (int | None, optional): If specified returns trajectory from the current timestamp to `until`.
                Otherwise returns whole time-length since the current timestamp. Defaults to None.

        Returns:
        -------
            Trajectory: Trajectory instance in shape (B, Tf, D).

        """
        waypoints = self.future_track.trajectory[self.target_indices].copy()
        return Trajectory(waypoints) if until is None else Trajectory(waypoints[:until])
