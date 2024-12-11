from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final

import numpy as np
from typing_extensions import Self

if TYPE_CHECKING:
    from awml_pred.typing import ArrayLike, ArrayShape, NDArray, NDArrayBool, NDArrayF32, NDArrayI32, NDArrayStr

__all__ = ["State", "Trajectory", "AgentTrack"]


@dataclass
class AgentStateBase(ABC):
    """An abstract base class of `State`, `Trajectory` and `AgentTrack`."""

    # NOTE: For the 1DArray indices must be a list.
    XYZ_IDX: ClassVar[list[int]] = [0, 1, 2]
    XY_IDX: ClassVar[list[int]] = [0, 1]
    SIZE_IDX: ClassVar[list[int]] = [3, 4, 5]
    YAW_IDX: ClassVar[int] = 6
    VEL_IDX: ClassVar[list[int]] = [7, 8]
    IS_VALID_IDX: ClassVar[int] = 9

    num_dim: ClassVar[int] = 10

    @property
    @abstractmethod
    def xyz(self) -> NDArray: ...

    @xyz.setter
    @abstractmethod
    def xyz(self, xyz: ArrayLike) -> None: ...

    @property
    @abstractmethod
    def xy(self) -> NDArray: ...

    @xy.setter
    @abstractmethod
    def xy(self, xy: ArrayLike) -> None: ...

    @property
    @abstractmethod
    def size(self) -> NDArray: ...

    @size.setter
    @abstractmethod
    def size(self, size: ArrayLike) -> None: ...

    @property
    @abstractmethod
    def yaw(self) -> NDArray | float: ...

    @yaw.setter
    @abstractmethod
    def yaw(self, yaw: ArrayLike | float) -> None: ...

    @property
    @abstractmethod
    def vxy(self) -> NDArray: ...

    @vxy.setter
    @abstractmethod
    def vxy(self, vxy: ArrayLike) -> None: ...

    @property
    @abstractmethod
    def is_valid(self) -> NDArrayBool | bool: ...

    @property
    @abstractmethod
    def shape(self) -> ArrayShape: ...


@dataclass
class State(AgentStateBase):
    """A class represents agent state at the specific time.

    Attributes
    ----------
        state (NDArray): `(x, y, z, length, width, height, yaw, vx, vy, is_valid)` in shape (D,).

    """

    state: NDArray

    @property
    def xyz(self) -> NDArray:
        """Return 3D position.

        Returns
        -------
            NDArray: 3D position in shape (3,).

        """
        return self.state[self.XYZ_IDX]

    @xyz.setter
    def xyz(self, xyz: ArrayLike) -> None:
        self.state[self.XYZ_IDX] = xyz

    @property
    def xy(self) -> NDArray:
        """Return 3D position.

        Returns
        -------
            NDArray: 3D position in shape (2,).

        """
        return self.state[self.XY_IDX]

    @xy.setter
    def xy(self, xy: ArrayLike) -> None:
        self.state[self.XY_IDX] = xy

    @property
    def size(self) -> NDArray:
        """Return 3D position.

        Returns
        -------
            NDArray: 3D position in shape (3,).

        """
        return self.state[self.SIZE_IDX]

    @size.setter
    def size(self, size: ArrayLike) -> None:
        self.state[self.SIZE_IDX] = size

    @property
    def yaw(self) -> float:
        """Return yaw in [rad].

        Returns
        -------
            NDArray: Yaw in [rad].

        """
        return self.state[self.YAW_IDX]

    @yaw.setter
    def yaw(self, yaw: float) -> None:
        self.state[self.YAW_IDX] = yaw

    @property
    def vxy(self) -> NDArray:
        """Return velocity (vx, vy).

        Returns
        -------
            NDArray: (vx, y)

        """
        return self.state[self.YAW_IDX]

    @vxy.setter
    def vxy(self, vxy: ArrayLike) -> None:
        self.state[self.VEL_IDX] = vxy

    @property
    def is_valid(self) -> bool:
        """Whether state is valid.

        Returns
        -------
            bool: True if state is valid.

        """
        return self.state[self.IS_VALID_IDX]

    @property
    def shape(self) -> ArrayShape:
        """Shape of state array.

        Returns
        -------
            ArrayShape: _description_

        """
        return self.state.shape

    def as_array(self) -> NDArray:
        """Return state as NDArray.

        Returns
        -------
            NDArray: Array of state.

        """
        return self.state.copy()


@dataclass
class Trajectory(AgentStateBase):
    """A class represents agent trajectory.

    Attributes
    ----------
        waypoints (NDArray): Trajectory waypoints in shape (..., D).

    """

    waypoints: NDArray

    def __post_init__(self) -> None:
        assert self.waypoints.shape[-1] == self.num_dim

    def at(self, at: int) -> State | Trajectory:
        """Return the state or trajectory at the specified index in `axis=0`.

        Args:
        ----
            at (int): Index of state.

        Returns:
        -------
            State | Trajectory: If the number of dimensions of waypoints is greater than 2,
                returns `Trajectory` instance. Otherwise, returns `State` instance.

        """
        min_ndim: Final[int] = 2
        if self.waypoints.ndim > min_ndim:
            return Trajectory(self.waypoints[at].copy())
        return State(self.waypoints[at].copy())

    @property
    def xyz(self) -> NDArray:
        """3D positions of trajectory.

        Returns
        -------
            NDArray: 3D positions in shape (M, 3).

        """
        return self.waypoints[..., self.XYZ_IDX]

    @xyz.setter
    def xyz(self, xyz: ArrayLike) -> None:
        self.waypoints[..., self.XYZ_IDX] = xyz

    @property
    def xy(self) -> NDArray:
        """2D positions of trajectory.

        Returns
        -------
            NDArray: 2D positions in shape (M, 2).

        """
        return self.waypoints[..., self.XY_IDX]

    @xy.setter
    def xy(self, xy: ArrayLike) -> None:
        self.waypoints[..., self.XY_IDX] = xy

    @property
    def size(self) -> NDArray:
        """3D dimensions of trajectory, in the order of `(length, width, height)`.

        Returns
        -------
            NDArray: 3D positions in shape (M, 3).

        """
        return self.waypoints[..., self.SIZE_IDX]

    @size.setter
    def size(self, size: ArrayLike) -> None:
        self.waypoints[..., self.SIZE_IDX] = size

    @property
    def yaw(self) -> NDArray:
        """Yaw heading of trajectory in `rad`.

        Returns
        -------
            NDArray: Yaw heading in shape `M`.

        """
        return self.waypoints[..., self.YAW_IDX]

    @yaw.setter
    def yaw(self, yaw: ArrayLike) -> None:
        self.waypoints[..., self.YAW_IDX] = yaw

    @property
    def vxy(self) -> NDArray:
        """2D Velocities of trajectory.

        Returns
        -------
            NDArray: 2D velocities in shape (M, 2).

        """
        return self.waypoints[..., self.VEL_IDX]

    @vxy.setter
    def vxy(self, vxy: ArrayLike) -> None:
        self.waypoints[..., self.VEL_IDX] = vxy

    @property
    def is_valid(self) -> NDArrayBool:
        """Flags to represent if the state at `t` is valid.

        Returns
        -------
            NDArrayBool: Flags in shape (M,).

        """
        return self.waypoints[..., self.IS_VALID_IDX] == 1

    @property
    def shape(self) -> ArrayShape:
        """Return the shape of trajectory array.

        Returns
        -------
            ArrayShape: Shape of array.

        """
        return self.waypoints.shape

    def as_array(self) -> NDArray:
        """Return this trajectory as `NDArray`.

        Returns
        -------
            NDArray: Array of trajectory.

        """
        return self.waypoints.copy()


@dataclass
class AgentTrack(AgentStateBase):
    """Represents agent tracks.

    The state dimension is the order of `(x, y, z, length, width, height, yaw, vx, vy, is_valid)`.

    Attributes
    ----------
        scenario_id (str): Unique ID of scenario in str.
        timestamps (NDArrayF32): Sequence of timestamps corresponding to this track in [sec] in shape (T,).
        trajectory (NDArrayF32): Array of trajectory in shape (N, T, D).
        mask (NDArrayBool): Array of trajectory mask in shape (N, T).
        types (NDArrayStr): Array of agent types in shape (N,).
        ids (NDArrayI32): Array of agent unique IDs in shape (N,).

    """

    scenario_id: str
    timestamps: NDArrayF32
    trajectory: NDArrayF32
    mask: NDArrayBool
    types: NDArrayStr
    ids: NDArrayI32

    def __init__(
        self,
        scenario_id: str,
        timestamps: ArrayLike,
        trajectory: ArrayLike,
        types: ArrayLike,
        ids: ArrayLike,
    ) -> None:
        """Construct instance.

        Args:
        ----
            scenario_id (str): Unique ID of scenario in str.
            timestamps (ArrayLike): Sequence of timestamps corresponding to this track in [sec].
            trajectory (ArrayLike): Array of trajectory in shape (N, T, D).
            types (ArrayLike): Array of agent types in shape (N,).
            ids (ArrayLike): Array of agent unique IDs in shape (N,).

        """
        self.scenario_id = scenario_id
        self.timestamps = np.asarray(timestamps, dtype=np.float32)
        self.trajectory = np.asarray(trajectory, dtype=np.float32)
        self.mask = self.trajectory[..., -1] > 0
        self.types = np.asarray(types, dtype=np.str_)
        self.ids = np.asarray(ids, dtype=np.int32)

        self.__post_init__()

    def __post_init__(self) -> None:
        assert self.trajectory.shape[-1] == self.num_dim

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Construct a instance from a dict data.

        Args:
        ----
            data (dict): Dict data of `AgentTrack`.

        Returns:
        -------
            AgentTrack: Constructed instance.

        """
        scenario_id: str = data["scenario_id"]
        timestamps: NDArrayF32 = np.asarray(data["timestamps"], dtype=np.float32)
        trajectory: NDArrayF32 = np.asarray(data["trajectory"], dtype=np.float32)
        types: NDArrayStr = np.asarray(data["types"], dtype=np.str_)
        ids: NDArrayI32 = np.asarray(data["ids"], dtype=np.int32)
        return cls(scenario_id, timestamps, trajectory, types, ids)

    def at(self, at: int, axis: int = 0) -> Trajectory:
        """Return trajectory at n-th agent or t-th time.

        Args:
        ----
            at (int): Index of trajectory.
            axis (int, optional): Axis number 0 or 1. Defaults to 0.

        Returns:
        -------
            Trajectory: Extracted trajectory.

        """
        if axis == 0:
            return Trajectory(self.trajectory[at, ...].copy())
        elif axis == 1:
            return Trajectory(self.trajectory[:, at, ...].copy())
        else:
            msg = "Index is out of range."
            raise ValueError(msg)

    @property
    def xyz(self) -> NDArray:
        """3D positions of trajectory.

        Returns
        -------
            NDArray: 3D positions in shape (N, T, 3).

        """
        return self.trajectory[..., self.XYZ_IDX]

    @xyz.setter
    def xyz(self, xyz: NDArray) -> None:
        self.trajectory[..., self.XYZ_IDX] = xyz

    @property
    def xy(self) -> NDArray:
        """2D positions of trajectory.

        Returns
        -------
            NDArray: 2D positions in shape (N, T, 2).

        """
        return self.trajectory[..., self.XY_IDX]

    @xy.setter
    def xy(self, xy: NDArray) -> None:
        self.trajectory[..., self.XY_IDX] = xy

    @property
    def size(self) -> NDArray:
        """3D dimensions of trajectory, in the order of `(length, width, height)`.

        Returns
        -------
            NDArray: 3D positions in shape (N, T, 3).

        """
        return self.trajectory[..., self.SIZE_IDX]

    @size.setter
    def size(self, size: ArrayLike) -> None:
        self.trajectory[..., self.SIZE_IDX] = size

    @property
    def yaw(self) -> NDArray:
        """Yaw heading of trajectory in `rad`.

        Returns
        -------
            NDArray: Yaw heading in shape (N, T).

        """
        return self.trajectory[..., self.YAW_IDX]

    @yaw.setter
    def yaw(self, yaw: ArrayLike) -> None:
        self.trajectory[..., self.YAW_IDX] = yaw

    @property
    def vxy(self) -> NDArray:
        """2D Velocities of trajectory.

        Returns
        -------
            NDArray: 2D velocities in shape (N, T, 2).

        """
        return self.trajectory[..., self.VEL_IDX]

    @vxy.setter
    def vxy(self, vxy: ArrayLike) -> None:
        self.trajectory[..., self.VEL_IDX] = vxy

    @property
    def is_valid(self) -> NDArrayBool:
        """Flags to represent if the state at `t` is valid.

        Returns
        -------
            NDArray: Flags in shape (N, T).

        """
        return self.trajectory[..., self.IS_VALID_IDX] == 1

    @is_valid.setter
    def is_valid(self, is_valid: ArrayLike) -> None:
        self.trajectory[..., self.IS_VALID_IDX] = is_valid

    @property
    def shape(self) -> ArrayShape:
        """Return the shape of trajectory array.

        Returns
        -------
            ArrayShape: Shape of array.

        """
        return self.trajectory.shape

    def get_trajectory(self) -> Trajectory:
        """Return the trajectory as the `Trajectory` instance.

        Returns
        -------
            Trajectory: `Trajectory` instance.

        """
        return Trajectory(self.trajectory.copy())
