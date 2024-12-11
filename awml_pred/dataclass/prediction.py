from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Final, Iterator

import numpy as np
from typing_extensions import Self

if TYPE_CHECKING:
    from awml_pred.typing import NDArrayBool, NDArrayFloat, NDArrayInt, NDArrayStr


__all__ = ("EvaluationData", "Prediction", "GroundTruth")


@dataclass(frozen=True)
class EvaluationData:
    """A dataclass to store data for evaluation.

    Attributes
    ----------
        scenario_id (str): Unique scenario identifier.
        timestamps (NDArrayFloat): Sequence of timestamps in whole scenario.
        prediction (Prediction): Prediction for the corresponding scenario.
        ground_truth (GroundTruth): Ground truth for the corresponding scenario.

    """

    scenario_id: str
    timestamps: NDArrayFloat
    prediction: Prediction
    ground_truth: GroundTruth

    def __post_init__(self) -> None:
        assert len(self.timestamps) == self.ground_truth.num_scenario_frame
        assert self.prediction.num_agent == self.ground_truth.num_agent

    @property
    def num_scenario_frame(self) -> int:
        """Return the number of scenario frames.

        Returns
        -------
            int: Scenario frame length.

        """
        return len(self.timestamps)

    @property
    def num_agent(self) -> int:
        """Return the number of agents per scenario.

        Returns
        -------
            int: Th number of agents per scenario.

        """
        return self.ground_truth.num_agent

    @property
    def num_mode(self) -> int:
        """Return the number of predicted modes.

        Returns
        -------
            int: The number of predicted modes.

        """
        return self.prediction.num_mode

    @property
    def num_future(self) -> int:
        """Return the number of predicted future frames.

        Returns
        -------
            int: The number of predicted future frames.

        """
        return self.prediction.num_future

    def as_dict(self) -> dict[str, Any]:
        """Convert the instance to a dict.

        Returns
        -------
            dict[str, Any]: Converted data.

        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Construct instance from dict data.

        Args:
        ----
            data (dict[str, Any]): Serialized dict data.

        Returns:
        -------
            Self: Instance.

        """
        scenario_id = data["scenario_id"]
        timestamps = data["timestamps"]
        prediction = Prediction.from_dict(data["prediction"])
        ground_truth = GroundTruth.from_dict(data["ground_truth"])
        return cls(scenario_id=scenario_id, timestamps=timestamps, prediction=prediction, ground_truth=ground_truth)


@dataclass
class Prediction:
    """A dataclass to represent prediction of model.

    Attributes
    ----------
        score (NDArrayFloat): Score array, in the shape of (N, M) or (M,).
        xy (NDArrayFloat): (x, y)[m] array, in shape (N, M, T, 2) or (M, T, 2).
        size (NDArrayF32): (length, width)[m] array, in the shape of (N, M, T, 2) or (M, T, 2).
            If not initialized, all elements are `np.nan`.
        yaw (NDArrayF32): Heading[rad] array, in the shape of (N, M, T) or (M, T).
            If not initialized, all elements are `np.nan`.
        vxy (NDArrayF32): (vx, vy)[m/s] array, in the shape of (N, M, T, 2) or (M, T, 2).
            If not initialized, all elements are `np.nan`.

    """

    score: NDArrayFloat
    xy: NDArrayFloat
    size: NDArrayFloat = field(default=None)
    yaw: NDArrayFloat = field(default=None)
    vxy: NDArrayFloat = field(default=None)

    def __post_init__(self) -> None:
        dim3d: Final[int] = 3
        dim4d: Final[int] = 4
        if self.xy.ndim == dim3d:
            self._num_mode, self._num_future = self.xy.shape[:-1]
            self._num_agent = 1
        elif self.xy.ndim == dim4d:
            self._num_agent, self._num_mode, self._num_future = self.xy.shape[:-1]
        else:
            msg = "Input dimension must be 3d or 4d."
            raise ValueError(msg)

        if self.size is None:
            self.size = np.full_like(self.xy, fill_value=np.nan)
        if self.yaw is None:
            self.yaw = np.full(self.xy.shape[:-1], fill_value=np.nan)
        if self.vxy is None:
            self.vxy = np.full_like(self.xy, fill_value=np.nan)

        xy_dim: Final[int] = 2
        assert self.xy.shape[-1] == xy_dim
        assert self.xy.shape == self.vxy.shape == self.size.shape

    @property
    def num_agent(self) -> int:
        """Return the number of agents.

        Returns
        -------
            int: The number of agents.

        """
        return self._num_agent

    @property
    def num_mode(self) -> int:
        """Return the number of predicted modes.

        Returns
        -------
            int: The number of predicted modes.

        """
        return self._num_mode

    @property
    def num_future(self) -> int:
        """Return the length of predicted future.

        Returns
        -------
            int: The length of predicted future.

        """
        return self._num_future

    @property
    def is_batched(self) -> bool:
        """Return True if the array shape is (N, M, T, D) not (M, T, D).

        Returns
        -------
            bool: True if the `self._num_agent > 1`.

        """
        return self.num_agent > 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Construct instance from dict data.

        Args:
        ----
            data (dict[str, Any]): Serialized dict data.

        Returns:
        -------
            Self: Instance.

        """
        return cls(**data)

    def sort_by_score(self) -> Self:
        """Sort predicted each mode by its score.

        Returns
        -------
            Self: Sorted prediction.

        """
        score = np.clip(self.score, a_min=0.0, a_max=1.0)
        indices = np.argsort(-score, axis=-1)  # (N, M) or (M,)

        # NOTE: expecting prediction shape is (N, M, T, D) or (M, T, D)
        axis: int = self.score.ndim - 1
        score = np.take_along_axis(self.score, indices, axis=axis)
        xy = np.take_along_axis(self.xy, indices[..., None, None], axis=axis)
        size = np.take_along_axis(self.size, indices[..., None, None], axis=axis) if self.size is not None else None
        yaw = np.take_along_axis(self.yaw, indices[..., None], axis=axis) if self.yaw is not None else None
        vxy = np.take_along_axis(self.vxy, indices[..., None, None], axis=axis) if self.vxy is not None else None

        return Prediction(score=score, xy=xy, size=size, yaw=yaw, vxy=vxy)

    def apply_mask(self, mask: NDArrayBool) -> Self:
        """Filter by mask.

        Args:
        ----
            mask (NDArrayBool): Mask for agent in the shape of (N,).

        Returns:
        -------
            Self: Filtered instance.

        """
        if mask.ndim != 1:
            msg = "Input mask must be 1d."
            raise ValueError(msg)

        score = self.score[mask]
        xy = self.xy[mask]
        size = self.size[mask] if self.size is not None else None
        yaw = self.yaw[mask] if self.yaw is not None else None
        vxy = self.vxy[mask] if self.vxy is not None else None

        return Prediction(score=score, xy=xy, size=size, yaw=yaw, vxy=vxy)

    def __iter__(self) -> Iterator:
        size = [None] * self.num_agent if self.size is None else self.size
        yaw = [None] * self.num_agent if self.yaw is None else self.yaw
        vxy = [None] * self.num_agent if self.vxy is None else self.vxy
        yield from zip(self.score, self.xy, size, yaw, vxy)


@dataclass
class GroundTruth:
    """A dataclass to represent.

    Attributes
    ----------
        types (NDArrayStr): Sequence of agent types.
        ids (NDArrayInt): Sequence of agent ids.
        xy (NDArrayFloat): (x, y)[m] array, in shape (N, M, T, 2) or (M, T, 2).
        size (NDArrayFloat): (length, width)[m] array, in the shape of (N, M, T, 2) or (M, T, 2).
        yaw (NDArrayFloat): Heading[rad] array, in the shape of (N, M, T) or (M, T).
        vxy (NDArrayFloat): (vx, vy)[m/s] array, in the shape of (N, M, T, 2) or (M, T, 2).
        is_valid (NDarrayBool): Flag array indicates corresponding element is valid.

    """

    types: NDArrayStr
    ids: NDArrayInt
    xy: NDArrayFloat
    size: NDArrayFloat
    yaw: NDArrayFloat
    vxy: NDArrayFloat
    is_valid: NDArrayBool

    def __post_init__(self) -> None:
        assert (
            len(self.types)
            == len(self.ids)
            == len(self.xy)
            == len(self.size)
            == len(self.yaw)
            == len(self.vxy)
            == len(self.is_valid)
        )
        self.is_valid = self.is_valid.astype(np.bool_)

    @property
    def num_agent(self) -> int:
        """Return the number of GT agents.

        Returns
        -------
            int: The number of GT agents.

        """
        return len(self.types)

    @property
    def num_scenario_frame(self) -> int:
        """Return the number of scenario frame length.

        Returns
        -------
            int: The number of scenario frame length.

        """
        return self.yaw.shape[-1]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Construct instance from dict data.

        Args:
        ----
            data (dict[str, Any]): Serialized dict data.

        Returns:
        -------
            Self: Instance.

        """
        return cls(**data)

    def apply_mask(self, mask: NDArrayBool) -> Self:
        """Filter by mask.

        Args:
        ----
            mask (NDArrayBool): Mask for agent in the shape of (N,).

        Returns:
        -------
            Self: Filtered instance.

        """
        if mask.ndim != 1:
            msg = "Input mask must be 1d."
            raise ValueError(msg)

        types = self.types[mask]
        ids = self.ids[mask]
        xy = self.xy[mask]
        size = self.size[mask]
        yaw = self.yaw[mask]
        vxy = self.vxy[mask]
        is_valid = self.is_valid[mask]

        return GroundTruth(types=types, ids=ids, xy=xy, size=size, yaw=yaw, vxy=vxy, is_valid=is_valid)

    def __iter__(self) -> Iterator:
        yield from zip(self.types, self.ids, self.xy, self.size, self.yaw, self.vxy, self.is_valid)
