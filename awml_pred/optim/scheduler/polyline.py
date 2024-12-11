from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from torch.optim.lr_scheduler import _LRScheduler

from awml_pred.common import LR_SCHEDULERS

if TYPE_CHECKING:
    from awml_pred.typing import Optimizer

__all__ = ("PolylineLR",)


@LR_SCHEDULERS.register()
class PolylineLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        milestones: list[int],
        values: list[float],
        last_epoch: int = -1,
        verbose: str = "deprecated",
    ) -> None:
        """Wrap constructor."""
        assert len(milestones) == len(values), "[PolylineLR] length must be same"
        assert all(x >= 0 for x in milestones), "[PolylineLR] milestones must be positive"
        assert all(x >= 0 for x in values), "[PolylineLR] values must be positive"
        assert milestones[0] == 0, "[PolylineLR] milestones must start from 0"
        assert milestones == sorted(milestones), "[PolylineLR] milestones must be in ascending order"

        self.milestones = milestones
        self.values = values
        self.n_intervals = len(self.milestones) - 1
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        """Get the current learning rates."""
        if not self._get_lr_called_within_step:
            warnings.warning(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.",
                UserWarning,
            )
        lr = self._get_value(self.last_epoch)
        return [lr for _ in self.optimizer.param_groups]

    def _get_value(self, epoch: int) -> float:
        assert epoch >= 0
        for i in range(self.n_intervals):
            e_lb = self.milestones[i]
            e_ub = self.milestones[i + 1]
            if epoch < e_lb or epoch >= e_ub:
                continue  # not in this interval
            v_lb = self.values[i]
            v_ub = self.values[i + 1]
            return (epoch - e_lb) / (e_ub - e_lb) * (v_ub - v_lb) + v_lb
        return self.values[-1]
