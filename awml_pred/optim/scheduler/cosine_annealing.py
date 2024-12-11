from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

from torch.optim.lr_scheduler import _LRScheduler

from awml_pred.common import LR_SCHEDULERS

if TYPE_CHECKING:
    from awml_pred.typing import Optimizer

__all__ = ("CosineAnnealingWithWarmupLR",)


@LR_SCHEDULERS.register()
class CosineAnnealingWithWarmupLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,  # noqa: N803
        eta_min: int,
        T_warmup: int,  # noqa: N803
        last_epoch: int = -1,
        verbose: str = "deprecated",
    ) -> None:
        """Wrap constructor."""
        self.T_max = T_max - T_warmup
        self.eta_min = eta_min
        self.T_warmup = T_warmup
        self.init_lr = optimizer.param_groups[0]["lr"]
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        """Get the current learning rates."""
        if not self._get_lr_called_within_step:
            warnings.warning(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch <= self.T_warmup:
            lr = (self.last_epoch / self.T_warmup) * (self.init_lr - self.eta_min) + self.eta_min
            return [lr for _ in self.optimizer.param_groups]

        net_last_epoch = self.last_epoch - self.T_warmup
        if self._step_count == 1 and net_last_epoch > 0:
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(net_last_epoch * math.pi / self.T_max)) / 2
                for base_lr, _ in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif (net_last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * net_last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (net_last_epoch - 1) / self.T_max))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]
