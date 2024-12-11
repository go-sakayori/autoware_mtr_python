from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from torch.optim import lr_scheduler

from awml_pred.common import LR_SCHEDULERS

if TYPE_CHECKING:
    from awml_pred.typing import Optimizer

__all__ = ("CosineAnnealingWarmRestarts", "CosineAnnealingLR", "LambdaLR", "StepLR", "MultiStepLR", "ExponentialLR")


@LR_SCHEDULERS.register()
class CosineAnnealingWarmRestarts(lr_scheduler.CosineAnnealingWarmRestarts):
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,  # noqa: N803
        T_mult: int = 1,  # noqa: N803
        eta_min: int = 0,
        last_epoch: int = -1,
        verbose: str = "deprecated",
    ) -> None:
        """Wrap constructor."""
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch, verbose)


@LR_SCHEDULERS.register()
class CosineAnnealingLR(lr_scheduler.CosineAnnealingLR):
    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,  # noqa: N803
        eta_min: int = 0,
        last_epoch: int = -1,
        verbose: str = "deprecated",
    ) -> None:
        """Wrap constructor."""
        super().__init__(optimizer, T_max, eta_min, last_epoch, verbose)


@LR_SCHEDULERS.register()
class LambdaLR(lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable | list[Callable] | None = None,
        last_epoch: int = -1,
        verbose: str = "deprecated",
        *,
        milestones: list[int] | None = None,
        lr_decay: float | None = None,
        lr_clip: float | None = None,
    ) -> None:
        """Wrap constructor."""
        if lr_lambda is None:
            assert milestones is not None
            assert lr_decay is not None
            assert lr_clip is not None
            init_lr = optimizer.param_groups[0]["lr"]

            def lr_lambda(cur_epoch: int) -> float:
                cur_decay = 1
                for decay_step in milestones:
                    if cur_epoch >= decay_step:
                        cur_decay = cur_decay * lr_decay
                return max(cur_decay, lr_clip / init_lr)

        super().__init__(optimizer, lr_lambda, last_epoch, verbose)


@LR_SCHEDULERS.register()
class LinearLR(lr_scheduler.LinearLR):
    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 1 / 3,
        end_factor: float = 1,
        total_iters: int = 5,
        last_epoch: int = -1,
        verbose: str = "deprecated",
    ) -> None:
        """Wrap constructor."""
        super().__init__(optimizer, start_factor, end_factor, total_iters, last_epoch, verbose)


@LR_SCHEDULERS.register()
class StepLR(lr_scheduler.StepLR):
    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: float = -1,
        verbose: str = "deprecated",
    ) -> None:
        """Wrap constructor."""
        super().__init__(optimizer, step_size, gamma, last_epoch, verbose)


@LR_SCHEDULERS.register()
class MultiStepLR(lr_scheduler.MultiStepLR):
    def __init__(
        self,
        optimizer: Optimizer,
        milestones: list[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: str = "deprecated",
    ) -> None:
        """Wrap constructor."""
        super().__init__(optimizer, milestones, gamma, last_epoch, verbose)


@LR_SCHEDULERS.register()
class ExponentialLR(lr_scheduler.ExponentialLR):
    def __init__(self, optimizer: Optimizer, gamma: float, last_epoch: int = -1, verbose: str = "deprecated") -> None:
        """Wrap constructor."""
        super().__init__(optimizer, gamma, last_epoch, verbose)
