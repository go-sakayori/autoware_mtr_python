from __future__ import annotations

from typing import TYPE_CHECKING

from omegaconf import DictConfig

from awml_pred.common import LR_SCHEDULERS, OPTIMIZERS

if TYPE_CHECKING:
    from awml_pred.typing import LRScheduler, Module, Optimizer

__all__ = ("build_optimizer", "build_scheduler")


def build_optimizer(model: Module, cfg: DictConfig) -> Optimizer:
    """Return optimizer.

    Expecting configuration format as below:

    ```
    optimizer:
        name: <OPTIMIZER NAME>
        ...PARAMETERS
    ```

    Supported optimizers:
    * Adam
    * AdamW

    Args:
    ----
        model (Module): Module.
        cfg (DictConfig): Configuration about optimization.

    Returns:
    -------
        Optimizer: Optimizer instance.

    """
    new_cfg = dict(cfg)
    new_cfg["params"] = model.parameters()

    return OPTIMIZERS.build(new_cfg)


def build_scheduler(optimizer: Optimizer, cfg: DictConfig) -> LRScheduler:
    """Return LR scheduler.

    Expecting configuration as below:

    ```
    lr_scheduler:
        name: <LR SCHEDULER NAME>
        ...PARAMETERS
    ```

    The supported schedulers are following.
    * LambdaLR
    * LinearLR
    * StepLR
    * MultiStepLR
    * ExponentialLR
    * CosineAnnealingLR
    * CosineAnnealingWarmRestarts

    Args:
    ----
        optimizer (Optimizer): Optimizer instance.
        cfg (DictConfig): Configuration for optimization.

    Returns:
    -------
        LRScheduler: Constructed LR scheduler.

    """
    new_cfg = dict(cfg)
    new_cfg["optimizer"] = optimizer
    return LR_SCHEDULERS.build(new_cfg)
