from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from torch import optim

from awml_pred.common import OPTIMIZERS

if TYPE_CHECKING:
    from awml_pred.typing import Tensor

__all__ = ("Adam", "AdamW")


@OPTIMIZERS.register()
class Adam(optim.Adam):
    def __init__(
        self,
        params: Iterable,
        lr: float | Tensor = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        *,
        amsgrad: bool = False,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None:
        """Wrap constructor."""
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )


@OPTIMIZERS.register()
class AdamW(optim.AdamW):
    def __init__(
        self,
        params: Iterable,
        lr: float | Tensor = 0.001,
        betas: tuple[float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        *,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: bool | None = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None:
        """Wrap constructor."""
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
