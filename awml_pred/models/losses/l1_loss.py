import torch.nn.functional as F

from awml_pred.common import LOSSES
from awml_pred.models.losses import BaseLoss
from awml_pred.typing import Tensor

__all__ = ("L1Loss",)


@LOSSES.register()
class L1Loss(BaseLoss):
    def __init__(self, weight: float = 1.0, name: str | None = None) -> None:
        super().__init__(weight, name)

    def forward(self, pred: Tensor, target: Tensor, target_mask: Tensor) -> Tensor:
        loss = F.l1_loss(pred, target, reduction="none")
        return self.weight * (loss * target_mask).sum(dim=-1).sum(dim=-1)
