import torch.nn.functional as F

from awml_pred.common import LOSSES
from awml_pred.models.losses import BaseLoss
from awml_pred.typing import Tensor

__all__ = ("CrossEntropyLoss",)


@LOSSES.register()
class CrossEntropyLoss(BaseLoss):
    def __init__(self, weight: float = 1.0, name: str | None = None) -> None:
        """
        Construct instance.

        Args:
        ----
            weight (float, optional): Weight of loss. Defaults to 1.0.
            name (str | None, optional): Name of loss instance. Defaults to None.
        """
        super().__init__(weight, name)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute the cross entropy loss.

        Args:
        ----
            pred (Tensor): Prediction tensor.
            target (Tensor): Target tensor.

        Returns:
        -------
            Tensor: Loss score.
        """
        return self.weight * F.cross_entropy(pred, target, reduction="none").sum(dim=-1)
