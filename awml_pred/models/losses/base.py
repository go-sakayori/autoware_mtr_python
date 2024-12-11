from abc import abstractmethod
from typing import Any

from torch import nn

from awml_pred.typing import Tensor

__all__ = ("BaseLoss", "BaseModelLoss")


class BaseModelLoss(nn.Module):
    """Abstract base of model specific loss function."""

    def __init__(self) -> None:
        super().__init__()
        self.tb_dict = {}
        self.display_dict = {}

    def clear_log(self) -> None:
        """Clear cached log information."""
        self.tb_dict.clear()
        self.display_dict.clear()

    @abstractmethod
    def compute_loss(self, *args: Any, **kwargs: Any) -> Tensor:
        """
        Compute total loss.

        Returns
        -------
            Tensor: Total loss.
        """

    def forward(self, *args: Any, **kwargs: Any) -> dict:
        """
        Compute loss.

        Returns
        -------
            dict: Total Loss and containers for tensorboard and display log.
        """
        self.clear_log()

        loss = self.compute_loss(*args, **kwargs)

        log_prefix: str = kwargs.get("log_prefix", "")
        self.tb_dict[f"{log_prefix}loss"] = loss.item()
        self.display_dict[f"{log_prefix}loss"] = loss.item()

        return {"loss": loss, "tensorboard": self.tb_dict, "display": self.display_dict}


class BaseLoss(nn.Module):
    """Abstract base of loss function."""

    def __init__(self, weight: float, name: str | None = None) -> None:
        super().__init__()
        self.weight = weight
        self.name = name if name is not None else self.__class__.__name__
