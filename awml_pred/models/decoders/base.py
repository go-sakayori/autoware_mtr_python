from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any

from torch import nn

from awml_pred.typing import Tensor

from ..builder import build_loss

if TYPE_CHECKING:
    from ..losses import BaseLoss

__all__ = ("BaseDecoder",)


class BaseDecoder(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        in_channels: int,
        num_future_frames: int,
        num_motion_modes: int,
        decode_loss: dict | None = None,
    ) -> None:
        """
        Construct instance.

        Args:
        ----
            in_channels (int): The number of input channels.
            num_future_frames (int): The number of predicting future frames.
            num_motion_modes (int): The number of predicting modes.
            decode_loss (dict | None, optional): Configuration of the loss function. Defaults to None.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_future_frames = num_future_frames
        self.num_motion_modes = num_motion_modes

        if decode_loss is not None:
            self.decode_loss: BaseLoss = build_loss(decode_loss)
        else:
            self.decode_loss: BaseLoss = build_loss(self.__default_loss_cfg__())

    @staticmethod
    @abstractmethod
    def __default_loss_cfg__() -> dict:
        """
        Load default loss configuration.

        Returns
        -------
            dict: Default loss configuration.
        """
        ...

    @abstractmethod
    def get_loss(self, log_prefix: str = "") -> dict:
        """
        Return loss.

        Args:
        ----
            log_prefix (str, optional): Prefix of log message. Defaults to "".

        Returns:
        -------
            dict: Loss result.
        """
        ...

    @abstractmethod
    def get_prediction(self, **kwargs: Any) -> tuple[Tensor, Tensor]:
        """
        Return final prediction.

        Args:
        ----
            **kwargs (Any): ...

        Returns:
        -------
            tuple[Tensor, Tensor]: Predicted score and trajectory.
                score: in shape (N, M)
                trajectory: in shape (N, M, T, 7)
        """
        ...
