from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from torch import device, nn
from typing_extensions import Self

from awml_pred.models import build_decoder, build_encoder

if TYPE_CHECKING:
    from ..decoders import BaseDecoder

__all__ = ("BaseDetector",)


class BaseDetector(nn.Module):
    def __init__(self, encoder: dict, decoder: dict) -> None:
        """
        Construct detector.

        Args:
        ----
            encoder (dict): Configuration of encoder.
            decoder (dict): Configuration of decoder.
        """
        super().__init__()

        self.encoder = build_encoder(encoder)
        self.decoder: BaseDecoder = build_decoder(decoder)

    def to(self, *args, **kwargs) -> Self:
        """Move or cast all parameters and buffers.

        Returns:
            Self: Instance of myself.
        """
        self.encoder.to(args, kwargs)
        self.decoder.to(args, kwargs)
        return super().to(args, kwargs)

    def cuda(self, device: int | device | None = None) -> Self:
        """Move all parameters and buffers to the GPU.

        Args:
            device (int | device | None, optional): If specified, all parameters will be copied to that device.
                Defaults to None.

        Returns:
            Self: Instance of myself.
        """
        self.encoder.cuda(device)
        self.decoder.cuda(device)
        return super().cuda(device)

    @property
    def num_motion_modes(self) -> int:
        """
        Return the number of predicting modes.

        Returns
        -------
            int: The number of modes.
        """
        return self.decoder.num_motion_modes

    @abstractmethod
    def extract_feature(self, *args: Any, **kwargs: Any) -> Any:
        """Extract feature by encoder."""
        ...
