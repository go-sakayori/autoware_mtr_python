from math import gcd

from torch import nn

from awml_pred.common import LAYERS
from awml_pred.typing import Tensor

__all__ = ("Conv1d",)


@LAYERS.register()
class Conv1d(nn.Module):
    """A module extending `nn.Conv1d`, which contains a normalization layer and `ReLU` activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm: str = "GN",
        ng: int = 32,
        *,
        act: bool = True,
    ) -> None:
        """
        Construct instance.

        Args:
        ----
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int, optional): The size of kernel. Defaults to 3.
            stride (int, optional): The size of stride. Defaults to 1.
            norm (str, optional): Name of a normalization layer. Defaults to "GN".
            ng (int, optional): The number of groups, which is only used for `"GN"`. Defaults to 32.
            act (bool, optional): Whether to apply activation before output. Defaults to True.
        """
        super().__init__()
        assert norm in ("GN", "BN", "SyncBN"), f"Unexpected norm layer: {norm}"

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=stride,
            bias=False,
        )

        if norm == "GN":
            self.norm = nn.GroupNorm(gcd(ng, out_channels), out_channels)
        elif norm == "BN":
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.SyncBatchNorm(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        """
        Run forward operation.

        Args:
        ----
            x (Tensor): Input tensor.

        Returns:
        -------
            Tensor: Output tensor.
        """
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out
