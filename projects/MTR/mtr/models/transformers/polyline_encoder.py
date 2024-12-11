# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved


import torch
from torch import nn

from awml_pred.common import LAYERS
from awml_pred.models import build_mlps
from awml_pred.typing import Tensor

__all__ = ("PointNetPolylineEncoder",)


@LAYERS.register()
class PointNetPolylineEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int = 3,
        num_pre_layers: int = 1,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.pre_mlps = build_mlps(c_in=in_channels, mlp_channels=[hidden_dim] * num_pre_layers, ret_before_act=False)
        self.mlps = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False,
        )

        if out_channels is not None:
            self.out_mlps = build_mlps(
                c_in=hidden_dim,
                mlp_channels=[hidden_dim, out_channels],
                ret_before_act=True,
                without_norm=True,
            )
        else:
            self.out_mlps = None

    def forward(self, polylines: Tensor, polylines_mask: Tensor) -> Tensor:
        """Return polyline feature.

        Args:
        ----
            polylines (Tensor): in shape (batch_size, num_polylines, num_points_each_polylines, 9).
            polylines_mask (Tensor): in shape (batch_size, num_polylines, num_points_each_polylines).

        Returns:
        -------
            Tensor: Polyline feature.

        """
        batch_size, num_polylines, num_points_each_polylines, C = polylines.shape

        # pre-mlp
        # masked_polylines = polylines[polylines_mask]
        masked_polylines = (polylines * polylines_mask[..., None]).view(-1, C)
        polylines_pre_feature_valid: Tensor = self.pre_mlps(masked_polylines)
        polylines_pre_feature = (
            polylines_pre_feature_valid.view(batch_size, num_polylines, num_points_each_polylines, self.hidden_dim)
            * polylines_mask[..., None]
        )

        # get global feature
        pooled_feature: Tensor = polylines_pre_feature.max(dim=2)[0]
        polylines_feature = torch.cat(
            (
                polylines_pre_feature,
                pooled_feature[:, :, None, :].repeat(1, 1, num_points_each_polylines, 1),
            ),
            dim=3,
        )

        # mlp
        # masked_polylines = polylines_feature[polylines_mask]
        masked_polylines_feature = (polylines_feature * polylines_mask[..., None]).view(-1, self.hidden_dim * 2)
        polylines_feature_valid: Tensor = self.mlps(masked_polylines_feature)  # (N, self.hidden_dim)
        feature_buffers = (
            polylines_feature_valid.view(batch_size, num_polylines, num_points_each_polylines, self.hidden_dim)
            * polylines_mask[..., None]
        )

        # max-pooling
        feature_buffers, _ = feature_buffers.max(dim=2)

        # out-mlp
        if self.out_mlps is not None:
            valid_mask = polylines_mask.sum(dim=2) > 0
            masked_feature_buffers = (feature_buffers * valid_mask[..., None]).view(-1, self.hidden_dim)
            feature_buffers_valid: Tensor = self.out_mlps(masked_feature_buffers)
            feature_buffers = (
                feature_buffers_valid.view(batch_size, num_polylines, self.out_channels) * valid_mask[..., None]
            )
        return feature_buffers
