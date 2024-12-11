from typing import Any

import torch
from torch import nn

from awml_pred.common import ENCODERS, LAYERS
from awml_pred.models.utils import get_batch_offsets, sine_positional_embed
from awml_pred.typing import Tensor

from ..ops import knn_batch_mlogk

__all__ = ("MTREncoder",)


@ENCODERS.register()
class MTREncoder(nn.Module):
    def __init__(
        self,
        agent_polyline_encoder: dict[str, Any],
        map_polyline_encoder: dict[str, Any],
        attention_layer: dict[str, Any],
        use_local_attn: bool = True,
        num_attn_neighbors: int = 16,
    ) -> None:
        super().__init__()
        self.agent_polyline_encoder = LAYERS.build(agent_polyline_encoder)
        self.map_polyline_encoder = LAYERS.build(map_polyline_encoder)

        attn_cfg = attention_layer.copy()
        num_attn_layers = attn_cfg.pop("num_layers")
        self.self_attn_layers = nn.ModuleList([LAYERS.build(attn_cfg) for _ in range(num_attn_layers)])

        self.use_local_attn = use_local_attn
        self.num_attn_neighbors = num_attn_neighbors

    def apply_global_attn(self, x: Tensor, x_mask: Tensor, x_pos: Tensor) -> Tensor:
        """Apply global attention.

        Args:
        ----
            x (Tensor): in shape (B, T, D).
            x_mask (Tensor): in shape (B, T, D).
            x_pos (Tensor): in shape (B, T, 3).

        Returns:
        -------
            Tensor:

        """
        _, _, dim = x.shape
        x_t = x.permute(1, 0, 2)
        x_mask_t = x_mask.permute(1, 0, 2)
        x_pos_t = x_pos.permute(1, 0, 2)

        pos_embed = sine_positional_embed(x_pos_t, hidden_dim=dim)
        for attn in self.self_attn_layers:
            x_t = attn(src=x_t, src_key_padding_mask=~x_mask_t, pos=pos_embed)
        x_out = x_t.permute(1, 0, 2)
        return x_out

    def apply_local_attn(
        self,
        x: Tensor,
        x_mask: Tensor,
        x_pos: Tensor,
        num_neighbors: int,
    ) -> Tensor:
        """Apply local attention.

        Args:
        ----
            x (Tensor): (B, N, D)
            x_mask (Tensor): (B, N)
            x_pos (Tensor): (B, N, 3)
            num_neighbors (int): Number of TopK.

        Returns:
        -------
            Tensor: _description_

        """
        batch_size, num, dim = x.shape

        x_stack_full = x.view(batch_size * num, dim)
        x_mask_stack = x_mask.view(batch_size * num)
        x_pos_stack_full = x_pos.view(batch_size * num, 3)
        batch_idxs_full = torch.arange(batch_size, dtype=torch.int32, device=x.device)[:, None].repeat(1, num).view(-1)

        # filter invalid elements
        x_stack = x_stack_full * x_mask_stack[..., None]
        x_pos_stack = x_pos_stack_full * x_mask_stack[..., None]
        batch_idxs = batch_idxs_full * x_mask_stack

        # knn
        batch_offsets = get_batch_offsets(batch_idxs, batch_size)
        batch_cnt = batch_offsets[1:] - batch_offsets[:-1]
        # (num_valid_elements, num_k)
        index_pair = knn_batch_mlogk(x_pos_stack, x_pos_stack, batch_idxs, batch_offsets, num_neighbors)
        index_pair = torch.masked_fill(index_pair, x_mask_stack[..., None], -1)

        # positional encoding
        pos_embed = sine_positional_embed(x_pos_stack[None, :, 0:2], hidden_dim=dim)[0]
        pos_embed = pos_embed * x_mask_stack[..., None]

        # local attention
        output = x_stack
        for attn in self.self_attn_layers:
            output = attn(
                src=output,
                pos=pos_embed,
                index_pair=index_pair,
                query_batch_cnt=batch_cnt,
                key_batch_cnt=batch_cnt,
                index_pair_batch=batch_idxs,
            )
            output = output * x_mask_stack[..., None]

        ret_full_feature = output.view(batch_size, num, dim) * x_mask[..., None]
        return ret_full_feature

    def forward(
        self,
        obj_trajs: Tensor,
        obj_trajs_mask: Tensor,
        map_polylines: Tensor,
        map_polylines_mask: Tensor,
        map_polylines_center: Tensor,
        obj_trajs_last_pos: Tensor,
        track_index_to_predict: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward operation.

        Args:
        ----
            obj_trajs (Tensor)
            obj_trajs_mask (Tensor)
            map_polylines (Tensor)
            map_polylines_mask (Tensor)
            obj_trajs_last_pos (Tensor)
            track_index_to_predict (Tensor)

        Returns:
        -------
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
                - obj_feature (Tensor)
                - obj_mask (Tensor)
                - obj_pos (Tensor)
                - map_feature (Tensor)
                - map_mask (Tensor)
                - map_pos (Tensor)
                - center_objects_feature (Tensor)

        """
        num_center_objects, num_objects, _, _ = obj_trajs.shape

        # apply polyline encoder
        obj_trajs_in = torch.cat((obj_trajs, obj_trajs_mask[..., None].type_as(obj_trajs)), dim=-1)
        obj_polylines_feature = self.agent_polyline_encoder(obj_trajs_in, obj_trajs_mask)
        map_polylines_feature = self.map_polyline_encoder(map_polylines, map_polylines_mask)

        # apply self-attn
        obj_valid_mask = obj_trajs_mask.sum(dim=-1) > 0  # (num_center_objects, num_objects)
        map_valid_mask = map_polylines_mask.sum(dim=-1) > 0  # (num_center_objects, num_polylines)

        global_token_feature = torch.cat((obj_polylines_feature, map_polylines_feature), dim=1)
        global_token_mask = torch.cat((obj_valid_mask, map_valid_mask), dim=1)
        global_token_pos = torch.cat((obj_trajs_last_pos, map_polylines_center), dim=1)

        if self.use_local_attn:
            global_token_feature = self.apply_local_attn(
                x=global_token_feature,
                x_mask=global_token_mask,
                x_pos=global_token_pos,
                num_neighbors=self.num_attn_neighbors,
            )
        else:
            global_token_feature = self.apply_global_attn(
                x=global_token_feature,
                x_mask=global_token_mask,
                x_pos=global_token_pos,
            )

        obj_polylines_feature = global_token_feature[:, :num_objects]
        map_polylines_feature = global_token_feature[:, num_objects:]

        # organize return features
        center_objects_feature = obj_polylines_feature[torch.arange(num_center_objects), track_index_to_predict]

        return (
            obj_polylines_feature,  # obj_feature
            obj_valid_mask,  # obj_mask
            obj_trajs_last_pos,  # obj_pos
            map_polylines_feature,  # map_feature
            map_valid_mask,  # map_mask
            map_polylines_center,  # map_pos
            center_objects_feature,  # center_objects_feature
        )
