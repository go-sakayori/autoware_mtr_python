from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from awml_pred.common import DECODERS
from awml_pred.models import BaseDecoder, build_mlps, sine_positional_embed
from awml_pred.ops import batch_nms
from awml_pred.typing import Module, ModuleList, Tensor

from .transformers import TransformerDecoderLayer

__all__ = ("MTRDecoder",)


@DECODERS.register()
class MTRDecoder(BaseDecoder):
    def __init__(
        self,
        in_channels: int,
        num_future_frames: int,
        num_motion_modes: int,
        d_model: int,
        num_decoder_layers: int,
        num_attn_head: int,
        map_center_offset: tuple[float, float],
        num_waypoint_map_polylines: int,
        num_base_map_polylines: int,
        dropout: float = 0.1,
        map_d_model: int | None = None,
        nms_threshold: float = 2.5,
        *,
        use_place_holder: bool = False,
        decode_loss: dict | None = None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_future_frames=num_future_frames,
            num_motion_modes=num_motion_modes,
            decode_loss=decode_loss,
        )
        self.num_decoder_layers = num_decoder_layers
        self.map_center_offset = map_center_offset
        self.num_waypoint_map_polylines = num_waypoint_map_polylines
        self.num_base_map_polylines = num_base_map_polylines
        self.d_model = d_model
        self.map_d_model = d_model if map_d_model is None else map_d_model
        self.nms_threshold = nms_threshold
        self.use_place_holder = use_place_holder

        # cross-attn layers
        self.in_proj_center_obj = nn.Sequential(
            nn.Linear(in_channels, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.in_proj_obj, self.obj_decoder_layers = self.build_transformer_decoder(
            in_channels=in_channels,
            d_model=self.d_model,
            num_head=num_attn_head,
            dropout=dropout,
            num_decoder_layers=self.num_decoder_layers,
            use_local_attn=False,
        )

        self.in_proj_map, self.map_decoder_layers = self.build_transformer_decoder(
            in_channels=in_channels,
            d_model=self.map_d_model,
            num_head=num_attn_head,
            dropout=dropout,
            num_decoder_layers=self.num_decoder_layers,
            use_local_attn=True,
        )
        if self.map_d_model != self.d_model:
            temp_layer = nn.Linear(self.d_model, self.map_d_model)
            self.map_query_content_mlps = nn.ModuleList([deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])
            self.map_query_embed_mlps = nn.Linear(self.d_model, self.map_d_model)
        else:
            self.map_query_content_mlps = self.map_query_embed_mlps = None

        # define dense future prediction layers
        self.build_dense_future_prediction_layers(hidden_dim=self.d_model, num_future_frames=self.num_future_frames)

        # define the motion query
        self.intention_query, self.intention_query_mlps = self.build_motion_query(self.d_model, self.use_place_holder)

        # define the motion head
        temp_layer = build_mlps(
            c_in=self.d_model * 2 + self.map_d_model,
            mlp_channels=[self.d_model, self.d_model],
            ret_before_act=True,
        )
        self.query_feature_fusion_layers = nn.ModuleList([deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])

        self.motion_reg_heads, self.motion_cls_heads, self.motion_vel_heads = self.build_motion_head(
            in_channels=self.d_model,
            hidden_size=self.d_model,
            num_decoder_layers=self.num_decoder_layers,
        )

    @staticmethod
    def __default_loss_cfg__() -> dict:
        return {
            "name": "MTRLoss",
            "reg_cfg": {"name": "GMMLoss", "weight": 1.0, "use_square_gmm": False},
            "cls_cfg": {"name": "CrossEntropyLoss", "weight": 1.0},
            "vel_cfg": {"name": "L1Loss", "weight": 0.2},
        }

    def build_dense_future_prediction_layers(self, hidden_dim: int, num_future_frames: int) -> None:
        self.obj_pos_encoding_layer = build_mlps(
            c_in=2,
            mlp_channels=[hidden_dim, hidden_dim, hidden_dim],
            ret_before_act=True,
            without_norm=True,
        )
        self.dense_future_head = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim, hidden_dim, num_future_frames * 7],
            ret_before_act=True,
        )

        self.future_traj_mlps = build_mlps(
            c_in=4 * self.num_future_frames,
            mlp_channels=[hidden_dim, hidden_dim, hidden_dim],
            ret_before_act=True,
            without_norm=True,
        )
        self.traj_fusion_mlps = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim, hidden_dim, hidden_dim],
            ret_before_act=True,
            without_norm=True,
        )

    def build_transformer_decoder(
        self,
        in_channels: int,
        d_model: int,
        num_head: int,
        dropout: float = 0.1,
        num_decoder_layers: int = 1,
        use_local_attn: bool = False,
    ) -> tuple[Module, ModuleList]:
        in_proj_layer = nn.Sequential(nn.Linear(in_channels, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            num_head=num_head,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="relu",
            normalize_before=False,
            keep_query_pos=False,
            rm_self_attn_decoder=False,
            use_local_attn=use_local_attn,
        )
        decoder_layers = nn.ModuleList([deepcopy(decoder_layer) for _ in range(num_decoder_layers)])
        return in_proj_layer, decoder_layers

    def build_motion_query(self, d_model: int, use_place_holder: bool = False) -> tuple[None, ModuleList]:
        """Build motion query.

        Args:
        ----
            d_model (int): _description_
            use_place_holder (bool, optional): _description_. Defaults to False.

        Raises:
        ------
            NotImplementedError: _description_

        Returns:
        -------
            tuple[None, ModuleList]: Intention query and MLPs for intention query.

        """
        intention_query = intention_query_mlps = None

        if use_place_holder:
            raise NotImplementedError
        else:
            intention_query_mlps = build_mlps(c_in=d_model, mlp_channels=[d_model, d_model], ret_before_act=True)
        return intention_query, intention_query_mlps

    def build_motion_head(
        self,
        in_channels: int,
        hidden_size: int,
        num_decoder_layers: int,
    ) -> tuple[ModuleList, ModuleList, None]:
        motion_reg_head = build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_size, hidden_size, self.num_future_frames * 7],
            ret_before_act=True,
        )

        motion_cls_head = build_mlps(c_in=in_channels, mlp_channels=[hidden_size, hidden_size, 1], ret_before_act=True)

        motion_reg_heads = nn.ModuleList([deepcopy(motion_reg_head) for _ in range(num_decoder_layers)])
        motion_cls_heads = nn.ModuleList([deepcopy(motion_cls_head) for _ in range(num_decoder_layers)])
        motion_vel_heads = None
        return motion_reg_heads, motion_cls_heads, motion_vel_heads

    def apply_dense_future_prediction(
        self,
        obj_feature: Tensor,
        obj_mask: Tensor,
        obj_pos: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Args:
        ----
            obj_feature (Tensor): _description_
            obj_mask (Tensor): _description_
            obj_pos (Tensor): _description_.

        Returns
        -------
            tuple[Tensor, Tensor]: _description_

        """
        num_center_objects, num_objects, num_feature = obj_feature.shape

        # dense future prediction
        obj_pos_valid = (obj_pos[..., :2] * obj_mask[..., None]).view(-1, 2)
        obj_feature_valid = (obj_feature * obj_mask[..., None]).view(-1, num_feature)
        obj_pos_feature_valid: Tensor = self.obj_pos_encoding_layer(obj_pos_valid)
        obj_fused_feature_valid = torch.cat((obj_pos_feature_valid, obj_feature_valid), dim=-1)

        pred_dense_trajs_valid: Tensor = self.dense_future_head(obj_fused_feature_valid)
        pred_dense_trajs_valid = pred_dense_trajs_valid.view(
            num_center_objects * num_objects,
            self.num_future_frames,
            7,
        )

        temp_center = pred_dense_trajs_valid[:, :, 0:2] + obj_pos_valid[:, None, 0:2]
        pred_dense_trajs_valid = torch.cat((temp_center, pred_dense_trajs_valid[:, :, 2:]), dim=-1)

        # future feature encoding and fuse to past obj_feature
        obj_future_input_valid = pred_dense_trajs_valid[:, :, [0, 1, -2, -1]].flatten(start_dim=1, end_dim=2)
        obj_future_feature_valid = self.future_traj_mlps(obj_future_input_valid)

        obj_full_trajs_feature = torch.cat((obj_feature_valid, obj_future_feature_valid), dim=-1)
        obj_feature_valid = self.traj_fusion_mlps(obj_full_trajs_feature)

        ret_obj_feature = obj_feature_valid.view(num_center_objects, num_objects, num_feature) * obj_mask[..., None]

        ret_pred_dense_future_trajs = (
            pred_dense_trajs_valid.view(num_center_objects, num_objects, self.num_future_frames, 7)
            * obj_mask[..., None, None]
        )

        return ret_obj_feature, ret_pred_dense_future_trajs

    def get_motion_query(self, intention_points: Tensor) -> tuple[Tensor, Tensor]:
        """Return motion intention queries and points.

        Args:
        ----
            intention_points (Tensor): (num_center_objects, K, 2)

        Returns:
        -------
            tuple[Tensor, Tensor]: _description_

        """
        if self.use_place_holder:
            raise NotImplementedError
        else:
            num_center_objects = intention_points.size(0)
            intention_points = intention_points.permute(1, 0, 2)
            intention_query = sine_positional_embed(intention_points, hidden_dim=self.d_model)
            intention_query: Tensor = self.intention_query_mlps(intention_query.view(-1, self.d_model))
            intention_query = intention_query.view(-1, num_center_objects, self.d_model)
        return intention_query, intention_points

    def apply_cross_attention(
        self,
        kv_feature: Tensor,
        kv_mask: Tensor,
        kv_pos: Tensor,
        query_content: Tensor,
        query_embed: Tensor,
        attention_layer: Module,
        dynamic_query_center: Tensor | None = None,
        layer_idx: int = 0,
        use_local_attn: bool = False,
        query_index_pair: Tensor | None = None,
        query_content_pre_mlp: Module | None = None,
        query_embed_pre_mlp: Tensor | Module = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply cross attention.

        Args:
        ----
            kv_feature (Tensor): _description_
            kv_mask (Tensor): _description_
            kv_pos (Tensor): _description_
            query_content (Tensor): _description_
            query_embed (Tensor): _description_
            attention_layer (Module): _description_
            dynamic_query_center (Tensor | None, optional): _description_. Defaults to None.
            layer_idx (int, optional): _description_. Defaults to 0.
            use_local_attn (bool, optional): _description_. Defaults to False.
            query_index_pair (Tensor | None, optional): _description_. Defaults to None.
            query_content_pre_mlp (Module | None, optional): _description_. Defaults to None.
            query_embed_pre_mlp (Tensor | None, optional): _description_. Defaults to None.

        Returns:
        -------
            tuple[Tensor, Tensor]: _description_

        """
        if query_content_pre_mlp is not None:
            query_content = query_content_pre_mlp(query_content)
        if query_embed_pre_mlp is not None:
            query_embed = query_embed_pre_mlp(query_embed)

        num_q, batch_size, dim = query_content.shape
        searching_query = sine_positional_embed(dynamic_query_center, hidden_dim=dim)
        kv_pos = kv_pos.permute(1, 0, 2)[:, :, 0:2]
        kv_pos_embed = sine_positional_embed(kv_pos, hidden_dim=dim)

        if not use_local_attn:
            query_feature = attention_layer(
                tgt=query_content,
                query_pos=query_embed,
                query_sine_embed=searching_query,
                memory=kv_feature.permute(1, 0, 2),
                memory_key_padding_mask=~kv_mask,
                pos=kv_pos_embed,
                is_first=(layer_idx == 0),
            )  # (M, B, C)
        else:
            batch_size, num_kv, _ = kv_feature.shape

            kv_feature_stack = kv_feature.flatten(start_dim=0, end_dim=1)
            kv_pos_embed_stack = kv_pos_embed.permute(1, 0, 2).contiguous().flatten(start_dim=0, end_dim=1)
            kv_mask_stack = kv_mask.view(-1)

            key_batch_cnt = num_kv * torch.ones(batch_size, dtype=torch.int, device=kv_feature.device)
            query_index_pair = query_index_pair.view(batch_size * num_q, -1)
            index_pair_batch = torch.arange(batch_size).type_as(key_batch_cnt)[:, None].repeat(1, num_q).view(-1)

            query_feature: Tensor = attention_layer(
                tgt=query_content,
                query_pos=query_embed,
                query_sine_embed=searching_query,
                memory=kv_feature_stack,
                memory_valid_mask=kv_mask_stack,
                pos=kv_pos_embed_stack,
                is_first=(layer_idx == 0),
                key_batch_cnt=key_batch_cnt,
                index_pair=query_index_pair,
                index_pair_batch=index_pair_batch,
            )
            query_feature = query_feature.view(batch_size, num_q, dim).permute(1, 0, 2)  # (M, B, C)

        return query_feature

    def apply_dynamic_map_collection(
        self,
        map_pos: Tensor,
        map_mask: Tensor,
        pred_waypoints: Tensor,
        base_region_offset: tuple[float, float],
        num_query: int,
        num_waypoint_polylines: int = 128,
        num_base_polylines: int = 256,
        base_map_idxs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply dynamic map collection.

        Args:
        ----
            map_pos (Tensor): _description_
            map_mask (Tensor): _description_
            pred_waypoints (Tensor): _description_
            base_region_offset (tuple[float, float]): _description_
            num_query (int): _description_
            num_waypoint_polylines (int, optional): _description_. Defaults to 128.
            num_base_polylines (int, optional): _description_. Defaults to 256.
            base_map_idxs (Tensor | None, optional): _description_. Defaults to None.

        Returns:
        -------
            tuple[Tensor, Tensor]: _description_

        """
        map_pos = map_pos.clone()
        map_pos.masked_fill_(~map_mask[..., None], torch.nan)
        num_polylines = map_pos.shape[1]

        if base_map_idxs is None:
            base_points = torch.tensor(base_region_offset).type_as(map_pos)
            base_dist: Tensor = (map_pos[:, :, 0:2] - base_points[None, None, :]).norm(dim=-1)
            base_topk_dist, base_map_idxs = base_dist.topk(
                k=min(num_polylines, num_base_polylines),
                dim=-1,
                largest=True,
            )
            # NOTE: not to use tensor[mask] = other
            # base_map_idxs[base_topk_dist > 10000000] = -1
            base_map_idxs.masked_fill_(base_topk_dist.isnan(), -1)
            base_map_idxs = base_map_idxs[:, None, :].repeat(1, num_query, 1)
            if base_map_idxs.shape[-1] < num_base_polylines:
                base_map_idxs = F.pad(
                    base_map_idxs,
                    pad=(0, num_base_polylines - base_map_idxs.shape[-1]),
                    mode="constant",
                    value=-1,
                )

        dynamic_dist: Tensor = (pred_waypoints[:, :, None, :, 0:2] - map_pos[:, None, :, None, 0:2]).norm(dim=-1)
        dynamic_dist = dynamic_dist.min(dim=-1)[0]

        dynamic_topk_dist, dynamic_map_idxs = dynamic_dist.topk(
            k=min(num_polylines, num_waypoint_polylines),
            dim=-1,
            largest=False,
        )
        dynamic_map_idxs.masked_fill_(dynamic_topk_dist.isnan(), -1)
        if dynamic_map_idxs.shape[-1] > num_waypoint_polylines:
            dynamic_map_idxs = F.pad(
                dynamic_map_idxs,
                pad=(0, num_waypoint_polylines - dynamic_map_idxs.shape[-1]),
                mode="constant",
                value=-1,
            )

        collected_idxs = torch.cat([base_map_idxs, dynamic_map_idxs], dim=-1)

        # remove duplicate indices
        sorted_idxs: Tensor = collected_idxs.sort(dim=-1)[0]
        duplicate_mask_slice = sorted_idxs[..., 1:] - sorted_idxs[..., :-1] != 0
        duplicate_mask = torch.ones_like(collected_idxs, dtype=torch.bool)
        duplicate_mask[..., 1:] = duplicate_mask_slice
        sorted_idxs = torch.masked_fill(sorted_idxs, ~duplicate_mask, -1)

        return sorted_idxs.int(), base_map_idxs

    def apply_transformer_decoder(
        self,
        center_objects_feature: Tensor,
        intention_points: Tensor,
        obj_feature: Tensor,
        obj_mask: Tensor,
        obj_pos: Tensor,
        map_feature: Tensor,
        map_mask: Tensor,
        map_pos: Tensor,
    ) -> tuple[list[tuple[Tensor, Tensor]], Tensor]:
        """Apply transformer decoder.

        Args:
        ----
            center_objects_feature (Tensor): _description_
            intention_points (Tensor):
            obj_feature (Tensor): _description_
            obj_mask (Tensor): _description_
            obj_pos (Tensor): _description_
            map_feature (Tensor): _description_
            map_mask (Tensor): _description_
            map_pos (Tensor): _description_

        Returns:
        -------
            list[tuple[Tensor, Tensor]]: Predicted scores and trajectories.

        """
        intention_query, intention_points = self.get_motion_query(intention_points)
        query_content = torch.zeros_like(intention_query)

        num_query, num_center_objects = query_content.shape[:2]

        center_objects_feature = center_objects_feature[None, :, :].repeat(num_query, 1, 1)

        base_map_idxs = None
        pred_waypoints = intention_points.permute(1, 0, 2)[:, :, None, :]  # (num_center_objects, num_query, 1, 2)
        dynamic_query_center = intention_points

        pred_list: list[tuple[Tensor, Tensor]] = []
        for layer_idx in range(self.num_decoder_layers):
            # query object feature
            obj_query_feature = self.apply_cross_attention(
                kv_feature=obj_feature,
                kv_mask=obj_mask,
                kv_pos=obj_pos,
                query_content=query_content,
                query_embed=intention_query,
                attention_layer=self.obj_decoder_layers[layer_idx],
                dynamic_query_center=dynamic_query_center,
                layer_idx=layer_idx,
            )

            # query map feature
            collected_idxs, base_map_idxs = self.apply_dynamic_map_collection(
                map_pos=map_pos,
                map_mask=map_mask,
                pred_waypoints=pred_waypoints,
                base_region_offset=self.map_center_offset,
                num_query=num_query,
                num_waypoint_polylines=self.num_waypoint_map_polylines,
                num_base_polylines=self.num_base_map_polylines,
                base_map_idxs=base_map_idxs,
            )

            map_query_feature = self.apply_cross_attention(
                kv_feature=map_feature,
                kv_mask=map_mask,
                kv_pos=map_pos,
                query_content=query_content,
                query_embed=intention_query,
                attention_layer=self.map_decoder_layers[layer_idx],
                layer_idx=layer_idx,
                dynamic_query_center=dynamic_query_center,
                use_local_attn=True,
                query_index_pair=collected_idxs,
                query_content_pre_mlp=self.map_query_content_mlps[layer_idx],
                query_embed_pre_mlp=self.map_query_embed_mlps,
            )

            query_feature: Tensor = torch.cat((center_objects_feature, obj_query_feature, map_query_feature), dim=-1)
            query_content: Tensor = self.query_feature_fusion_layers[layer_idx](
                query_feature.flatten(start_dim=0, end_dim=1),
            ).view(num_query, num_center_objects, -1)

            # motion prediction
            query_content_t = query_content.permute(1, 0, 2).contiguous().view(num_center_objects * num_query, -1)
            pred_scores: Tensor = self.motion_cls_heads[layer_idx](query_content_t).view(num_center_objects, num_query)
            if self.motion_vel_heads is not None:
                pred_trajs: Tensor = self.motion_reg_heads[layer_idx](query_content_t).view(
                    num_center_objects,
                    num_query,
                    self.num_future_frames,
                    5,
                )
                pred_vel: Tensor = self.motion_vel_heads[layer_idx](query_content_t).view(
                    num_center_objects,
                    num_query,
                    self.num_future_frames,
                    2,
                )
                pred_trajs = torch.cat((pred_trajs, pred_vel), dim=-1)
            else:
                pred_trajs: Tensor = self.motion_reg_heads[layer_idx](query_content_t).view(
                    num_center_objects,
                    num_query,
                    self.num_future_frames,
                    7,
                )

            pred_list.append([pred_scores, pred_trajs])

            # update
            pred_waypoints = pred_trajs[:, :, :, 0:2]
            dynamic_query_center = pred_trajs[:, :, -1, 0:2].contiguous().permute(1, 0, 2)

        if self.use_place_holder:
            raise NotImplementedError

        return pred_list, intention_points.permute(1, 0, 2)

    def get_loss(
        self,
        pred_list: list[Tensor],
        pred_dense_trajs: Tensor,
        intention_points: Tensor,
        center_gt_trajs: Tensor,
        center_gt_trajs_mask: Tensor,
        center_gt_final_valid_idx: Tensor,
        obj_trajs_future_state: Tensor,
        obj_trajs_future_mask: Tensor,
        device: torch.device,
        tb_pre_tag: str = "",
    ) -> dict:
        """Calculate loss for MTR.

        Args:
        ----
            pred_list (list[Tensor]): List of predicted trajectories.
            intention_points (Tensor): Intention points for target agents.
            pred_dense_trajs (Tensor): Dense future prediction.
            center_gt_trajs (Tensor): GT trajectories, in shape (B, Tp, 4).
            center_gt_trajs_mask (Tensor): Mask of GT trajectories, in shape (B, Tp).
            center_gt_final_valid_idx (Tensor): Indices of target agents.
            obj_trajs_future_state (Tensor): Future trajectory.
            obj_trajs_future_mask (Tensor): Mask of future trajectory.
            device (torch.device): Device.
            tb_pre_tag (str, optional): Pre-tag for tensorboard log. Defaults to "".

        Returns:
        -------
            dict: A container of loss and logs.
            * loss (float): Calculated total loss.
            * tensorboard (dict): Loss result for tensorboard.
            * display (dict): Loss result for display.

        """
        return self.decode_loss(
            pred_list=pred_list,
            pred_dense_future=pred_dense_trajs,
            intention_points=intention_points,
            gt_trajs=center_gt_trajs,
            gt_trajs_mask=center_gt_trajs_mask,
            gt_indices=center_gt_final_valid_idx,
            agent_future=obj_trajs_future_state,
            agent_future_mask=obj_trajs_future_mask,
            device=device,
            log_prefix=tb_pre_tag,
        )

    def get_prediction(self, pred_list: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        """Returns final predictions.

        Args:
        ----
            pred_list (list[tuple[Tensor, Tensor]]): List of decoder predictions for each layer.

        Returns:
        -------
            tuple[Tensor, Tensor]: Scores and trajectories.
            - `pred_scores` in shape (B, M).
            - `pred_trajs` in shape (B, M, Tp, 7).

        """
        pred_scores, pred_trajs = pred_list[-1]
        pred_scores = torch.softmax(pred_scores, dim=-1)  # (num_center_objects, num_query)

        # (num_center_objects, num_query, num_future_timestamps, num_feat)
        _, num_query, _, _ = pred_trajs.shape
        if self.num_motion_modes != num_query:
            pred_trajs_final, pred_scores_final, _ = batch_nms(
                pred_trajs=pred_trajs,
                pred_scores=pred_scores,
                dist_thresh=self.nms_threshold,
                num_ret_modes=self.num_motion_modes,
            )
        else:
            pred_trajs_final = pred_trajs
            pred_scores_final = pred_scores

        return pred_scores_final, pred_trajs_final

    def forward(
        self,
        obj_feature: Tensor,
        obj_mask: Tensor,
        obj_pos: Tensor,
        map_feature: Tensor,
        map_mask: Tensor,
        map_pos: Tensor,
        center_objects_feature: Tensor,
        intention_points: Tensor,
        center_gt_trajs: Tensor | None = None,
        center_gt_trajs_mask: Tensor | None = None,
        center_gt_final_valid_idx: Tensor | None = None,
        obj_trajs_future_state: Tensor | None = None,
        obj_trajs_future_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | dict:
        """Args:
        ----
            obj_feature (Tensor): (B, N, Fa)
            obj_mask (Tensor): (B, N)
            obj_pos (Tensor): (B, N, 3)
            map_feature (Tensor): (B, K, Fp)
            map_mask (Tensor): (B, K)
            map_pos (Tensor): (B, K, 3)
            center_objects_feature (Tensor): (B, Fa)
            intention_points (Tensor): (B, K, 2)
            center_gt_trajs (Tensor | None)
            center_gt_trajs_mask (Tensor | None)
            center_gt_final_valid_idx (Tensor | None)
            obj_trajs_future_state (Tensor | None)
            obj_trajs_future_mask (Tensor | None).

        Returns
        -------
            tuple[Tensor, Tensor] | dict: In training, returns loss dict.
                Otherwise returns tensor of scores and trajectories.

        """
        num_center_objects, num_objects, num_obj_feature = obj_feature.shape
        _, num_polylines, num_map_feature = map_feature.shape

        # input projection
        center_objects_feature: Tensor = self.in_proj_center_obj(center_objects_feature)

        masked_obj_feature = (obj_feature * obj_mask[..., None]).view(-1, num_obj_feature)
        obj_feature_valid = self.in_proj_obj(masked_obj_feature)

        obj_feature = obj_feature_valid.view(num_center_objects, num_objects, self.d_model) * obj_mask[..., None]

        masked_map_feature = (map_feature * map_mask[..., None]).view(-1, num_map_feature)
        map_feature_valid = self.in_proj_map(masked_map_feature)
        map_feature = map_feature_valid.view(num_center_objects, num_polylines, self.map_d_model) * map_mask[..., None]

        # dense future prediction
        obj_feature, pred_dense_trajs = self.apply_dense_future_prediction(
            obj_feature=obj_feature,
            obj_mask=obj_mask,
            obj_pos=obj_pos,
        )
        # decoder layers
        pred_list, intention_points = self.apply_transformer_decoder(
            center_objects_feature=center_objects_feature,
            intention_points=intention_points,
            obj_feature=obj_feature,
            obj_mask=obj_mask,
            obj_pos=obj_pos,
            map_feature=map_feature,
            map_mask=map_mask,
            map_pos=map_pos,
        )

        if not self.training:
            pred_scores, pred_trajs = self.get_prediction(pred_list=pred_list)
            return pred_scores, pred_trajs
        else:
            loss = self.get_loss(
                pred_list=pred_list,
                pred_dense_trajs=pred_dense_trajs,
                intention_points=intention_points,
                center_gt_trajs=center_gt_trajs,
                center_gt_trajs_mask=center_gt_trajs_mask,
                center_gt_final_valid_idx=center_gt_final_valid_idx,
                obj_trajs_future_state=obj_trajs_future_state,
                obj_trajs_future_mask=obj_trajs_future_mask,
                device=obj_feature.device,
            )
            return loss
