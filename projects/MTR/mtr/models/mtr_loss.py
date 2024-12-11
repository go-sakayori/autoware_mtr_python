import torch

from awml_pred.common import LOSSES
from awml_pred.models import build_loss
from awml_pred.models.losses import BaseModelLoss
from awml_pred.typing import Tensor

__all__ = ("MTRLoss",)


@LOSSES.register()
class MTRLoss(BaseModelLoss):
    """Loss module for MTR."""

    def __init__(self, reg_cfg: dict, cls_cfg: dict, vel_cfg: dict) -> None:
        """Args:
        ----
            reg_cfg (dict): Configuration of regression loss.
            cls_cfg (dict): Configuration of class loss.
            vel_cfg (dict): Configuration of velocity loss.

        """
        super().__init__()
        self.reg_loss = build_loss(reg_cfg)
        self.cls_loss = build_loss(cls_cfg)
        self.vel_loss = build_loss(vel_cfg)

    def compute_decoder_loss(
        self,
        pred_list: list[Tensor],
        intention_points: Tensor,
        gt_trajs: Tensor,
        gt_trajs_mask: Tensor,
        gt_indices: Tensor,
        device: str | torch.device,
        log_prefix: str,
    ) -> Tensor:
        """Compute decoder loss.

        Args:
        ----
            pred_list (list[Tensor]): List of predictions, in shape (L, ((B, M), (B, M, T, D)).
                Where `L` is the number of decoder layers.
            intention_points (Tensor): Intention points for target agents, in shape (B, K, 2).
            gt_trajs (Tensor): GT trajectory, in shape (B, T, D).
            gt_trajs_mask (Tensor): Mask of GT trajectory, in shape (B, T).
            gt_indices (Tensor): Indices of valid GTs.
            device (str | torch.device): Device type.
            log_prefix (str): Prefix of log text.

        Returns:
        -------
            Tensor: Loss result.

        """
        # 4: (x, y, vx, vy), 6(x, y, vx, vy, cos, sin)
        assert gt_trajs.shape[-1] in (4, 6)
        gt_trajs = gt_trajs.to(device)
        gt_trajs_mask = gt_trajs_mask.to(device)
        gt_indices = gt_indices.long()

        num_targets: int = gt_trajs.shape[0]
        gt_goals = gt_trajs[torch.arange(num_targets), gt_indices, 0:2]

        distances: Tensor = (gt_goals[:, None, :] - intention_points).norm(dim=-1)
        gt_positive_idx: Tensor = distances.argmin(dim=-1)

        total_loss = 0.0
        for layer_idx, (pred_scores, pred_trajs) in enumerate(pred_list):
            pred_trajs_gmm, pred_vel = pred_trajs[..., 0:5], pred_trajs[..., 5:7]

            reg_gmm_loss = self.reg_loss(
                pred_scores=pred_scores,
                pred_trajs=pred_trajs_gmm,
                gt_trajs=gt_trajs[..., 0:2],
                gt_valid_mask=gt_trajs_mask,
                pre_nearest_mode_idxs=gt_positive_idx,
                timestamp_loss_weight=None,
            )

            pred_vel = pred_vel[torch.arange(num_targets), gt_positive_idx]
            reg_vel_loss = self.vel_loss(pred_vel, gt_trajs[..., 2:4], gt_trajs_mask[..., None])

            cls_loss = self.cls_loss(pred_scores, gt_positive_idx)

            layer_loss = (reg_gmm_loss + reg_vel_loss + cls_loss).mean()
            total_loss += layer_loss

            self.tb_dict[f"{log_prefix}loss_layer{layer_idx}"] = layer_loss.item()
            self.tb_dict[f"{log_prefix}loss_layer{layer_idx}_reg_gmm"] = reg_gmm_loss.mean().item()
            self.tb_dict[f"{log_prefix}loss_layer{layer_idx}_reg_vel"] = reg_vel_loss.mean().item()
            self.tb_dict[f"{log_prefix}loss_layer{layer_idx}_cls"] = cls_loss.mean().item()

        total_loss = total_loss / len(pred_list)
        return total_loss

    def compute_dense_future_loss(
        self,
        pred_dense_future: Tensor,
        agent_future: Tensor,
        agent_future_mask: Tensor,
        device: str | torch.device,
        log_prefix: str,
    ) -> Tensor:
        """Compute dense future loss.

        Args:
        ----
            pred_dense_future (Tensor): Predicted dense future
            agent_future (Tensor): Agent future trajectory, in shape (B, N, T, D).
            agent_future_mask (Tensor): Mask of agent future trajectory, in shape (B, N, T).
            device (str | torch.device): Device type.
            log_prefix (str): Prefix of log text.

        Returns:
        -------
            Tensor: Loss result.

        """
        agent_future = agent_future.to(device)
        agent_future_mask = agent_future_mask.to(device)

        pred_dense_future_gmm, pred_dense_future_vel = pred_dense_future[..., 0:5], pred_dense_future[..., 5:7]
        reg_vel_loss = self.vel_loss(pred_dense_future_vel, agent_future[..., 2:4], agent_future_mask[..., None])

        num_targets, num_agents, num_timestamps, _ = pred_dense_future.shape
        fake_scores = pred_dense_future.new_zeros((num_targets, num_agents)).view(-1, 1)

        tmp_pred_trajs = pred_dense_future_gmm.contiguous().view(num_targets * num_agents, 1, num_timestamps, 5)
        tmp_gt_idx = torch.zeros(num_targets * num_agents, dtype=torch.long, device=device)
        tmp_gt_trajs = agent_future[..., 0:2].contiguous().view(num_targets * num_agents, num_timestamps, 2)
        tmp_gt_trajs_mask = agent_future_mask.view(num_targets * num_agents, num_timestamps)
        reg_gmm_loss = self.reg_loss(
            pred_scores=fake_scores,
            pred_trajs=tmp_pred_trajs,
            gt_trajs=tmp_gt_trajs,
            gt_valid_mask=tmp_gt_trajs_mask,
            pre_nearest_mode_idxs=tmp_gt_idx,
            timestamp_loss_weight=None,
        )
        reg_gmm_loss = reg_gmm_loss.view(num_targets, num_agents)

        reg_loss = reg_vel_loss + reg_gmm_loss

        agent_valid_mask = agent_future_mask.sum(dim=-1) > 0
        reg_loss = (reg_loss * agent_valid_mask.float()).sum(dim=-1) / torch.clamp_min(
            agent_valid_mask.sum(dim=-1),
            min=1.0,
        )
        reg_loss = reg_loss.mean()

        self.tb_dict[f"{log_prefix}loss_dense_prediction"] = reg_loss.item()

        return reg_loss

    def compute_loss(
        self,
        pred_list: list[Tensor],
        pred_dense_future: Tensor,
        intention_points: Tensor,
        gt_trajs: Tensor,
        gt_trajs_mask: Tensor,
        gt_indices: Tensor,
        agent_future: Tensor,
        agent_future_mask: Tensor,
        device: str | torch.device,
        log_prefix: str = "",
    ) -> Tensor:
        """Compute MTR loss.

        Args:
        ----
            pred_list (list[Tensor]): List of predictions, in shape (L, ((B, M), (B, M, T, D)).
                Where `L` is the number of decoder layers.
            pred_dense_future (Tensor): Predicted dense future.
            intention_points (Tensor): Intention points for target agents, in shape (B, K, 2).
            gt_trajs (Tensor): GT trajectory, in shape (B, T, D).
            gt_trajs_mask (Tensor): Mask of GT trajectory, in shape (B, T).
            gt_indices (Tensor): Indices of valid GTs.
            agent_future (Tensor): Agent future trajectory, in shape (B, N, T, D).
            agent_future_mask (Tensor): Mask of agent future trajectory, in shape (B, N, T).
            device (str | torch.device): Device type.
            log_prefix (str): Prefix of log text. Defaults to "".

        Returns:
        -------
            Tensor: Total loss.

        """
        decoder_loss = self.compute_decoder_loss(
            pred_list,
            intention_points,
            gt_trajs,
            gt_trajs_mask,
            gt_indices,
            device,
            log_prefix,
        )
        dense_future_loss = self.compute_dense_future_loss(
            pred_dense_future,
            agent_future,
            agent_future_mask,
            device,
            log_prefix,
        )

        return decoder_loss + dense_future_loss
