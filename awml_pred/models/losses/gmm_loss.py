import torch

from awml_pred.common import LOSSES
from awml_pred.models.losses import BaseLoss
from awml_pred.typing import Tensor

__all__ = ("GMMLoss",)


@LOSSES.register()
class GMMLoss(BaseLoss):
    """GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508."""

    def __init__(
        self,
        weight: float = 1.0,
        *,
        use_square_gmm: bool = False,
        log_std_range: tuple[float, float] = (-1.609, 5.0),
        rho_limit: float = 0.5,
        name: str | None = None,
    ) -> None:
        """
        Construct GMM Loss for MTR.

        Args:
        ----
            weight (float, optional): Weight of loss.
            use_square_gmm (bool, optional): _description_. Defaults to False.
            log_std_range (tuple[float, float], optional): _description_. Defaults to (-1.609, 5.0).
            rho_limit (float, optional): _description_. Defaults to 0.5.
            name (str | None, optional): Name of loss. Defaults to None.
        """
        super().__init__(weight, name)
        self.weight = weight
        self.use_square_gmm = use_square_gmm
        self.log_std_range = log_std_range
        self.rho_limit = rho_limit

    def forward(
        self,
        pred_scores: Tensor,
        pred_trajs: Tensor,
        gt_trajs: Tensor,
        gt_valid_mask: Tensor,
        pre_nearest_mode_idxs: Tensor | None = None,
        timestamp_loss_weight: Tensor | None = None,
    ) -> Tensor:
        return self.weight * nll_loss_gmm_direct(
            pred_scores,
            pred_trajs,
            gt_trajs,
            gt_valid_mask,
            pre_nearest_mode_idxs,
            timestamp_loss_weight,
            self.use_square_gmm,
            self.log_std_range,
            self.rho_limit,
        )


def nll_loss_gmm_direct(
    pred_scores: Tensor,
    pred_trajs: Tensor,
    gt_trajs: Tensor,
    gt_valid_mask: Tensor,
    pre_nearest_mode_idxs: Tensor | None = None,
    timestamp_loss_weight: Tensor | None = None,
    use_square_gmm: bool = False,
    log_std_range: tuple[float, float] = (-1.609, 5.0),
    rho_limit: float = 0.5,
) -> Tensor:
    """
    GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
    Written by Shaoshuai Shi.

    Args:
    ----
        pred_scores (torch.Tensor): in shape (batch_size, num_modes).
        pred_trajs (torch.Tensor): in shape (batch_size, num_modes, num_timestamps, 5 or 3).
        gt_trajs (torch.Tensor): in shape (batch_size, num_timestamps, 2).
        gt_valid_mask (torch.Tensor): in shape (batch_size, num_timestamps).
        pre_nearest_mode_idxs (torch.Tensor | None, optional): _description_. Defaults to None.
        timestamp_loss_weight (torch.Tensor | None, optional): in shape (num_timestamps). Defaults to None.
        use_square_gmm (bool, optional): _description_. Defaults to False.
        log_std_range (tuple[float, float], optional): _description_. Defaults to (-1.609, 5.0).
        rho_limit (float, optional): _description_. Defaults to 0.5.

    Returns:
    -------
        Tensor: _description_
    """
    if use_square_gmm:
        assert pred_trajs.shape[-1] == 3
    else:
        assert pred_trajs.shape[-1] == 5

    batch_size = pred_scores.shape[0]

    if pre_nearest_mode_idxs is not None:
        nearest_mode_idxs = pre_nearest_mode_idxs
    else:
        distance: torch.Tensor = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :]).norm(dim=-1)
        distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1)

        nearest_mode_idxs = distance.argmin(dim=-1)
    nearest_mode_bs_idxs = torch.arange(batch_size).type_as(nearest_mode_idxs)  # (batch_size, 2)

    nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]  # (batch_size, num_timestamps, 5)
    res_trajs = gt_trajs - nearest_trajs[:, :, 0:2]  # (batch_size, num_timestamps, 2)
    dx = res_trajs[:, :, 0]
    dy = res_trajs[:, :, 1]

    if use_square_gmm:
        log_std1 = log_std2 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        std1 = std2 = torch.exp(log_std1)  # (0.2m to 150m)
        rho = torch.zeros_like(log_std1)
    else:
        log_std1 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        log_std2 = torch.clip(nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
        std1 = torch.exp(log_std1)  # (0.2m to 150m)
        std2 = torch.exp(log_std2)  # (0.2m to 150m)
        rho = torch.clip(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

    gt_valid_mask = gt_valid_mask.type_as(pred_scores)
    if timestamp_loss_weight is not None:
        gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

    # -log(a^-1 * e^b) = log(a) - b
    reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho**2)  # (batch_size, num_timestamps)
    reg_gmm_exp = (0.5 * 1 / (1 - rho**2)) * (
        (dx**2) / (std1**2) + (dy**2) / (std2**2) - 2 * rho * dx * dy / (std1 * std2)
    )  # (batch_size, num_timestamps)

    reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)

    return reg_loss
