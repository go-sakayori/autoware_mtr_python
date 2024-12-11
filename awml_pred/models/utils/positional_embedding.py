import torch

__all__ = ("sine_positional_embed",)


def sine_positional_embed(positions: torch.Tensor, hidden_dim: int) -> torch.Tensor:
    """
    Return the sine-positional embedded tensor.

    Args:
    ----
        positions (torch.Tensor): _description_
        hidden_dim (int): _description_

    Returns:
    -------
        torch.Tensor: _description_
    """
    half_hidden_dim = hidden_dim // 2
    scale = 2.0 * torch.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=positions.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = positions[:, :, 0] * scale
    y_embed = positions[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if positions.size(-1) == 2:
        return torch.cat((pos_y, pos_x), dim=2)
    elif positions.size(-1) == 4:
        w_embed = positions[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = positions[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=2).flatten(2)
        return torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError(f"Unexpected position.size(-1): {positions.size(-1)}")
