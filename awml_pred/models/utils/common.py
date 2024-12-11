import torch

__all__ = ("get_batch_offsets",)


def get_batch_offsets(
    batch_idxs: torch.Tensor,
    batch_size: int,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """
    Return the batch offsets.

    Args:
    ----
        batch_idxs (torch.Tensor): Batch indices, in shape (N,).
        bs (int): Batch size.
        device (str | torch.device): Device name. Defaults to "cuda".

    Returns:
    -------
        torch.Tensor: Batch offsets, in shape (bs + 1,).
    """
    batch_offsets = torch.zeros(batch_size + 1, device=device).int()
    for i in range(batch_size):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    # assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets
