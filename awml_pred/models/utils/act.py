from typing import Callable

import torch.nn.functional as F

__all__ = ("get_activation_fn",)


def get_activation_fn(activation: str) -> Callable:
    """
    Return activation function.

    Args:
    ----
        activation (str): Name of activation function.

    Raises:
    ------
        RuntimeError: If not relu/gelu/glu specified.

    Returns:
    -------
        Callable: Activation function.
    """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "glu":
        return F.glu
    else:
        raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
