import logging
from typing import Any

import torch
from torch import nn

from awml_pred.typing import Module, Optimizer, Tensor

__all__ = ("get_checkpoint_state", "save_checkpoint", "load_checkpoint", "is_parallel")

CHECKPOINT_KEYS = ("epoch", "model", "optimizer")


def get_checkpoint_state(model: Module, optimizer: Optimizer, epoch: int) -> dict[str, Any]:
    """Return checkpoint states as dict.

    The output state has the following keys:

    * epoch (int)
    * model (dict)
    * optimizer (dict)

    Args:
    ----
        model (Module): Model.
        optimizer (Optimizer): Optimizer.
        epoch (int): Epoch number.

    Returns:
    -------
        dict[str, Any]: State in dict.

    """

    def model2cpu(model_state: dict) -> dict:
        model_state_cpu = type(model_state)()
        for key, val in model_state.items():
            model_state_cpu[key] = val.cpu()
        return model_state_cpu

    model_state = model2cpu(model.module.state_dict()) if is_parallel(model) else model.state_dict()

    optimizer_state = optimizer.state_dict()

    return {"epoch": epoch, "model": model_state, "optimizer": optimizer_state}


def save_checkpoint(state: dict, filename: str) -> None:
    """Save checkpoint.

    Args:
    ----
        state (dict): state dict.
        filename (str): Name of file to be saved.

    """
    if set(state.keys()) != set(CHECKPOINT_KEYS):
        msg = f"Unexpected key is found in state, {set(state.keys())}"
        raise ValueError(msg)

    if not filename.endswith(".pth"):
        filename += ".pth"
    torch.save(state, filename)


def load_checkpoint(
    model: Module,
    checkpoint: dict | str,
    optimizer: Optimizer | None = None,
    *,
    is_distributed: bool = False,
) -> tuple[nn.Module, int]:
    """Load checkpoint.

    Args:
    ----
        model (Module): Target model to be loaded checkpoint.
        checkpoint (dict): State dict of checkpoint or filepath.
        optimizer (Optimizer | None, optional): Optimizer. Defaults to None.
        is_distributed (bool): Whether distributed is. Defaults to False.

    Returns:
    -------
        tuple[nn.Module, int]: Loaded model and the Last epoch.

    """
    if not isinstance(checkpoint, dict):
        map_location = None if is_distributed else torch.device("cpu")
        checkpoint = torch.load(checkpoint, map_location=map_location, weights_only=True)

    if is_parallel(model):
        state_dict: dict = model.module.state_dict()
    else:
        state_dict: dict = model.state_dict()

    model_checkpoint = checkpoint["model"]
    load_dict = {}
    for name, weight in state_dict.items():
        weight: Tensor
        if name not in model_checkpoint:
            msg = f"{name} is not in the checkpoint. Please double check and see if this is desired."
            logging.warning(msg)
            continue
        ckpt_weight: Tensor = model_checkpoint[name]
        if weight.shape != ckpt_weight.shape:
            msg = (
                f"Shape of {name} in checkpoint is {ckpt_weight.shape}, "
                f"while shape of {name} in model is {weight.shape}"
            )
            logging.warning(msg)
            continue
        load_dict[name] = ckpt_weight

    if is_parallel(model):
        model.module.load_state_dict(load_dict)
    else:
        model.load_state_dict(load_dict)

    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])

    return model, checkpoint["epoch"]


def is_parallel(model: Module) -> bool:
    """Return whether the model is in parallel mode.

    Args:
    ----
        model (Module): Model.

    Returns:
    -------
        bool: True, if the type of model is `DataParallel` or `DistributedDataParallel`.

    """
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))
