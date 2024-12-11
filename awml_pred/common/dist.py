import os
import shutil
import subprocess
from typing import Any

import torch
import torch.distributed as torch_dist

from .io import load_pkl, save_pkl

_LOCAL_PROCESS_GROUP = None


def get_dist_info() -> tuple[int, int]:
    """Return the rank of the current process and the number of processes.

    Returns
    -------
        tuple[int, int]: `(rank, world_size)`.

    """
    rank: int = get_rank()
    world_size: int = get_world_size()
    return rank, world_size


def get_num_devices() -> int:
    """Return the total number of visible GPU devices.

    Returns
    -------
        int: Total number of devices.

    """
    gpu_list = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if gpu_list is not None:
        return len(gpu_list.split(","))
    else:
        return torch.cuda.device_count()


def get_world_size() -> int:
    """Return the number of processes in the current process group.

    Returns
    -------
        int: The world size of the process group -1, if not part of the group.

    """
    if not torch_dist.is_available() or not torch_dist.is_initialized():
        return 1
    else:
        return torch_dist.get_world_size()


def get_rank() -> int:
    """Return the rank of the current process in the provided group.

    Returns
    -------
        int: The rank of process group -1, if not part of the group.

    """
    if not torch_dist.is_available() or not torch_dist.is_initialized():
        return 0
    else:
        return torch_dist.get_rank()


def get_local_rank() -> int:
    """Return the rank of the current process in the provided group within the local.

    Returns
    -------
        The rank of the current process within the local (per-machine) process group.

    """
    if not torch_dist.is_available():
        return 0
    if not torch_dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return torch_dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """Return the number of processes in the current process group within the local.

    Returns
    -------
        int: The size of the per-machine process group, i.e. the number of processes per machine.

    """
    if not torch_dist.is_available():
        return 1
    if not torch_dist.is_initialized():
        return 1
    return torch_dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    """Return whether the current process is main one, which means `rank==0`.

    Returns
    -------
        bool: True, if the current process is main one.

    """
    return get_rank() == 0


def init_dist_slurm(tcp_port: str, _local_rank: int, backend: str = "nccl") -> tuple[int, int]:
    """Init slurm distribution.

    modified from https://github.com/open-mmlab/mmdetection.

    Args:
    ----
        tcp_port (str): TCP port.
        _local_rank (int): Local rank.
        backend (str, optional): Backend name.

    Returns:
    -------
        tuple[int, int]: Total number of GPUs and rank.

    """
    proc_id = int(os.environ["SLURM_PROCID"])
    ntasks = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")  # noqa: S605
    os.environ["MASTER_PORT"] = str(tcp_port)
    os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(ntasks)
    os.environ["RANK"] = str(proc_id)
    torch_dist.init_process_group(backend=backend)

    total_gpus = get_world_size()
    rank = get_rank()
    return total_gpus, rank


def init_dist_pytorch(_tcp_port: str, local_rank: int, backend: str = "nccl") -> tuple[int, int]:
    """Init pytorch distribution.

    Args:
    ----
        tcp_port (str): TCP port
        local_rank (int): Local rank.
        backend (str, optional): Backend name. Defaults to "nccl".

    Returns:
    -------
        tuple[int, int]: Total number of GPUs and rank.

    """
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)

    torch_dist.init_process_group(backend=backend)
    rank = get_rank()
    return num_gpus, rank


def merge_dist_results(result_part: list[Any], size: int, tmpdir: str) -> list[dict] | None:
    """Merge distributed results.

    Args:
    ----
        result_part (list[Any]): List of results.
        size (int): Size of world.
        tmpdir (str): Directory path used temporally.

    Returns:
    -------
        list[dict] | None: List of merged results.

    """
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    torch_dist.barrier()
    save_pkl(result_part, os.path.join(tmpdir, f"result_part_{rank}.pkl"))
    torch_dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, f"result_part_{i}.pkl")
        part_list.append(load_pkl(part_file))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results
