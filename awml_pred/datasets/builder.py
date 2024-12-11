from __future__ import annotations

import logging
import random
from functools import partial
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from awml_pred.common import DATASETS, get_dist_info, get_num_devices

if TYPE_CHECKING:
    from awml_pred.datasets.base import BaseDataset

    AWMLDataset = TypeVar("AWMLDataset", bound=BaseDataset)

__all__ = ("build_dataset", "build_dataloader")


def build_dataset(cfg: DictConfig) -> AWMLDataset:
    """Return `Dataset`.

    Args:
    ----
        cfg (DictConfig): Configuration of dataset.

    Returns:
    -------
        BaseDataset: `Dataset` instance.

    """
    return DATASETS.build(cfg)


def build_dataloader(
    dataset_cfg: DictConfig,
    *,
    is_distributed: bool,
    training: bool,
    seed: int | None = None,
) -> DataLoader:
    """Return `DataLoader`.

    Args:
    ----
        dataset_cfg (DictConfig): Configuration for dataset.
        is_distributed (bool): Whether to use distributed execution.
        training (bool): Whether to train or not.
        seed (int | None, optional): Seed number to be set. Defaults to None.

    Returns:
    -------
        DataLoader: Generated `DataLoader` instance.

    """
    dataset: AWMLDataset = build_dataset(dataset_cfg.train if training else dataset_cfg.test)

    rank, world_size = get_dist_info()
    if is_distributed:
        sampler = (
            DistributedSampler(dataset) if training else DistributedSampler(dataset, world_size, rank, shuffle=False)
        )
    else:
        sampler = None

    num_gpus: int = get_num_devices()
    batch_size: int = dataset_cfg.samples_per_gpu
    num_workers: int = dataset_cfg.workers_per_gpu
    pin_memory: bool = dataset_cfg.get("pin_memory", True)
    shuffle: bool = sampler is None and training
    drop_last: bool = dataset_cfg.get("drop_last", False) and training
    init_fn = partial(_worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=dataset.collate_batch,
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=drop_last,
    )
    msg = f"Batch size: {batch_size}, Num GPUs: {num_gpus}, Num workers: {num_workers}"
    logging.info(msg)
    return data_loader


def _worker_init_fn(worker_id: int, num_workers: int, rank: int, seed: int | None = None) -> None:
    worker_seed = num_workers * rank + worker_id
    if seed is not None:
        worker_seed += seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
