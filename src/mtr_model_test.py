import argparse
import os
from datetime import datetime, timezone

import torch
from omegaconf import DictConfig

from awml_pred.common import Config, create_logger, get_num_devices, init_dist_pytorch, init_dist_slurm, load_checkpoint
from awml_pred.datasets import build_dataloader
from awml_pred.models import build_model
from awml_pred.runner import TestRunner


def parse_args() -> argparse.Namespace:
    """Return parsed.

    Returns
    -------
        argparse.Namespace: _description_

    """
    parser = argparse.ArgumentParser(description="Test model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", type=str, help="specify the config for training")
    parser.add_argument("checkpoint", type=str, help="checkpoint to start from")
    parser.add_argument("-w", "--work_dir", type=str, default=None, help="working directory path")

    parser.add_argument("--samples", type=int, default=None, help="number of samples per gpu")
    parser.add_argument("--workers", type=int, default=4, help="number of workers for dataloader")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm"], default="none")
    parser.add_argument("--tcp_port", type=int, default=18888, help="tcp port for distrbuted training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument("--seed", action="store_true", default=False, help="")
    parser.add_argument("--start_epoch", type=int, default=0, help="")

    return parser.parse_args()


def init_with_args(cfg: DictConfig, args: argparse.Namespace) -> DictConfig:
    """Update config with command line arguments.

    Args:
    ----
        cfg (DictConfig): Config.
        args (argparse.Namespace): Command line arguments.

    Returns:
    -------
        DictConfig: Updated config.

    """
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir)

    if args.samples is not None:
        cfg.dataset.samples_per_gpu = args.samples

    if args.workers is not None:
        cfg.dataset.workers_per_gpu = args.workers

    if os.getenv("CUDA_VISIBLE_DEVICES", None) is None:
        num_devices = get_num_devices()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([f"{i}" for i in range(num_devices)])

    return cfg


def main() -> None:
    """Run test."""
    args = parse_args()
    cfg = Config.from_file(args.config)

    if args.launcher == "none":
        is_distributed = False
    else:
        if args.launcher == "pytorch":
            _, cfg.local_rank = init_dist_pytorch(args.tcp_port, args.local_rank, backend="nccl")
        elif args.launcher == "slurm":
            _, cfg.local_rank = init_dist_slurm(args.tcp_port, args.local_rank, backend="nccl")
        is_distributed = True

    cfg = init_with_args(cfg, args)
    cfg.work_dir = os.path.join(cfg.work_dir, "eval")
    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir, exist_ok=True)

    log_file = os.path.join(cfg.work_dir, f"log_eval_{datetime.now(tz=timezone.utc).strftime('%Y%m%d-%H%M%S')}.txt")
    logger = create_logger(log_file, rank=cfg.local_rank)

    # log to file
    logger.info("**********************Start logging**********************")
    gpu_list = os.getenv("CUDA_VISIBLE_DEVICES", None)
    logger.info(f"CUDA_VISIBLE_DEVICES={gpu_list}")

    # test_loader = build_dataloader(cfg.dataset, is_distributed=is_distributed, training=False, seed=args.seed)

    model = build_model(cfg.model)
    model.cuda()
    model, args.start_epoch = load_checkpoint(model, args.checkpoint, is_distributed=is_distributed)
    print(model.named_buffers)
    # runner = TestRunner(config=cfg, model=model, test_loader=test_loader, is_distributed=is_distributed, logger=logger)
    # with torch.no_grad():
    #     _ = runner.evaluate()


if __name__ == "__main__":
    main()
