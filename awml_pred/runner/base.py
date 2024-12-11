from __future__ import annotations

import os.path as osp
from abc import ABC
from dataclasses import asdict
from time import time
from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from awml_pred.common import create_logger, is_main_process, items2device, merge_dist_results, save_pkl

if TYPE_CHECKING:
    from tensorboardX import SummaryWriter

    from awml_pred.dataclass import EvaluationData
    from awml_pred.typing import DataLoader, Dataset, DeviceLike, Logger, Module


class BaseRunner(ABC):  # noqa: B024
    """Abstract base class of runner."""

    def __init__(
        self,
        config: DictConfig,
        model: Module,
        train_loader: DataLoader | None,
        test_loader: DataLoader | None,
        *,
        is_distributed: bool = False,
        logger: Logger | None = None,
        device: DeviceLike = "cuda",
    ) -> None:
        """Construct instance.

        Args:
        ----
            config (Config): Experiment configuration.
            model (Module): `Module`instance.
            train_loader (DataLoader | None): `DataLoader` instance for training.
            test_loader (DataLoader | None): `DataLoader` instance for testing.
            is_distributed (bool, optional): Indicates whether running on a distributed environment.
                Defaults to False.
            logger (Logger | None, optional): `Logger` instance. Defaults to None.
            device (DeviceLike, optional): Device name. Defaults to cuda.

        """
        super().__init__()
        self.work_dir: str = config.work_dir
        self.log_iter_interval: int = config.log_iter_interval
        self.ckpt_save_interval: int = config.ckpt_save_interval

        self.start_epoch: int = config.start_epoch
        self.max_epoch: int = config.max_epoch

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.is_distributed = is_distributed

        if logger is None:
            self.logger = create_logger()
        else:
            self.logger = logger

        self.device = torch.device(device)

        self.cur_epoch = self.start_epoch
        self.tb_logger: SummaryWriter | None = None
        self.tb_image_ids: list[str] | None = None

    def evaluate(self, test_loader: DataLoader | None = None) -> dict:
        """Evaluate step.

        Args:
        ----
            test_loader (DataLoader | None, optional): If None, `self.test_loader` is candidate of loader.
                Moreover, both of them are None, skip evaluation step. Defaults to None.

        Returns:
        -------
            dict: Evaluated result.

        """
        if test_loader is None:
            test_loader = self.test_loader

        if test_loader is None:
            self.logger.info("Skip evaluation step because test_loader is None")
            return {"mAP": 0.0}

        self.model.eval()

        pbar = None
        if is_main_process():
            pbar = tqdm(
                total=len(test_loader),
                leave=True,
                desc="eval",
                dynamic_ncols=True,
            )

        eval_data_list: list[EvaluationData] = []
        total_iter = len(test_loader)
        dataset: Dataset = test_loader.dataset
        start_time = time()
        last_iter: int = 0
        for batch_iter, (batch_input, batch_meta) in enumerate(test_loader):
            with torch.no_grad():
                batch_input_cuda = items2device(batch_input, self.device)
                pred_scores, pred_trajs = self.model(**batch_input_cuda)
                eval_data_list += dataset.generate_prediction(
                    pred_scores,
                    pred_trajs,
                    batch_meta,
                )

            if is_main_process():
                disp_dict = {}
                batch_size = test_loader.batch_size
                self.__eval_log(disp_dict, batch_size, batch_iter, total_iter, pbar)
                pbar.update(batch_iter - last_iter)
            last_iter = batch_iter

        if is_main_process():
            pbar.close()

        eval_data = self.__merge_dist_results(eval_data_list, len(dataset))

        self.logger.info("*************** Performance of EPOCH *****************")
        sec_per_epoch = (time() - start_time) / total_iter
        msg = f"Generate label finished(sec_per_example: {sec_per_epoch:.4f} second)."
        self.logger.info(msg)

        if not is_main_process():
            return {}

        save_result = [asdict(data) for data in eval_data]
        save_pkl(save_result, osp.join(self.work_dir, "result.pkl"))
        result_str, result_dict = dataset.evaluate(eval_data)
        self.logger.info(f"\n{result_str}")
        return result_dict

    def __eval_log(
        self,
        disp_dict: dict,
        batch_size: int,
        batch_iter: int,
        total_iter: int,
        pbar: tqdm,
    ) -> None:
        """Log evaluation result in main process.

        Args:
        ----
            disp_dict (dict): Display information.
            batch_size (int): Batch size.
            batch_iter (int): Current iteration number in one epoch.
            total_iter (int): Total number of iterations in one epoch.
            pbar (tqdm): Progress bar.

        """
        if not is_main_process():
            msg = "Expected running on main process"
            raise RuntimeError(msg)
        if batch_iter % self.log_iter_interval != 0 and batch_iter != 0 and batch_iter + 1 != total_iter:
            return
        past_time = pbar.format_dict["elapsed"]
        second_each_iter = past_time / max(batch_iter, 1.0)
        remaining_time = second_each_iter * (total_iter - batch_iter)
        disp_str = ", ".join(
            [f"{key}={val:.3f}" for key, val in disp_dict.items() if key != "lr"],
        )
        msg = (
            f"eval: epoch={self.cur_epoch}, "
            f"batch_iter={batch_iter}/{total_iter}, batch_size={batch_size}, iter_cost={second_each_iter:.2f}s, "
            f"time_cost: {pbar.format_interval(past_time)}/{pbar.format_interval(remaining_time)}, "
            f"{disp_str}"
        )
        self.logger.info(msg)

    def __merge_dist_results(self, eval_data: list[EvaluationData], num_data: int) -> list[EvaluationData]:
        """Merge distributed results into single result.

        Args:
        ----
            eval_data (list[dict]): List of results.
            num_data (int): Total number of data.

        Returns:
        -------
            list[EvaluationData]: Merged result.

        """
        if self.is_distributed:
            eval_data = merge_dist_results(
                eval_data,
                num_data,
                tmpdir=osp.join(self.work_dir, "tmpdir"),
            )
        return eval_data
