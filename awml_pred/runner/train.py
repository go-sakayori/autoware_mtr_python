from __future__ import annotations

import math
import os.path as osp
from numbers import Number
from typing import TYPE_CHECKING, Final

import torch
import tqdm
from omegaconf import DictConfig
from tensorboardX import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_

from awml_pred.common import get_checkpoint_state, is_main_process, items2device
from awml_pred.common import save_checkpoint as save_ckpt

from .base import BaseRunner

if TYPE_CHECKING:
    from awml_pred.typing import DataLoader, DeviceLike, Logger, LRScheduler, Module, Optimizer, Tensor


class TrainRunner(BaseRunner):
    def __init__(
        self,
        config: DictConfig,
        model: Module,
        train_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        *,
        is_distributed: bool = False,
        logger: Logger | None = None,
        test_loader: DataLoader | None = None,
        device: DeviceLike = "cuda",
    ) -> None:
        """Construct instance.

        Args:
        ----
            config (DictConfig): Experiment configuration.
            model (Module): `Module`instance.
            train_loader (DataLoader): `DataLoader` instance for training.
            optimizer (Optimizer): Learning rate optimizer.
            scheduler (LRScheduler): Learning rate scheduler.
            is_distributed (bool, optional): Indicates whether running on a distributed environment.
                Defaults to False.
            logger (Logger | None, optional): `Logger` instance. Defaults to None.
            test_loader (DataLoader | None, optional): `DataLoader` instance for testing.
            device (DeviceLike, optional): Device name. Defaults to cuda.

        """
        super().__init__(
            config=config,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            is_distributed=is_distributed,
            logger=logger,
            device=device,
        )
        self.grad_clip_norm: float = config.get("grad_norm_clip", 1000.0)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.test_loader = test_loader

        self.tb_logger = SummaryWriter(log_dir=osp.join(self.work_dir, "tensorboard")) if is_main_process() else None

        self.total_iter_each_epoch = len(self.train_loader)
        self.batch_size: int = self.train_loader.batch_size

        self.acc_iter: int = 0
        self.start_iter: int = 0  # TODO

    def train(self) -> None:
        """Execute training."""
        with tqdm.trange(
            self.start_epoch,
            self.max_epoch,
            desc="epochs",
            dynamic_ncols=True,
            leave=is_main_process(),
        ) as tbar:
            for cur_epoch in tbar:
                self.cur_epoch = cur_epoch
                torch.cuda.empty_cache()
                if self.is_distributed and self.train_loader.sampler is not None:
                    self.train_loader.sampler.set_epoch(self.cur_epoch)
                self.__train_one_epoch(tbar)
                self.scheduler.step()
                if cur_epoch % self.ckpt_save_interval == 0 and is_main_process():
                    filename: str = osp.join(self.work_dir, f"checkpoint_epoch_{cur_epoch}.pth")
                    self.save_checkpoint(filename)
                    msg = f"Saved checkpoint: {filename}"
                    self.logger.info(msg)

                tb_dict = self.evaluate()
                if self.is_best_performance(tb_dict):
                    filename: str = osp.join(self.work_dir, "best_model.pth")
                    self.save_checkpoint(filename)
                    msg = f"Saved best checkpoint: {filename}"
                    self.logger.info(msg)

    def __train_one_epoch(self, tbar: tqdm.tqdm) -> None:
        self.model.train()
        if is_main_process():
            pbar = tqdm.tqdm(
                self.total_iter_each_epoch,
                leave=False,
                desc="train",
                dynamic_ncols=True,
            )
        else:
            pbar = None

        start_iter = self.acc_iter % self.total_iter_each_epoch

        dataloader_iter = iter(self.train_loader)
        for cur_iter in range(start_iter, self.total_iter_each_epoch):
            try:
                batch_input, _ = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(self.train_loader)
                batch_input, _ = next(dataloader_iter)
                self.logger.info("New iterator is generated")
            batch_input_cuda = items2device(batch_input, self.device)
            loss_dict: dict = self.model(**batch_input_cuda)
            loss: Tensor = loss_dict["loss"]

            self.optimizer.zero_grad()
            if torch.isnan(loss).sum() == 0:
                loss.backward()
                total_norm = clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_norm,
                )
                self.optimizer.step()
            else:
                total_norm = 0.0
                self.logger.warning("Nan detected in loss, skip loss.backward()")

            try:
                cur_lr = self.scheduler.get_last_lr()[-1]
            except AttributeError:
                cur_lr = self.optimizer.param_groups[0]["lr"]

            disp_dict: dict = loss_dict["display"]
            disp_dict.update({"loss": loss.item(), "lr": cur_lr})

            self.acc_iter += 1

            if is_main_process():
                self.__log_display(disp_dict, tbar, pbar, cur_iter)
                # log in the tensorboard
                tb_dict = loss_dict["tensorboard"]
                self.__log_tensorboard(tb_dict, tag="train", global_step=self.acc_iter)
                self.__log_tensorboard(cur_lr, tag="train/lr", global_step=self.acc_iter)
                self.__log_tensorboard(total_norm, tag="train/total_norm", global_step=self.acc_iter)

        self.cur_epoch += 1

    def save_checkpoint(self, filename: str) -> None:
        """Save current model state.

        Args:
        ----
            filename (str): Path to save checkpoint

        """
        state = get_checkpoint_state(self.model, self.optimizer, self.cur_epoch)
        save_ckpt(state, filename)

    def is_best_performance(self, tb_dict: dict) -> bool:
        """Check whether current epoch's result is best performance.

        If best is, save the result into `self.work_dir/best_eval_record.txt`.

        Args:
        ----
            tb_dict (dict): Evaluation result.

        Returns:
        -------
            bool: Indicates whether the result is best ever.

        """
        if not is_main_process():
            return False

        self.__log_tensorboard(tb_dict, tag="eval", global_step=self.cur_epoch)

        if "mAP" in tb_dict:
            best_record_file = osp.join(self.work_dir, "best_eval_record.txt")

            default_score: Final[float] = -1.0

            try:
                with open(best_record_file) as f:
                    best_src_data = f.readlines()

                best_performance = best_src_data[-1].strip().split(" ")[-1]
                best_performance = float(best_performance)
            except FileNotFoundError:
                with open(best_record_file, "a") as f:
                    pass
                best_performance = default_score

            with open(best_record_file, "a") as f:
                print(f"epoch_{self.cur_epoch} mAP {tb_dict['mAP']}", file=f)

            if best_performance == default_score or tb_dict["mAP"] > float(
                best_performance,
            ):
                with open(best_record_file, "a") as f:
                    print(
                        f"best_epoch_{self.cur_epoch} mAP {tb_dict['mAP']}",
                        file=f,
                    )
                return True
            with open(best_record_file, "a") as f:
                print(f"{best_src_data[-1].strip()}", file=f)
            return False
        msg = f"Cannot find mAP in tb_dict: {tb_dict.keys()}"
        self.logger.info(msg)
        return False

    def __log_display(
        self,
        disp_dict: dict,
        tbar: tqdm.tqdm,
        pbar: tqdm.tqdm,
        cur_iter: int,
    ) -> None:
        if (
            self.acc_iter % self.log_iter_interval == 0
            or cur_iter == self.start_iter
            or cur_iter + 1 == self.total_iter_each_epoch
        ):
            trained_time_past_all = tbar.format_dict["elapsed"]
            second_each_iter = pbar.format_dict["elapsed"] / max(
                cur_iter - self.start_iter + 1,
                1.0,
            )

            trained_time = pbar.format_dict["elapsed"]
            remain_sec_each_epoch = second_each_iter * (self.total_iter_each_epoch - cur_iter)
            remain_sec_all = second_each_iter * (
                (self.max_epoch - self.cur_epoch) * self.total_iter_each_epoch - cur_iter
            )

            disp_str = ", ".join(
                [f"{key}={val:.3f}" for key, val in disp_dict.items() if key != "lr"],
            )
            disp_str += f", lr={disp_dict['lr']}"
            msg = (
                f"epoch: {self.cur_epoch}/{self.max_epoch}, acc_iter={self.acc_iter}, "
                f"cur_iter={cur_iter}/{self.total_iter_each_epoch}, "
                f"batch_size={self.batch_size}, iter_cost={second_each_iter:.2f}s, "
                "time_cost(epoch): "
                f"{tbar.format_interval(trained_time)}/{tbar.format_interval(remain_sec_each_epoch)}, "
                "time_cost(all): "
                f"{tbar.format_interval(trained_time_past_all)}/{tbar.format_interval(remain_sec_all)}, "
                f"{disp_str}"
            )
            self.logger.info(msg)

    def __log_tensorboard(
        self,
        data: Number | dict[str, Number | Tensor | dict],
        tag: str,
        global_step: int | None = None,
    ) -> None:
        """Create a log in the tensorboard.

        Args:
        ----
            data (Number | dict[str, Number | Tensor | dict]): Source data.
            tag (str): Tag to log in the tensorboard.
            global_step (int | None, optional): Global step to log.

        """
        if self.tb_logger is None:
            self.logger.warning("No tensorboard logger exists.")
            return

        def can_plot_number(value: Number) -> bool:
            return not math.isnan(value) and not math.isinf(value)

        if isinstance(data, Number) and can_plot_number(data):
            self.tb_logger.add_scalar(tag, data, global_step=global_step)
        elif isinstance(data, torch.Tensor) and can_plot_number(data.item()):
            self.tb_logger.add_scalar(tag, data.item(), global_step=global_step)
        elif isinstance(data, dict):
            for key, item in data.items():
                self.__log_tensorboard(item, osp.join(tag, key), global_step=global_step)
