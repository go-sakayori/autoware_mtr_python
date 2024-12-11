from __future__ import annotations

from typing import TYPE_CHECKING

from omegaconf import DictConfig

from .base import BaseRunner

if TYPE_CHECKING:
    from awml_pred.typing import DataLoader, DeviceLike, Logger, Module


class TestRunner(BaseRunner):
    def __init__(
        self,
        config: DictConfig,
        model: Module,
        test_loader: DataLoader,
        *,
        is_distributed: bool = False,
        logger: Logger | None = None,
        device: DeviceLike = "cuda",
    ) -> None:
        """Construct instance.

        Args:
        ----
            config (DictConfig): Experiment configuration.
            model (Module): `Module` instance.
            test_loader (DataLoader): `DataLoader` for testing.
            is_distributed (bool, optional): Indicates whether running on the distributed environment.
                Defaults to False.
            logger (Logger | None, optional): `Logger` instance. Defaults to None.
            device (DeviceLike, optional): Device name. Defaults to cuda.

        """
        super().__init__(
            config=config,
            model=model,
            train_loader=None,
            test_loader=test_loader,
            is_distributed=is_distributed,
            logger=logger,
            device=device,
        )
