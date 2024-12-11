from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, ListConfig, OmegaConf

if TYPE_CHECKING:
    from pathlib import Path


class Config(OmegaConf):
    """Configuration class."""

    @staticmethod
    def from_file(filename: str | Path) -> DictConfig | ListConfig:
        """Construct instance from file.

        Args:
        ----
            filename (str | Path): File path to config.
            nest (bool, optional): Whether to set nested items as attribute.

        Returns:
        -------
            Config: Constructed instance.

        Warning:
        -------
            This method is deprecated, please use `Config.load(...)`.

        """
        ret = Config.load(filename)

        if ret.get("local_rank", None) is None:
            ret.setdefault("local_rank", 0)

        # import custom modules from paths
        for name in ret.get("custom_imports", []):
            import_module(name)

        return ret

    def get(self, key: str, default: Any | None = None) -> Any | None:
        """Get a specified attribute.

        Args:
        ----
            key (str): Name of the attribute.
            default (Any | None, optional): Default value. Defaults to None.

        Returns:
        -------
            Any | None: Got value.

        """
        return getattr(self, key, default)
