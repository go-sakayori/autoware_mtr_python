from typing import Callable, Sequence

from omegaconf import DictConfig

from awml_pred.common import TRANSFORMS

__all__ = ("Compose",)


class Compose:
    """Compose multiple transforms sequentially."""

    def __init__(self, transforms: Sequence[DictConfig | Callable] | None = None) -> None:
        """Construct instance.

        Args:
        ----
            transforms (Sequence[DictConfig | Callable] | None, optional): Sequence of transform object or
                config dict to be composed. Defaults to None.

        Raises:
        ------
            TypeError: Expecting transforms are list of dicts or callable.

        """
        self.transforms: list[Callable] = []

        if transforms is None:
            transforms = []

        for transform in transforms:
            if isinstance(transform, DictConfig):
                t = TRANSFORMS.build(transform)
                if not callable(t):
                    raise TypeError(f"transform must be a callable object, but got {type(t)}")
                self.transforms.append(t)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f"Unexpected transform type: {type(transform)}")

    def __call__(self, info: dict) -> dict:
        """Run registered transformations.

        Args:
        ----
            info (dict): Source info.

        Returns:
        -------
            dict: Output info.

        """
        for t in self.transforms:
            info = t(info)
        return info

    def __repr__(self) -> str:
        format_str = self.__class__.__name__ + "("
        for t in self.transforms:
            format_str += "\n"
            format_str += f"    {t}"
        format_str += "\n)"
        return format_str
