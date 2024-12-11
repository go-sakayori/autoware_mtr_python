import pickle
from pathlib import Path
from typing import Any

import yaml

__all__ = ("load_yaml", "save_yaml", "load_pkl", "save_pkl")


def load_yaml(filename: str | Path) -> dict:
    """Load yaml file and return as dict.

    Args:
    ----
        filename (str | Path): File path.

    Returns:
    -------
        dict: Loaded dict object.

    """
    with open(filename) as f:
        return yaml.safe_load(f)


def save_yaml(data: Any, filename: str) -> None:
    """Save data into yaml file.

    Args:
    ----
        data (Any): Save data.
        filename (str): Save path.

    """
    with open(filename, "w") as f:
        yaml.safe_dump(data, f, indent=4, allow_unicode=False)


def load_pkl(filename: str | Path) -> Any:
    """Load pickle file.

    Args:
    ----
        filename (str | Path): File path.

    Returns:
    -------
        Any: Loaded object.

    """
    with open(filename, "rb") as f:
        return pickle.load(f)  # noqa: S301


def save_pkl(data: Any, filename: str) -> None:
    """Save input data to pickle.

    Args:
    ----
        data (Any): _description_
        filename (str): _description_

    """
    with open(filename, "wb") as f:
        pickle.dump(data, f)
