from __future__ import annotations

import inspect
from typing import Any, Callable, Iterable, KeysView

from omegaconf import DictConfig

__all__ = (
    "build_from_cfg",
    "DETECTORS",
    "ENCODERS",
    "DECODERS",
    "LAYERS",
    "LOSSES",
    "DATASETS",
    "TRANSFORMS",
    "SCENARIO_FILTERS",
    "LR_SCHEDULERS",
    "OPTIMIZERS",
)


def build_from_cfg(cfg: dict | DictConfig, manager: ModuleManager) -> Any:
    """Return any objects from config.

    Args:
    ----
        cfg (dict | DictConfig): Any configuration.
        manager (ModuleManager): Module manager.

    Raises:
    ------
        TypeError: If input config is not dict.
        TypeError: If input manager is not ModuleManager.
        TypeError: If name in config is not both str and class.

    Returns:
    -------
        Any: Any class registered in manager.

    """
    if not isinstance(cfg, (dict, DictConfig)):
        msg = f"type of cfg must be a dict, but got {type(cfg)}"
        raise TypeError(msg)

    if not isinstance(manager, ModuleManager):
        msg = f"type of module must a ModuleManager, but got {type(manager)}"
        raise TypeError(msg)

    # Copy cfg not to destroy
    cfg_args = cfg.copy()

    obj_name: str | object = cfg_args.pop("name")
    if isinstance(obj_name, str):
        obj_cls = manager.get(obj_name)
    elif inspect.isclass(obj_name):
        obj_cls = obj_name
    else:
        msg = f"type of name must be str or valid type, but got {type(obj_name)}"
        raise TypeError(msg)

    return obj_cls(**cfg_args)


class ModuleManager:
    """A manager to map strings to classes.

    Args:
    ----
        name (str): Manger name
        build_func (Callable | None, optional): Build function to construct instance from registry

    """

    def __init__(self, name: str, build_func: Callable | None = None) -> None:
        self._name = name
        self._module_dict = {}

        # Set build func
        if build_func is None:
            self.build_func = build_from_cfg
        else:
            self.build_func = build_func

    def __len__(self) -> int:
        return len(self._module_dict)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def ___repr__(self) -> str:
        return self.__class__.__name__ + f", name={self.name}, items={self.module_dict}"

    def keys(self) -> KeysView[str]:
        return self._module_dict.keys()

    def __add__(self, other: ModuleManager) -> ModuleManager:
        self._module_dict.update(other.module_dict)
        return self

    def __sub__(self, other: ModuleManager) -> ModuleManager:
        pop_keys = set(self.keys()) & set(other.keys())
        for key in pop_keys:
            self._module_dict.pop(key)
        return self

    @property
    def name(self) -> str:
        return self._name

    @property
    def module_dict(self) -> dict:
        return self._module_dict

    def get(self, key: str) -> Any:
        """Get registered modules with key.

        Args:
        ----
            key (str): The class name in string format
        Returns:
            class: The corresponding class.

        """
        if key not in self._module_dict:
            msg = f"{key} is not registered yet"
            raise KeyError(msg)
        return self._module_dict[key]

    def build(self, *args: Any, **kwargs: Any) -> Any:
        return self.build_func(manager=self, *args, **kwargs)  # noqa: B026

    def register(
        self,
        name: str | Iterable[str] | None = None,
        *,
        force: bool = False,
        module: object = None,
    ) -> Callable:
        """Register a module.

        Args:
        ----
            name (str | Iterable[str] | None, optional): Name to be registered. Defaults to None.
            force (bool, optional): Whether to force update if there is the already registered module in same name.
                Defaults to False.
            module (object, optional): Module to be registered. Defaults to None.

        Raises:
        ------
            TypeError: _description_
            TypeError: _description_

        Returns:
        -------
            Callable: _description_

        """
        if name is not None and not isinstance(name, (str, list, tuple)):
            msg = f"name must be a str or list, but got {type(name)}"
            raise TypeError(msg)

        if module is not None:
            # Use as a normal function
            # >>> x.register(module=``obj``)
            self._add_module(module, name=name, force=force)
            return module

        def _register_decorator(obj: object) -> object:
            # Use as a decoretor
            # >>> @x.register(...).
            self._add_module(obj, name=name, force=force)
            return obj

        return _register_decorator

    def _add_module(self, module: object, name: str | Iterable[str] | None = None, *, force: bool = False) -> None:
        """Register class object.

        Args:
        ----
            module (object): Class object.
            name (str | Iterable[str] | None, optional): Name of class. Defaults to None.
            force (bool, optional): Whether to force update if there is the already registered module in same name.
                Defaults to False.

        Raises:
        ------
            TypeError: If the input module is not class.
            KeyError: If the force is False and there is the already registered module in same name.

        """
        if not inspect.isclass(module):
            msg = f"module must be a class, but got {type(module)}"
            raise TypeError(msg)

        name = module.__name__ if name is None else name
        name_list = [name] if isinstance(name, str) else name

        for nm in name_list:
            if not force and nm in self.module_dict:
                # skip registering if the module has already been registered
                continue
            self._module_dict[nm] = module


DETECTORS = ModuleManager("detectors")
ENCODERS = ModuleManager("encoders")
DECODERS = ModuleManager("decoders")
LAYERS = ModuleManager("layers")
LOSSES = ModuleManager("losses")
DATASETS = ModuleManager("datasets")
TRANSFORMS = ModuleManager("transforms")
SCENARIO_FILTERS = ModuleManager("scenario_filters")
LR_SCHEDULERS = ModuleManager("lr_schedulers")
OPTIMIZERS = ModuleManager("optimizers")
