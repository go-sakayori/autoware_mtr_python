import inspect
from typing import Any, Callable

from awml_pred.typing import Module

from .constants import IR, Backend
from .rewriter_utils import Checker, RewriterRegistry, collect_env, eval_with_import


class ModuleRewriter:
    def __init__(self) -> None:
        """Construct instance."""
        self._registry = RewriterRegistry()

    def register(
        self,
        module_type: str,
        backend: str | Backend = Backend.DEFAULT,
        ir: IR = IR.DEFAULT,
        extra_checkers: Checker | list[Checker] | None = None,
        **kwargs: Any,
    ) -> Callable:
        """Register interface of module rewriter decorator.

        Args:
        ----
            module_type (str): Name of module type to rewrite.
            backend (str | Backend, optional): Backend name. Defaults to Backend.DEFAULT.
            ir (IR, optional): IR name. Defaults to IR.DEFAULT.
            extra_checkers (Checker | list[Checker] | None, optional): Other requirements defined by Checker.
                Defaults to None.
            **kwargs (Any): ...

        Returns:
        -------
            Callable: The rewritten model.

        """
        return self._registry.register(module_type, backend, ir, extra_checkers, **kwargs)

    def patch_model(
        self,
        model: Module,
        backend: str | Backend = Backend.DEFAULT,
        ir: IR = IR.DEFAULT,
        *,
        recursive: bool = True,
        **kwargs: Any,
    ) -> Module:
        """Replace the model that was registered.

        Args:
        ----
            model (Module): `Module` instance.
            backend (Backend, optional): Backend name. Defaults to Backend.DEFAULT.
            ir (IR, optional): IR name. Defaults to IR.DEFAULT.
            recursive (bool, optional): Whether to run recursively. Defaults to True.
            **kwargs (Any): ...

        Returns:
        -------
            Module: Patched model.

        """
        if isinstance(backend, str):
            backend = Backend.get(backend)

        env = collect_env(backend, ir)
        self._collect_record(env)
        return self._replace_module(model, recursive, **kwargs)

    def _replace_module(self, _model: Module, *, recursive: bool, **_kwargs: Any) -> Module:
        def _replace_module_impl(model: Module, **kwargs: Any) -> Module:
            if type(model) in self._cls_set:
                # skip if model has already been patched.
                return model

            if recursive and hasattr(model, "named_children"):
                for name, module in model.named_children():
                    model._modules[name] = _replace_module_impl(module, **kwargs)  # noqa: SLF001

            return self._replace_one_module(model, **kwargs)

    def _replace_one_module(self, module: Module, **kwargs: Any) -> Module:
        """Build rewritten model.

        Args:
        ----
            module (_type_): `Module` instance.
            **kwargs (Any): ...

        Returns:
        -------
            Module: `Module` instance.

        """
        object_candidate_dict = {}
        for k, v in self._records.items():
            if isinstance(module, k):
                object_candidate_dict[k] = v
        if len(object_candidate_dict) == 0:
            return module

        type_sequence = [type(module)]
        while len(type_sequence) > 0:
            module_type = type_sequence.pop(0)
            if module_type == object:
                continue
            object_dict = object_candidate_dict.get(module_type)
            if object_dict is not None:
                break
            type_sequence.extend(module_type.__bases__)

        module_class = object_dict["_object"]

        # Pop arguments that are not supported
        input_args = kwargs.copy()
        supported_args = inspect.getfullargspec(module_class.__init__).args
        redundant_key_name = []
        for k in input_args:
            if k not in supported_args:
                redundant_key_name.append(k)
        for k in redundant_key_name:
            input_args.pop(k)

        return module_class(module, **kwargs)

    def _collect_record(self, env: dict) -> None:
        self._records = {}
        self._cls_set = set()
        records = self._registry.get(env)
        for name, kwargs in records:
            self._cls_set.add(kwargs["_object"])
            self._records[eval_with_import(name)] = kwargs
