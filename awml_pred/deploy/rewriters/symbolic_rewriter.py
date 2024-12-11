import logging
from collections import defaultdict
from typing import Any, Callable, Sequence

import torch
from torch.autograd import Function
from torch.onnx.symbolic_helper import parse_args

from .constants import IR, Backend
from .rewriter_utils import (
    Checker,
    ContextCaller,
    RewriterRegistry,
    copy_function,
    eval_with_import,
    get_frame_func,
    get_func_qualname,
)


class SymbolicRewriter:
    def __init__(self) -> None:
        """Construct instance."""
        self._registry = RewriterRegistry()
        self._func_contexts = defaultdict(list)

    def register(
        self,
        func_name: str,
        backend: str | Backend = Backend.DEFAULT,
        *,
        is_pytorch: bool = False,
        arg_descriptors: Sequence[str] | None = None,
        ir: IR = IR.DEFAULT,
        extra_checkers: Checker | list[Checker] | None = None,
        **kwargs: Any,
    ) -> Callable:
        """Register the custom symbolic.

        Args:
        ----
            func_name (str): The function name/path to override the symbolic.
            backend (str | Backend, optional): Name of backend which the rewriter will be activated.
                Defaults to Backend.DEFAULT.
            is_pytorch (bool, optional): Enable this flag if `func_name` is the name of a pytorch builtin function.
                Defaults to False.
            arg_descriptors (Sequence[str] | None, optional): The argument descriptors of the symbol. Defaults to None.
            ir (IR, optional): Name of ir which the rewriter will be activated. Defaults to IR.DEFAULT.
            extra_checkers (Checker | list[Checker] | None, optional): Other requirements defined by Checker.
                Defaults to None.
            **kwargs (Any): ...

        Returns:
        -------
            Callable:

        """
        return self._registry.register(
            func_name,
            backend,
            ir,
            extra_checkers,
            is_pytorch=is_pytorch,
            arg_descriptors=arg_descriptors,
            **kwargs,
        )

    def enter(self, env: dict | None = None, opset: int = 11, **kwargs: Any) -> None:
        """Enter to the environment.

        Args:
        ----
            env (dict | None, optional): Environment info. Defaults to None.
            opset (int, optional): Opset version. Defaults to 11.
            **kwargs (Any): ...

        """
        if env is None:
            env = {}

        self._func_contexts.clear()

        symbolic_records = self._registry.get(env)

        self._pytorch_symbolic = []
        self._extra_symbolic = []
        new_functions = []
        for func_name, record_dict in symbolic_records:
            symbolic_function = record_dict["_object"]
            symbolic_function = copy_function(symbolic_function)
            arg_descriptors = record_dict["arg_descriptors"]
            extra_kwargs = kwargs.copy()
            extra_kwargs.update(record_dict)
            context_caller = ContextCaller(symbolic_function, None, **extra_kwargs)

            # register context
            qualname = get_func_qualname(symbolic_function)
            self._func_contexts[qualname].append(context_caller)
            self._func_contexts[func_name].append(context_caller)

            if arg_descriptors is not None and len(arg_descriptors) > 0:
                symbolic_function = parse_args(*arg_descriptors)(symbolic_function)

            is_pytorch: bool = record_dict["is_pytorch"]
            if is_pytorch:
                from torch.onnx import register_custom_op_symbolic

                register_custom_op_symbolic(f"::{func_name}", symbolic_function, opset)

                # save domain and version
                self._pytorch_symbolic.append((func_name, "", opset))
            else:
                # check if the origin function exists
                try:
                    origin_func = eval_with_import(func_name)
                    assert issubclass(origin_func, Function), f"{func_name} is not an torch.autograd.Function"
                except Exception:  # noqa: BLE001
                    origin_func = None
                    msg = f"Can not add symbolic for {func_name}"
                    logging.warning(msg)

                # only register functions that exist
                if origin_func is not None:
                    origin_symbolic = getattr(origin_func, "symbolic", None)

                    # save origin function
                    self._extra_symbolic.append((origin_func, origin_symbolic))

                    # cache new the function to avoid homonymic bug
                    new_functions.append((origin_func, symbolic_function))

            for origin_func, new_func in new_functions:
                origin_symbolic = getattr(origin_func, "symbolic", None)
                new_func.origin_func = origin_symbolic
                origin_func.symbolic = new_func

    def exit(self) -> None:
        """Exit from the environment."""
        self._func_contexts.clear()

        # unregister pytorch op
        if hasattr(torch.onnx, "unregister_custom_op_symbolic"):
            from torch.onnx import unregister_custom_op_symbolic

            for func_name, _, version in self._pytorch_symbolic:
                unregister_custom_op_symbolic(f"::{func_name}", version)
        else:
            from torch.onnx.symbolic_registry import _registry as pytorch_registry

            for func_name, domain, version in self._pytorch_symbolic:
                del pytorch_registry[(domain, version)][func_name]
                if not pytorch_registry[(domain, version)]:
                    del pytorch_registry[(domain, version)]

        # unregister custom op
        for origin_func, origin_symbolic in self._extra_symbolic:
            origin_func.symbolic = origin_symbolic

    def get_context(self, key: str | None = None) -> ContextCaller:
        """Return the context.

        Args:
        ----
            key (str | None, optional): Key of the context. Defaults to None.

        Returns:
        -------
            ContextCaller: Collected context.

        """
        func = None
        if key is None:
            func = get_frame_func(2)
            key = get_func_qualname(func)

        # get all contexts
        ctxs = self._func_contexts.get(key, [])

        if func is None:
            assert len(ctxs) == 1
            return ctxs[0]

        ctx = None
        for tmp_ctx in ctxs:
            if tmp_ctx.func == func:
                ctx = tmp_ctx

        if ctx is None:
            msg = f"Can not found context of {key}"
            logging.warning(msg)
        return ctx
