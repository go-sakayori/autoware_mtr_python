import logging
import types
from collections import defaultdict
from typing import Any, Callable, MutableSequence

from .constants import IR, Backend
from .rewriter_utils import (
    Checker,
    ContextCaller,
    RewriterRegistry,
    copy_function,
    get_frame_func,
    get_func_qualname,
    import_function,
)

try:
    try:
        from torch.fx._symbolic_trace import _wrapped_fns_to_patch
    except ImportError:
        # torch>=1.8.0,<1.10.0
        from torch.fx.symbolic_trace import _wrapped_fns_to_patch
except ImportError:
    # torch<1.8.0
    _wrapped_fns_to_patch = []


class FunctionRewriter:
    """A function rewriter which maintains rewritten functions.

    Examples
    --------
        >>> @FUNCTION_REWRITER.register(func_name="torch.Tensor.size", backend="onnx")
        >>> def size_of_tensor_static(self, *args):
        >>>     ctx = FUNCTION_REWRITER.get_context()
        >>>     ret = ctx.origin_func(self, *args)
        >>>     if isinstance(ret, torch.Tensor):
        >>>         ret = int(ret)
        >>>     else:
        >>>         ret = [int(r) for r in ret]
        >>>         ret = tuple(ret)
        >>>     return ret

    """

    def __init__(self) -> None:
        """Construct instance."""
        self._registry = RewriterRegistry()
        self._func_contexts = defaultdict(list)

    def register(
        self,
        func_name: str,
        backend: str | Backend = Backend.DEFAULT,
        ir: IR = IR.DEFAULT,
        extra_checkers: Checker | list[Checker] | None = None,
        **kwargs: Any,
    ) -> Callable:
        """Register The interface of function rewriter decorator.

        Args:
        ----
            func_name (str): The function name/path to be rewritten.
            backend (str | Backend, optional): The backend where rewriter will be activated on.
                Defaults to Backend.DEFAULT.
            ir (IR, optional): The IR where rewriter will be activated on. Defaults to IR.DEFAULT.
            extra_checkers (Checker | list[Checker] | None, optional): The other requirements. Defaults to None.
            **kwargs (Any): ...

        Returns:
        -------
            Callable: The process of registering function.

        """
        return self._registry.register(func_name, backend, ir, extra_checkers, **kwargs)

    def enter(self, env: dict | None = None, **kwargs: Any) -> None:
        """Enter the environment.

        Args:
        ----
            env (dict | None, optional): _description_. Defaults to None.
            **kwargs (Any): ...

        """
        if env is None:
            env = {}

        self._func_contexts.clear()
        # get current records
        function_records = self._registry.get(env)
        # get current fx wrapped func nums
        self._ori_fx_wrap_num = len(_wrapped_fns_to_patch)

        self._origin_functions = []
        self._additional_functions = []
        new_functions = []
        for function_path, record in function_records:
            # check if the origin function exists
            try:
                origin_func, origin_class = import_function(function_path)
            except Exception:  # noqa: BLE001
                origin_func = None
                msg = f"Can not find {function_path}, function rewrite will not be applied"
                logging.warning(msg)

            # only rewrite functions that exist
            if origin_func is not None:
                is_additional_func = False
                if origin_class is not None:
                    function_name = function_path.split(".")[-1]
                    try:
                        origin_class.__getattribute__(origin_class, function_name)
                    except Exception:  # noqa: BLE001
                        # the function is a method and it is derived from base class
                        msg = f"There is no function named: {function_name}"
                        logging.warning(msg)
                        is_additional_func = True

                if is_additional_func:
                    self._additional_functions.append(function_path)

                # save origin function
                self._origin_functions.append({"func_path": function_path, "origin_func": origin_func})

                # create context_caller
                rewrite_function = record["_object"]
                # the func before and after copy has different globals
                rewrite_function = copy_function(rewrite_function)
                extra_kwargs = kwargs.copy()
                extra_kwargs.update(record)
                context_caller = ContextCaller(rewrite_function, origin_func, **extra_kwargs)

                # If there is a function wrapped by torch.fx.wrap in
                # rewrite_function's globals, we need to wrap the same name
                # function in copied function's globals.
                _fx_wrap_copied_fn(record["_object"], context_caller.func)

                qualname = get_func_qualname(rewrite_function)
                self._func_contexts[qualname].append(context_caller)
                self._func_contexts[function_path].append(context_caller)

                # Cache new the function to avoid homonymic bug
                new_functions.append({"func_path": function_path, "origin_func": rewrite_function})

        for func_dict in new_functions:
            function_path = func_dict["func_path"]
            new_function = func_dict["origin_func"]
            # Rewrite functions
            _set_func(function_path, new_function)

    def exit(self) -> None:
        """Recover the function rewrite."""
        cur_fx_wrap_num = len(_wrapped_fns_to_patch)
        for _ in range(cur_fx_wrap_num - self._ori_fx_wrap_num):
            _wrapped_fns_to_patch.pop(-1)

        for func_dict in self._origin_functions:
            func_path = func_dict["func_path"]
            func = func_dict["origin_func"]
            _set_func(func_path, func)

        for func_path in self._additional_functions:
            _del_func(func_path)

        self._func_contexts.clear()

    def get_context(self, key: str | None = None) -> ContextCaller | None:
        """Get the context of rewriter.

        Args:
        ----
            key (str | None, optional): Key to the context. Defaults to None.

        Returns:
        -------
            ContextCaller | None: Context of function.

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


def _set_func(
    origin_func_path: str,
    rewrite_func: Callable,
    ignore_refs: tuple[Any] = (),
    ignore_keys: tuple[str] = ("origin_func",),
) -> None:
    """Set function.

    Args:
    ----
        origin_func_path (str): Import path of the original function.
        rewrite_func (Callable): Import path of the function to rewrite.
        ignore_refs (tuple[Any], optional): References to be ignored. Defaults to ().
        ignore_keys (tuple[str], optional): Keys to be ignored. Defaults to ("origin_func",).

    """
    # Import necessary module
    split_path = origin_func_path.split(".")
    for i in range(len(split_path), 0, -1):
        try:
            exec("import {}".format(".".join(split_path[:i])))  # noqa: S102
            break
        except Exception as e:  # noqa: BLE001
            logging.warning(f"Exception occurred: {e}")
            continue
    origin_func = eval(origin_func_path)  # noqa: PGH001, S307
    method_class = False
    if len(split_path) > 1:
        module_or_class = eval(".".join(split_path[:-1]))  # noqa: PGH001, S307
        if isinstance(module_or_class, type):
            method_class = True
    # Assign function
    if not method_class:
        _replace_all_obj(origin_func, rewrite_func, ignore_refs=ignore_refs, ignore_keys=ignore_keys)
    exec(f"{origin_func_path} = rewrite_func")  # noqa: S102


def _del_func(path: str) -> None:
    """Delete a function that is denoted by a path.

    Args:
    ----
        path (str): The path to evaluate.

    """
    split_path = path.split(".")
    for i in range(len(split_path), 0, -1):
        try:
            exec("import {}".format(".".join(split_path[:i])))  # noqa: S102
            exec(f"del {path}")  # noqa: S102
            break
        except Exception:  # noqa: BLE001
            logging.warning("Exception occurred.")
            continue


def _replace_all_obj(
    obj: Any,
    new_obj: Any,
    ignore_refs: tuple[Any] = (),
    ignore_keys: tuple[str] = (),
) -> None:
    """Replace all object reference with new_object.

    Args:
    ----
        obj (Any): The object to be replaced.
        new_obj (Any): The object to replace obj.
        ignore_refs (Tuple[Any]): These refs will be ignored. Defaults to ().
        ignore_keys (Tuple[str]): object with these keys will be ignored. Defaults to ().

    """
    import gc

    refs = gc.get_referrers(obj)
    obj_id = id(obj)
    for ref in refs:
        if ref in ignore_refs:
            continue
        if isinstance(ref, MutableSequence):
            for i, v in enumerate(ref):
                if id(v) == obj_id:
                    ref[i] = new_obj
        elif isinstance(ref, dict):
            for k, v in ref.items():
                if id(v) == obj_id and k not in ignore_keys:
                    ref[k] = new_obj
        else:
            # TODO: check if we can replace tuple
            pass


def _fx_wrap_copied_fn(func: types.FunctionType, copied_func: types.FunctionType) -> None:
    """Check function has the attribute named `__globals__`.

    If a function is wrapped by torch.fx.wrap, its copy also needs to be
    wrapped by torch.fx.wrap.
    """
    if not hasattr(func, "__globals__"):
        return

    wrapped_fns_globals = [item[0] for item in _wrapped_fns_to_patch]
    wrapped_fns_names = [item[1] for item in _wrapped_fns_to_patch]

    # check if wrapped by torch.fx.wrap
    if func.__globals__ in wrapped_fns_globals:
        idx = wrapped_fns_globals.index(func.__globals__)
        fn_name = wrapped_fns_names[idx]
        # a hacky way to wrap the func in copied func
        _wrapped_fns_to_patch.append((copied_func.__globals__, fn_name))
