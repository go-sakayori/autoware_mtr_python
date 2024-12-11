import functools
import inspect
import types
import warnings
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable

from .constants import IR, Backend


class Checker(ABC):
    def __init__(self, required: Any) -> None:
        """Construct instance.

        Args:
        ----
            required (Any): Required enum for inherent checkers.

        """
        self.required = required

    @abstractmethod
    def check(self, env: dict) -> bool:
        """Check whether the rewriter is valid in the environment.

        Args:
        ----
            env (dict): Environment info.

        Returns:
        -------
            bool: Return `True` if check is passed.

        """


class IRChecker(Checker):
    def __init__(self, required: IR) -> None:
        """Construct instance.

        Args:
        ----
            required (IR): Required IR info.

        """
        super().__init__(required=required)

    def check(self, env: dict) -> bool:
        """Check whether IR information is valid in the environment.

        Args:
        ----
            env (dict): Environment info.

        Returns:
        -------
            bool: Return `True` if check is passed.

        """
        # TODO: return env["ir"] == self.required
        return env["ir"] == IR.DEFAULT or env["ir"] == self.required


class BackendChecker(Checker):
    def __init__(self, required: Backend) -> None:
        """Construct instance.

        Args:
        ----
            required (Backend): Required backend info.

        """
        super().__init__(required=required)

    def check(self, env: dict) -> bool:
        """Check whether backend information is valid in the environment.

        Args:
        ----
            env (dict): Environment info.

        Returns:
        -------
            bool: Return `True` if check is passed.

        """
        # TODO: return env["backend"] == self.required
        return env["backend"] == Backend.DEFAULT or env["backend"] == self.required


class RewriterRegistry:
    """A registry that records rewrite objects.

    Example:
    -------
        >>> FUNCTION_REGISTRY = RewriterRegistry()
        >>>
        >>> @FUNCTION_REGISTRY.register(backend="default")
        >>> def add(a: int, b: int) -> int:
        >>>     return a + b
        >>>
        >>> records = FUNCTION_REGISTRY.get(
        ...     "default"
        ... )

    """

    def __init__(self) -> None:
        """Construct instance."""
        self._rewrite_records: dict[str, list[dict]] = {}

    def get(self, env: dict) -> list[dict]:
        """Return all registered records that are valid in the given environment from record table.

        Args:
        ----
            env (dict): Environment dict that includes backend, IR, codebase version, etc.

        Returns:
        -------
            list[dict]: A list that includes valid records.

        """
        default_records = []
        records = []
        for origin_function, rewriter_records in self._rewrite_records.items():
            default_rewriter = None
            final_rewriter = None
            for record in rewriter_records:
                checkers: list[Checker] = record["_checkers"]

                # check if the rewriter is default rewriter
                if len(checkers) == 0:
                    # process the default rewriter exceptionally
                    if default_rewriter is None:
                        default_rewriter = record
                    else:
                        warnings.warn(
                            f"Found multiple valid rewriters for {origin_function}, use the first rewriter.",
                            stacklevel=2,
                        )
                else:
                    # check if the checker is valid.
                    # the checker is valid only if all the checks are passed
                    is_valid = all(checker.check(env) for checker in checkers)

                    if is_valid:
                        # check if there are multiple valid rewriters
                        if final_rewriter is not None:
                            warnings.warn(
                                f"Found multiple valid rewriters for {origin_function}, use the first rewriter.",
                                stacklevel=2,
                            )
                        else:
                            final_rewriter = record

            # append the final rewriter
            # if there is no valid rewriter, try not apply default rewriter
            if final_rewriter is not None:
                records.append((origin_function, final_rewriter))
            elif default_rewriter is not None:
                default_records.append((origin_function, default_rewriter))

        # make the default records como to the front of list because we may
        # want to the non-default records to override them.
        return default_records + records

    def register(
        self,
        name: str,
        backend: str | Backend,
        ir: IR,
        extra_checkers: Checker | list[Checker] | None = None,
        **kwargs: Any,
    ) -> Callable:
        """Register an object.

        Args:
        ----
            name (str): The import path to access the function/module.
            backend (str | Backend): The name of backend where rewriter will be activated.
            ir (IR): The IR enumeration which the rewriter will be activated.
            extra_checkers (Checker | list[Checker] | None, optional): Other requirements for the rewriters.
                Defaults to None.
            **kwargs (Any): ...

        Returns:
        -------
            Callable: The decorator.

        """
        if extra_checkers is None:
            extra_checkers = []
        elif isinstance(extra_checkers, Checker):
            extra_checkers = [extra_checkers]

        def decorator(obj: Any) -> Any:
            self.__register_impl(name, backend, ir, extra_checkers, _object=obj, **kwargs)
            return obj

        return decorator

    def __register_impl(
        self,
        name: str,
        backend: str | Backend,
        ir: IR,
        extra_checkers: list[Checker],
        **kwargs: Any,
    ) -> None:
        """Run the registration invoked in `self.register()`.

        Args:
        ----
            name (str): Name of the object to register.
            backend (str | Backend): Backend name.
            ir (IR): IR name.
            extra_checkers (list[Checker]): List of extra checkers.
            **kwargs (Any): ...

        """
        # merge checkers to kwargs
        record_dict = kwargs

        if isinstance(backend, str):
            backend = Backend.get(backend)

        # try to create a checker according to 'backend'
        if backend != Backend.DEFAULT:
            extra_checkers.append(BackendChecker(backend))

        # try to create a checker according to 'ir'
        if ir != IR.DEFAULT:
            extra_checkers.append(IRChecker(ir))

        record_dict["_checkers"] = extra_checkers

        # there may be multiple rewriters of function/module.
        # we use a list to store the rewriters of a function/module.
        if name not in self._rewrite_records:
            self._rewrite_records[name] = []
        self._rewrite_records[name].append(record_dict)

    def remove(self, obj: Any, filter_cb: Callable | None = None) -> None:
        """Remove record.

        Args:
        ----
            obj (Any): The object to be remove.
            filter_cb (Callable | None, optional): Check if the object need to be remove.
                Defaults to None.

        """
        key_to_pop: list[tuple[str, dict]] = []
        for key, records in self._rewrite_records.items():
            for rec in records:
                if rec["_object"] == obj:
                    if filter_cb is not None and filter_cb(rec):
                        continue
                    key_to_pop.append((key, rec))

        for key, rec in key_to_pop:
            records = self._rewrite_records[key]
            records.remove(rec)
            if len(records) == 0:
                self._rewrite_records.pop(key)


class ContextCaller:
    """A callable object used in RewriteContext.

    Args:
    ----
        func (Callable): The rewritten function to call.
        origin_func (Callable): The function that is going to be rewritten.
            Note that in symbolic function `origin_func` may be `None`.

    Example:
    -------
        >>> @FUNCTION_REWRITER.register(func_name="torch.add")
        >>> def func(x, y):
        >>> # ctx is an instance of ContextCaller
        >>>     ctx = FUNCTION_REWRITER.get_context()
        >>>     return x + y

    """

    def __init__(self, func: Callable, origin_func: Callable, **kwargs: Any) -> None:
        """Construct instance.

        Args:
        ----
            func (Callable): Callable function.
            origin_func (Callable): Original function to be rewritten.
            **kwargs (Any): ...

        """
        self.func = func
        self.origin_func = origin_func
        # PyTorch will do annotation check on symbolic function
        # update the annotation so ContextCaller can pass the check.
        if origin_func is not None:
            wraps(origin_func)(self)
        else:
            self.__annotations__ = getattr(func, "__annotations__", {})

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call self.func directly."""
        return self.func(self, *args, **kwargs)

    def get_wrapped_caller(self) -> Callable:
        """Return a wrapped caller.

        Returns
        -------
            Callable: Callable function.

        """

        def wrapper(*args: Any, **kwargs: Any) -> Callable:
            return self.func(self, *args, **kwargs)

        return wrapper


def get_func_qualname(func: Callable) -> str:
    """Get function name."""
    assert isinstance(func, Callable), f"{func} is not a Callable object."
    _func_name = None
    if hasattr(func, "__qualname__"):
        _func_name = f"{func.__module__}.{func.__qualname__}"
    elif hasattr(func, "__class__"):
        _func_name = func.__class__
    else:
        _func_name = str(func)
    return _func_name


def get_frame_func(top: int = 1) -> Callable:
    """Get function of frame."""
    frameinfo = inspect.stack()[top]
    frame = frameinfo.frame

    g_vars = frame.f_globals
    func_name = frameinfo.function
    assert func_name in g_vars, f"Can not find function: {func_name} in global."
    return g_vars[func_name]


def import_function(path: str) -> tuple[Callable, type | None]:
    """Import and evaluate a function. If the function is defined in a class, evaluate the class additionally.

    Args:
    ----
        path (str): The path to evaluate.

    Returns:
    -------
        Callable: The function of evaluation.
        type | None: The class of evaluation if the function is defined in a class, or None.

    """
    split_path = path.split(".")
    for i in range(len(split_path), 0, -1):
        try:
            exec(f"import {'.'.join(split_path[:i])}")  # noqa: S102
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Exception occurred: {e}", stacklevel=2)
            continue

    obj = eval(path)  # noqa: PGH001, S307

    # the path that might be a class
    previous_obj = eval(".".join(split_path[:-1]))  # noqa: PGH001, S307

    # check if the path leads to a class
    if inspect.isclass(previous_obj):
        return obj, previous_obj
    else:
        return obj, None


def copy_function(f: types.FunctionType) -> types.FunctionType:
    """Copy the function."""
    # copy the global so we can get different func for different origin.
    glb = f.__globals__.copy()
    name = f.__name__
    g = types.FunctionType(f.__code__, glb, name=name, argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    glb[name] = g
    return g


def collect_env(backend: Backend, ir: IR, **kwargs: Any) -> dict:
    """Collect current environment information, including backend, ir and version.

    Args:
    ----
        backend (Backend): Current backend.
        ir (IR): Current IR.
        **kwargs (Any): ...

    Returns:
    -------
        dict: Record of the environment information.

    """
    env = {"backend": backend, "ir": ir}
    env.update(kwargs)
    return env


def eval_with_import(path: str) -> Any:
    """Evaluate the string as Python script.

    Args:
    ----
        path (str): The path to evaluate.

    Returns:
    -------
        Any: The result of evaluation.

    """
    split_path = path.split(".")
    for i in range(len(split_path), 0, -1):
        try:
            module_path = ".".join(split_path[:i])
            exec(f"import {module_path}")  # noqa: S102
            break
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Exception occurred: {e}", stacklevel=2)
            continue
    return eval(path)  # noqa: PGH001, S307
