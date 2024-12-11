from typing import Any

from awml_pred.typing import Module

from .constants import IR, Backend
from .function_rewriter import FunctionRewriter
from .module_rewriter import ModuleRewriter
from .rewriter_utils import collect_env
from .symbolic_rewriter import SymbolicRewriter


class RewriterManager:
    """The manager that manages some rewriters."""

    def __init__(self) -> None:
        """Construct instance."""
        self.module_rewriter = ModuleRewriter()
        self.function_rewriter = FunctionRewriter()
        self.symbolic_rewriter = SymbolicRewriter()


REWRITER_MANAGER = RewriterManager()

MODULE_REWRITER = REWRITER_MANAGER.module_rewriter
FUNCTION_REWRITER = REWRITER_MANAGER.function_rewriter
SYMBOLIC_REWRITER = REWRITER_MANAGER.symbolic_rewriter


def patch_model(
    model: Module,
    backend: str | Backend = Backend.DEFAULT,
    ir: IR = IR.DEFAULT,
    *,
    recursive: bool = True,
    **kwargs: Any,
) -> Module:
    """Patch the model, replace the modules that can be rewritten.

    Note that, the original model will be modified permanently.

    Args:
    ----
        model (Module): The model to be patched.
        backend (str | Backend, optional): Backend name. Defaults to Backend.DEFAULT.
        ir (IR, optional): IR name. Defaults to IR.DEFAULT.
        recursive (bool, optional): Whether to enable recursive patching. Defaults to True.
        **kwargs (Any): ...

    Returns:
    -------
        Module: The patched model.

    """
    return MODULE_REWRITER.patch_model(model, backend, ir, recursive, **kwargs)


class RewriterContext:
    """Rewrite context.

    The context is used to manage the rewrite functions and the backend.
    """

    def __init__(
        self,
        cfg: dict | None = None,
        backend: str | Backend = Backend.DEFAULT,
        ir: IR = IR.DEFAULT,
        rewriter_manager: RewriterManager = REWRITER_MANAGER,
        **kwargs: Any,
    ) -> None:
        """Construct instance.

        Args:
        ----
            cfg (dict | None, optional): _description_. Defaults to None.
            backend (str, Backend, optional): _description_. Defaults to Backend.DEFAULT.
            ir (IR, optional): _description_. Defaults to IR.DEFAULT.
            rewriter_manager (RewriterManager, optional): _description_. Defaults to REWRITER_MANAGER.
            **kwargs (Any): ...

        """
        self._cfg = cfg if cfg is not None else {}
        self._kwargs = kwargs
        self._rewriter_manager = rewriter_manager
        if isinstance(backend, str):
            backend = Backend.get(backend)
        self._env = collect_env(backend, ir)

    def enter(self) -> None:
        """Let rewriters enter the environment.

        NOTE: currently deploy config is not used.
            - self._rewriter_manager.function_rewriter.enter(self._cfg, self._env, **self._kwargs)
            - self._rewriter_manager.symbolic_rewriter.enter(self._cfg, self._env, **self._kwargs)
        """
        self._rewriter_manager.function_rewriter.enter(self._env, **self._kwargs)
        self._rewriter_manager.symbolic_rewriter.enter(self._env, **self._kwargs)

    def exit(self) -> None:
        """Let rewriters exit from the environment."""
        self._rewriter_manager.function_rewriter.exit()
        self._rewriter_manager.symbolic_rewriter.exit()

    def __enter__(self) -> None:
        """Call `self.enter()`."""
        self.enter()

    def __exit__(self, _type: object, _val: object, _tb: object) -> None:
        """Call `self.exit()`."""
        self.exit()
