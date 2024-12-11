# Copyright (c) OpenMMLab. All rights reserved.
# ============================================================================
#
#   We referred to open-mmlab/mmdeploy(https://github.com/open-mmlab/mmdeploy)
#
# ============================================================================

from .rewriter_manager import FUNCTION_REWRITER, MODULE_REWRITER, SYMBOLIC_REWRITER, RewriterContext, patch_model

__all__ = ("FUNCTION_REWRITER", "MODULE_REWRITER", "SYMBOLIC_REWRITER", "RewriterContext", "patch_model")
