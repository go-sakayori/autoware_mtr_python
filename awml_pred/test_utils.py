from typing import Any

import pytest
import torch


def require_cuda(test_case: Any) -> None:
    """Return a decorator making a test that requires CUDA.

    These tests are skipped when CUDA isn't installed.

    Args:
    ----
        test_case (Any): Any test cases.

    """
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available.")(test_case)
