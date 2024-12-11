from typing import Any

import torch
from torch.autograd.function import Function, FunctionCtx
from torch.onnx import symbolic_helper

from awml_pred.deploy.rewriters import FUNCTION_REWRITER
from awml_pred.typing import GraphCtx, Tensor

# References:
#   https://github.com/open-mmlab/mmdeploy/blob/main/mmdeploy/mmcv/ops/nms.py


class KnnBatch(Function):
    @staticmethod
    def symbolic(
        g: GraphCtx,
        xyz: Tensor,
        query_xyz: Tensor,
        batch_idxs: Tensor,
        query_batch_offsets: Tensor,
        top_k: int,
    ) -> Any:
        """Load the symbolic of this module.

        Args:
        ----
            g (GraphCtx): `GraphCtx` instance.
            xyz (Tensor): A tensor.
            query_xyz (Tensor): A tensor.
            batch_idxs (Tensor): A tensor.
            query_batch_offsets (Tensor): A tensor.
            top_k (int): The number of top-K.

        Returns:
        -------
            Any: Symbolic.

        """
        if not symbolic_helper._is_value(top_k):  # noqa: SLF001
            top_k = g.op("Constant", value_t=torch.tensor(top_k, torch.long))
        return g.op("awml_pred::KnnBatch", xyz, query_xyz, batch_idxs, query_batch_offsets, top_k)

    @staticmethod
    def forward(
        _ctx: FunctionCtx,
        xyz: Tensor,
        query_xyz: Tensor,
        batch_idxs: Tensor,
        query_batch_offsets: Tensor,
        top_k: int,
    ) -> Tensor:
        """Run forward operation.

        Args:
        ----
            _ctx (FunctionCtx): `FunctionCtx` instance.
            xyz (Tensor): A tensor.
            query_xyz (Tensor): A tensor.
            batch_idxs (Tensor): A tensor.
            query_batch_offsets (Tensor): A tensor.
            top_k (int): The number of top-K.

        Returns:
        -------
            Tensor: Forward result.

        """
        from awml_pred.ops import knn_batch

        return knn_batch(xyz, query_xyz, batch_idxs, query_batch_offsets, top_k)


class TRTKnnBatch(Function):
    @staticmethod
    def symbolic(
        g: GraphCtx,
        xyz: Tensor,
        query_xyz: Tensor,
        batch_idxs: Tensor,
        query_batch_offsets: Tensor,
        top_k: int,
    ) -> Any:
        """Load the symbolic of this module.

        Args:
        ----
            g (GraphCtx): `GraphCtx` instance.
            xyz (Tensor): A tensor.
            query_xyz (Tensor): A tensor.
            batch_idxs (Tensor): A tensor.
            query_batch_offsets (Tensor): A tensor.
            top_k (int): The number of top-K.

        Returns:
        -------
            Any: Symbolic.

        """
        return g.op(
            "awml_pred::TRTKnnBatch",
            xyz,
            query_xyz,
            batch_idxs,
            query_batch_offsets,
            top_k_i=top_k,
            outputs=1,
        )

    @staticmethod
    def forward(
        _ctx: FunctionCtx,
        xyz: Tensor,
        _query_xyz: Tensor,
        _batch_idxs: Tensor,
        _query_batch_offsets: Tensor,
        top_k: int,
    ) -> Tensor:
        """Run forward operation.

        Args:
        ----
            _ctx (FunctionCtx): `FunctionCtx` instance.
            xyz (Tensor): A tensor.
            _query_xyz (Tensor): A tensor.
            _batch_idxs (Tensor): A tensor.
            _query_batch_offsets (Tensor): A tensor.
            top_k (int): The number of top-K.

        Returns:
        -------
            Tensor: Forward result.

        """
        n = xyz.size(0)
        return torch.zeros(n, top_k, dtype=torch.int)


class KnnBatchMlogK(Function):
    @staticmethod
    def symbolic(
        g: GraphCtx,
        xyz: Tensor,
        query_xyz: Tensor,
        batch_idxs: Tensor,
        query_batch_offsets: Tensor,
        top_k: int,
    ) -> Any:
        """Load the symbolic of this module.

        Args:
        ----
            g (GraphCtx): `GraphCtx` instance.
            xyz (Tensor): A tensor.
            query_xyz (Tensor): A tensor.
            batch_idxs (Tensor): A tensor.
            query_batch_offsets (Tensor): A tensor.
            top_k (int): The number of top-K.

        Returns:
        -------
            Any: Symbolic.

        """
        if not symbolic_helper._is_value(top_k):  # noqa: SLF001
            top_k = g.op("Constant", value_t=torch.tensor(top_k, torch.long))
        return g.op("awml_pred::KnnBatchMlogK", xyz, query_xyz, batch_idxs, query_batch_offsets, top_k)

    @staticmethod
    def forward(
        _ctx: FunctionCtx,
        xyz: Tensor,
        query_xyz: Tensor,
        batch_idxs: Tensor,
        query_batch_offsets: Tensor,
        top_k: int,
    ) -> Tensor:
        """Run forward operation.

        Args:
        ----
            _ctx (FunctionCtx): `FunctionCtx` instance.
            xyz (Tensor): A tensor.
            query_xyz (Tensor): A tensor.
            batch_idxs (Tensor): A tensor.
            query_batch_offsets (Tensor): A tensor.
            top_k (int): The number of top-K.

        Returns:
        -------
            Tensor: Forward result.

        """
        from awml_pred.ops import knn_batch_mlogk

        return knn_batch_mlogk(xyz, query_xyz, batch_idxs, query_batch_offsets, top_k)


class TRTKnnBatchMlogK(Function):
    @staticmethod
    def symbolic(
        g: GraphCtx,
        xyz: Tensor,
        query_xyz: Tensor,
        batch_idxs: Tensor,
        query_batch_offsets: Tensor,
        top_k: int,
    ) -> Any:
        """Load the symbolic of this module.

        Args:
        ----
            g (GraphCtx): `GraphCtx` instance.
            xyz (Tensor): A tensor.
            query_xyz (Tensor): A tensor.
            batch_idxs (Tensor): A tensor.
            query_batch_offsets (Tensor): A tensor.
            top_k (int): A tensor.

        Returns:
        -------
            Any: Symbolic.

        """
        return g.op(
            "awml_pred::TRTKnnBatchMlogK",
            xyz,
            query_xyz,
            batch_idxs,
            query_batch_offsets,
            top_k_i=top_k,
            outputs=1,
        )

    @staticmethod
    def forward(
        _ctx: FunctionCtx,
        xyz: Tensor,
        _query_xyz: Tensor,
        _batch_idxs: Tensor,
        _query_batch_offsets: Tensor,
        top_k: int,
    ) -> Tensor:
        """Run forward operation.

        Args:
        ----
            _ctx (FunctionCtx): `FunctionCtx` instance.
            xyz (Tensor): A tensor.
            _query_xyz (Tensor): A tensor.
            _batch_idxs (Tensor): A tensor.
            _query_batch_offsets (Tensor): A tensor.
            top_k (int): The number of top-K.

        Returns:
        -------
            Tensor: Forward result.

        """
        n = xyz.size(0)
        return torch.zeros(n, top_k, dtype=torch.int)


@FUNCTION_REWRITER.register(func_name="mtr.ops.knn_batch", backend="tensorrt")
def knn_batch__tensorrt(
    xyz: Tensor,
    query_xyz: Tensor,
    batch_idxs: Tensor,
    query_batch_offsets: Tensor,
    top_k: int,
) -> Tensor:
    """Run KNN batch computation with TRT backend.

    Args:
    ----
        xyz (Tensor): A tensor.
        query_xyz (Tensor): A tensor.
        batch_idxs (Tensor): A tensor.
        query_batch_offsets (Tensor): A tensor.
        top_k (int): A tensor.

    Returns:
    -------
        Tensor: Forward result.

    """
    return TRTKnnBatch.apply(xyz, query_xyz, batch_idxs, query_batch_offsets, top_k)


@FUNCTION_REWRITER.register(func_name="mtr.ops.knn_batch_mlogk", backend="tensorrt")
def knn_batch_mlogk__tensorrt(
    xyz: Tensor,
    query_xyz: Tensor,
    batch_idxs: Tensor,
    query_batch_offsets: Tensor,
    top_k: int,
) -> Tensor:
    """Run KNN batch MLogK computation with TRT backend.

    Args:
    ----
        xyz (Tensor): A tensor.
        query_xyz (Tensor): A tensor.
        batch_idxs (Tensor): A tensor.
        query_batch_offsets (Tensor): A tensor.
        top_k (int): The number of top-K.

    Returns:
    -------
        Tensor: Forward result.

    """
    return TRTKnnBatchMlogK.apply(xyz, query_xyz, batch_idxs, query_batch_offsets, top_k)
