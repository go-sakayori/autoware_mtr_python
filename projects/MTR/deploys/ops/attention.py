from typing import Any

import torch
from torch.autograd.function import Function, FunctionCtx
from torch.onnx.symbolic_helper import parse_args

from awml_pred.deploy.rewriters import FUNCTION_REWRITER
from awml_pred.typing import GraphCtx, JitValue, Tensor
from torch.onnx import register_custom_op_symbolic


# References:
#   https://github.com/open-mmlab/mmdeploy/blob/main/mmdeploy/mmcv/ops/nms.py


class TRTAttentionWeightComputation(Function):
    @staticmethod
    @parse_args("v", "v", "v", "v", "v", "v")
    def symbolic(
        g: GraphCtx,
        query_batch_cnt: JitValue,
        key_batch_cnt: JitValue,
        index_pair_batch: JitValue,
        index_pair: JitValue,
        query_features: JitValue,
        key_features: JitValue,
    ) -> Any:
        """Load symbolic of this module.

        Args:
        ----
            g (GraphCtx): `GraphCtx` instance.
            query_batch_cnt (JitValue): A tensor.
            key_batch_cnt (JitValue): A tensor.
            index_pair_batch (JitValue): A tensor.
            index_pair (JitValue): A tensor.
            query_features (JitValue): A tensor.
            key_features (JitValue): A tensor.

        Returns:
        -------
            Any: Symbolic.

        """
        return g.op(
            "awml_pred::TRTAttentionWeightComputation",
            query_batch_cnt,
            key_batch_cnt,
            index_pair_batch,
            index_pair,
            query_features,
            key_features,
            outputs=1,
        )

    @staticmethod
    def forward(
        _ctx: FunctionCtx,
        _query_batch_cnt: Tensor,
        _key_batch_cnt: Tensor,
        _index_pair_batch: Tensor,
        index_pair: Tensor,
        _query_features: Tensor,
        key_features: Tensor,
    ) -> Tensor:
        """Run forward operation.

        Args:
        ----
            _ctx (FunctionCtx): `FunctionCtx` instance.
            _query_batch_cnt (Tensor): A tensor.
            _key_batch_cnt (Tensor): A tensor.
            _index_pair_batch (Tensor): A tensor.
            index_pair (Tensor): A tensor.
            _query_features (Tensor): A tensor.
            key_features (Tensor): A tensor.

        Returns:
        -------
            Tensor: Forward result.

        """
        total_query_num, local_size = index_pair.size()
        nhead = key_features.size(1)
        return torch.rand(total_query_num, local_size, nhead)


class TRTAttentionValueComputation(Function):
    @staticmethod
    @parse_args("v", "v", "v", "v", "v", "v")
    def symbolic(
        g: GraphCtx,
        query_batch_cnt: JitValue,
        key_batch_cnt: JitValue,
        index_pair_batch: JitValue,
        index_pair: JitValue,
        attn_weight: JitValue,
        value_features: JitValue,
    ) -> Any:
        """Load the symbolic of this module.

        Args:
        ----
            g (GraphCtx): `GraphCtx` instance.
            query_batch_cnt (JitValue): A tensor.
            key_batch_cnt (JitValue): A tensor.
            index_pair_batch (JitValue): A tensor.
            index_pair (JitValue): A tensor.
            attn_weight (JitValue): A tensor.
            value_features (JitValue): A tensor.

        Returns:
        -------
            Any: Symbolic.

        """
        return g.op(
            "awml_pred::TRTAttentionValueComputation",
            query_batch_cnt,
            key_batch_cnt,
            index_pair_batch,
            index_pair,
            attn_weight,
            value_features,
            outputs=1,
        )

    @staticmethod
    def forward(
        _ctx: FunctionCtx,
        _query_batch_cnt: Tensor,
        _key_batch_cnt: Tensor,
        _index_pair_batch: Tensor,
        index_pair: Tensor,
        _attn_weight: Tensor,
        value_features: Tensor,
    ) -> Tensor:
        """Run forward operation.

        Args:
        ----
            _ctx (FunctionCtx): `FunctionCtx` instance.
            _query_batch_cnt (Tensor): A tensor.
            _key_batch_cnt (Tensor): A tensor.
            _index_pair_batch (Tensor): A tensor.
            index_pair (Tensor): A tensor.
            _attn_weight (Tensor): A tensor.
            value_features (Tensor): A tensor.

        Returns:
        -------
            Tensor: Forward result.

        """
        total_query_num = index_pair.size(0)
        _, nhead, hdim = value_features.size()
        return torch.zeros(total_query_num, nhead, hdim)


@FUNCTION_REWRITER.register(func_name="mtr.ops.attention_weight_computation", backend="tensorrt")
def attention_weight_computation__tensorrt(
    query_batch_cnt: Tensor,
    key_batch_cnt: Tensor,
    index_pair_batch: Tensor,
    index_pair: Tensor,
    query_features: Tensor,
    key_features: Tensor,
) -> Tensor:
    """Run attention weight computation with TRT backend.

    Args:
    ----
        query_batch_cnt (Tensor): A tensor.
        key_batch_cnt (Tensor): A tensor.
        index_pair_batch (Tensor): A tensor.
        index_pair (Tensor): A tensor.
        query_features (Tensor): A tensor.
        key_features (Tensor): A tensor.

    Returns:
    -------
        Tensor: Forward result.

    """
    return TRTAttentionWeightComputation.apply(
        query_batch_cnt,
        key_batch_cnt,
        index_pair_batch,
        index_pair,
        query_features,
        key_features,
    )


@FUNCTION_REWRITER.register(func_name="mtr.ops.attention_value_computation", backend="tensorrt")
def attention_value_computation__tensorrt(
    query_batch_cnt: Tensor,
    key_batch_cnt: Tensor,
    index_pair_batch: Tensor,
    index_pair: Tensor,
    attn_weight: Tensor,
    value_features: Tensor,
) -> Tensor:
    """Run attention value computation with TRT backend.

    Args:
    ----
        query_batch_cnt (Tensor): A tensor.
        key_batch_cnt (Tensor): A tensor.
        index_pair_batch (Tensor): A tensor.
        index_pair (Tensor): A tensor.
        attn_weight (Tensor): A tensor.
        value_features (Tensor): A tensor.

    Returns:
    -------
        Tensor: Forward result.

    """
    return TRTAttentionValueComputation.apply(
        query_batch_cnt,
        key_batch_cnt,
        index_pair_batch,
        index_pair,
        attn_weight,
        value_features,
    )
