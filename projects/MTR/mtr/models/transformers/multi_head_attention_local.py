# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved


"""Mostly copy-paste from
https://github.com/dvlab-research/DeepVision3D/blob/master/EQNet/eqnet/transformer/multi_head_attention.py.
"""

import torch
from torch import nn
from torch.nn import Linear
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.parameter import Parameter

from awml_pred.common import LAYERS
from awml_pred.typing import Tensor

from ...ops import attention_value_computation, attention_weight_computation

__all__ = ("MultiheadAttentionLocal",)


@LAYERS.register()
class MultiheadAttentionLocal(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need.

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V).

    Args:
    ----
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::
        >>> multihead_attn = nn.MultiheadAttentionLocal(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(
                query, key, value, index_pair, query_batch_cnt, key_batch_cnt, index_pair_batch
            )

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        without_weight: bool = False,
        vdim: int | None = None,
    ) -> None:
        super(MultiheadAttentionLocal, self).__init__()
        self.embed_dim = embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))

        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = Linear(self.vdim, self.vdim, bias=True)

        self.without_weight = without_weight
        if self.without_weight:
            self.in_proj_weight = self.in_proj_bias = None
            constant_(self.out_proj.bias, 0.0)
        else:
            self._reset_parameters()

    def _reset_parameters(self) -> None:
        xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def _proj_qkv(self, t: Tensor, start: int, end: int) -> Tensor:
        _w = self.in_proj_weight[start:end, :]
        _b = self.in_proj_bias[start:end]
        t = F.linear(t, _w, _b)
        return t

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        index_pair: Tensor,
        query_batch_cnt: Tensor,
        key_batch_cnt: Tensor,
        index_pair_batch: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """To reduce memory cost in attention computation, use index to indicate attention pair.

        Args:
        ----
            query (Tensor): Query embeddings in shape (N, C) where N is the total query tokens length,
                C is the embedding dimensions.
            key (Tensor): Key embeddings in shape (M, C) where M is the total key tokens length.
            value (Tensor): Value embeddings in shape (M, C).
            index_pair (Tensor): The associated key indices of each query for computing attention, in shape (N, L)
                where L is the maximum number of keys for attention computation.
            query_batch_cnt (Tensor): Indicates the query amount in each batch, in shape (B) where B is batch size.
            key_batch_cnt (Tensor): Indicates the key/value amount in each batch, in shape (B).
            index_pair_batch (Tensor): The batch index of each query, in shape (N).

        Returns:
        -------
            tuple[Tensor, Tensor]: Attention computation result in shape (N, C) and weight in shape (N, L, H).

        """
        total_query_len, embed_dim = query.size()
        _, vdim = value.size()
        v_head_dim = vdim // self.num_heads

        scaling: float = float(self.head_dim) ** -0.5

        # generate qkv features.
        if not self.without_weight:
            q = self._proj_qkv(query, 0, embed_dim)
            q = q * scaling
            k = self._proj_qkv(key, embed_dim, embed_dim * 2)
            v = self._proj_qkv(value, embed_dim * 2, embed_dim * 3)
        else:
            q = query * scaling
            k, v = key, value

        q = q.contiguous().view(total_query_len, self.num_heads, self.head_dim)
        k = k.contiguous().view(-1, self.num_heads, self.head_dim)
        v = v.contiguous().view(-1, self.num_heads, v_head_dim)

        # compute attention weight.
        attn_output_weights = attention_weight_computation(
            query_batch_cnt,
            key_batch_cnt,
            index_pair_batch,
            index_pair,
            q,
            k,
        )

        attn_mask = index_pair == -1
        # NOTE: float("-inf") make nan TRT output
        attn_output_weights = attn_output_weights.masked_fill_(attn_mask[..., None], -1000.0)
        attn_output_weights = F.softmax(attn_output_weights, dim=1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = attention_value_computation(
            query_batch_cnt,
            key_batch_cnt,
            index_pair_batch,
            index_pair,
            attn_output_weights,
            v,
        )

        attn_output = attn_output.view(total_query_len, vdim)

        if self.out_proj is not None:
            attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        return attn_output, attn_output_weights.sum(dim=-1) / self.num_heads
