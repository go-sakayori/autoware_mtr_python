# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Modified by Shaoshuai Shi
# All Rights Reserved


"""Reference:
https://github.com/dvlab-research/DeepVision3D/blob/master/EQNet/eqnet/transformer/multi_head_attention.py.
"""

from torch import nn

from awml_pred.common import LAYERS
from awml_pred.models.utils import get_activation_fn
from awml_pred.typing import Tensor

from .multi_head_attention import MultiheadAttention
from .multi_head_attention_local import MultiheadAttentionLocal

__all__ = ("TransformerEncoderLayer",)


@LAYERS.register()
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_head: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
        use_local_attn: bool = False,
    ) -> None:
        """Encoder layer for Transformer.

        Args:
        ----
            d_model (int): Number of model dimensions.
            num_head (int): Number of heads.
            dim_feedforward (int, optional): Feedforward dimension. Defaults to 2048.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            activation (str, optional): Name of activation function. Defaults to "relu".
            normalize_before (bool, optional): Whether to normalize first. Defaults to False.
            use_local_attn (bool, optional): Whether to use local attention. Defaults to False.

        """
        super().__init__()
        # NOTE: If False, exception occurs
        self.use_local_attn = True  # self.use_local_attn = use_local_attn

        if self.use_local_attn:
            self.self_attn = MultiheadAttentionLocal(d_model, num_head, dropout=dropout)
        else:
            self.self_attn = MultiheadAttention(d_model, num_head, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src: Tensor,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        index_pair: Tensor | None = None,
        query_batch_cnt: Tensor | None = None,
        key_batch_cnt: Tensor | None = None,
        index_pair_batch: Tensor | None = None,
    ) -> Tensor:
        q = k = self.with_pos_embed(src, pos)
        if self.use_local_attn:
            src2, _ = self.self_attn(
                q,
                k,
                value=src,
                index_pair=index_pair,
                query_batch_cnt=query_batch_cnt,
                key_batch_cnt=key_batch_cnt,
                index_pair_batch=index_pair_batch,
            )
        else:
            src2, _ = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src: Tensor,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        index_pair: Tensor | None = None,
        query_batch_cnt: Tensor | None = None,
        key_batch_cnt: Tensor | None = None,
        index_pair_batch: Tensor | None = None,
    ) -> Tensor:
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        if self.use_local_attn:
            src2, _ = self.self_attn(
                q,
                k,
                value=src,
                index_pair=index_pair,
                query_batch_cnt=query_batch_cnt,
                key_batch_cnt=key_batch_cnt,
                index_pair_batch=index_pair_batch,
            )
        else:
            src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        index_pair: Tensor | None = None,
        query_batch_cnt: Tensor | None = None,
        key_batch_cnt: Tensor | None = None,
        index_pair_batch: Tensor | None = None,
    ) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(
                src,
                src_key_padding_mask,
                pos,
                index_pair=index_pair,
                query_batch_cnt=query_batch_cnt,
                key_batch_cnt=key_batch_cnt,
                index_pair_batch=index_pair_batch,
            )
        else:
            return self.forward_post(
                src,
                src_key_padding_mask,
                pos,
                index_pair=index_pair,
                query_batch_cnt=query_batch_cnt,
                key_batch_cnt=key_batch_cnt,
                index_pair_batch=index_pair_batch,
            )
