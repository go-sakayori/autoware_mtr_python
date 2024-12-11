from pathlib import Path
from typing import Sequence

import pytest
import torch

from awml_pred.models.layers.multi_head_attention import MultiheadAttention
from awml_pred.test_utils import require_cuda

NUM_CENTER_OBJECTS = 10
NUM_OBJECTS = 15

D_MODEL = 512
CONFIG = dict(embed_dim=D_MODEL * 2, num_heads=8, dropout=0.1)


# =====================MultiHeadAttention=====================
# query: torch.Size([64, 10, 1024])
# key: torch.Size([15, 10, 1024]), torch.float32
# value: torch.Size([15, 10, 512]), torch.float32
# key_padding_mask: (torch.Size([10, 15]), torch.bool)
# need_weights: True
# attn_mask: None
# =================================================================
# torch.Size([64, 10, 512]) torch.Size([10, 64, 15])
# .=====================MultiHeadAttention=====================
# query: torch.Size([64, 10, 512])
# key: torch.Size([64, 10, 512]), torch.float32
# value: torch.Size([64, 10, 512]), torch.float32
# key_padding_mask: None
# need_weights: True
# attn_mask: None
# =================================================================
# torch.Size([64, 10, 512]) torch.Size([10, 64, 64])
def get_dummy_input(use_key_padding_mask: bool, device="cuda") -> Sequence[torch.Tensor]:
    embed_dim: int = CONFIG["embed_dim"]
    if use_key_padding_mask:
        key = torch.ones(NUM_OBJECTS, NUM_CENTER_OBJECTS, embed_dim, device=device)
        value = torch.ones(NUM_OBJECTS, NUM_CENTER_OBJECTS, embed_dim, device=device)
        key_padding_mask = torch.ones(NUM_CENTER_OBJECTS, NUM_OBJECTS, dtype=torch.bool, device=device)
    else:
        key = key = torch.ones(64, NUM_CENTER_OBJECTS, embed_dim, device=device)
        value = torch.ones(64, NUM_CENTER_OBJECTS, embed_dim, device=device)
        key_padding_mask = None

    query = torch.ones(64, NUM_CENTER_OBJECTS, embed_dim, device=device)
    need_weights = True
    attn_mask = None

    return (query, key, value, key_padding_mask, need_weights, attn_mask)


@require_cuda
@pytest.mark.parametrize("use_key_padding_mask", [True, False])
def test_multi_head_attention(use_key_padding_mask: bool) -> None:
    multi_head_attn = MultiheadAttention(**CONFIG)
    multi_head_attn = multi_head_attn.eval().cuda()

    dummy_input = get_dummy_input(use_key_padding_mask)

    outputs, weights = multi_head_attn(*dummy_input)

    assert outputs.shape == (64, NUM_CENTER_OBJECTS, CONFIG["embed_dim"])
    if use_key_padding_mask:
        assert weights.shape == (NUM_CENTER_OBJECTS, 64, NUM_OBJECTS)
    else:
        assert weights.shape == (NUM_CENTER_OBJECTS, 64, 64)


@require_cuda
@pytest.mark.parametrize(
    ("use_key_padding_mask", "filename"),
    [
        (True, "test_multi_head_attn_with_key_pad_mask.onnx"),
        (False, "test_multi_head_attn_without_key_pad_mask.onnx"),
    ],
)
def test_multi_head_attention_export_onnx(
    tmp_path: Path,
    save_onnx: bool,
    use_key_padding_mask: bool,
    filename: str,
) -> None:
    multi_head_attn = MultiheadAttention(**CONFIG).eval()

    dummy_input = get_dummy_input(use_key_padding_mask, "cpu")

    input_names = ["query", "key", "value", "key_padding_mask", "need_weights", "attn_mask"]
    output_names = ["outputs", "weights"]
    dynamic_axes = {
        "query": {1: "num_center_objects"},
        "key": {0: "num_objects", 1: "num_center_objects"},
        "value": {0: "num_objects", 1: "num_center_objects"},
        "outputs": {1: "num_center_objects"},
    }

    if use_key_padding_mask:
        dynamic_axes.update(
            {
                "key_padding_mask": {0: "num_center_objects", 1: "num_objects"},
                "weights": {0: "num_center_objects", 2: "num_objects"},
            },
        )
    else:
        dynamic_axes.update({"weights": {0: "num_center_objects"}})

    if not save_onnx:
        filename = tmp_path / filename

    with torch.no_grad():
        torch.onnx.export(
            multi_head_attn,
            dummy_input,
            filename,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
        )
