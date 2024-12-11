from pathlib import Path
from typing import Sequence

import pytest
import torch

from awml_pred.models.layers.multi_head_attention_local import \
    MultiheadAttentionLocal
from awml_pred.test_utils import require_cuda

D_MODEL = 256
NUM_CENTER_OBJECTS = 10
NUM_POLYLINES = 768

CONFIG = dict(
    embed_dim=2 * D_MODEL,
    num_heads=8,
    dropout=0.0,
    without_weight=True,
    vdim=D_MODEL,
)


# query: torch.Size([640, 512])
# key: torch.Size([7680, 512]), torch.float32
# value: torch.Size([7680, 256]), torch.float32
# index_pair: torch.Size([640, 384]), torch.int32
# query_batch_cnt: torch.Size([10]), torch.int32
# key_batch_cnt: torch.Size([10]), torch.int32
# index_pair_batch: torch.Size([640]), torch.int32
# =================================================================
# torch.Size([640, 256]) torch.Size([640, 384])
# .=================================================================
def get_dummy_input(device="cuda") -> Sequence[torch.Tensor]:
    embed_dim = CONFIG["embed_dim"]

    query = torch.ones(64 * NUM_CENTER_OBJECTS, embed_dim, device=device)
    key = torch.ones(NUM_POLYLINES * NUM_CENTER_OBJECTS, embed_dim, device=device)
    value = torch.ones(NUM_POLYLINES * NUM_CENTER_OBJECTS, D_MODEL, device=device)
    index_pair = torch.ones(64 * NUM_CENTER_OBJECTS, NUM_POLYLINES // 2, dtype=torch.int, device=device)
    query_batch_cnt = torch.ones(NUM_CENTER_OBJECTS, dtype=torch.int32, device=device)
    key_batch_cnt = torch.ones(NUM_CENTER_OBJECTS, dtype=torch.int32, device=device)
    index_pair_batch = torch.ones(64 * NUM_CENTER_OBJECTS, dtype=torch.int32, device=device)

    return (
        query,
        key,
        value,
        index_pair,
        query_batch_cnt,
        key_batch_cnt,
        index_pair_batch,
    )


@require_cuda
def test_multi_head_attention_local() -> None:
    multi_head_attn_local = MultiheadAttentionLocal(**CONFIG)
    multi_head_attn_local = multi_head_attn_local.eval().cuda()

    dummy_input = get_dummy_input()

    outputs, weights = multi_head_attn_local(*dummy_input)

    assert outputs.shape == (64 * NUM_CENTER_OBJECTS, D_MODEL)
    assert weights.shape == (64 * NUM_CENTER_OBJECTS, NUM_POLYLINES // 2)


@require_cuda
@pytest.mark.parametrize("dynamic", (True, False))
def test_multi_head_attention_local_export_onnx(tmp_path: Path, save_onnx: bool, dynamic: bool) -> None:
    multi_head_attn_local = MultiheadAttentionLocal(**CONFIG).eval()

    dummy_input = get_dummy_input("cpu")

    input_names = ["query", "key", "value", "index_pair", "query_batch_cnt", "key_batch_cnt", "index_pair_batch"]
    output_names = ["outputs", "weights"]

    if dynamic:
        dynamic_axes = {
            "query": {0: "num_query"},
            "key": {0: "num_key"},
            "value": {0: "num_value"},
            "index_pair": {0: "num_query", 1: "num_key_half"},
            "query_batch_cnt": {0: "num_center_objects"},
            "key_batch_cnt": {0: "num_center_objects"},
            "index_pair_batch": {0: "num_query"},
        }
        filename = "test_multi_head_attn_local_dynamic.onnx"
    else:
        dynamic_axes = None
        filename = "test_multi_head_attn_local_static.onnx"

    if not save_onnx:
        filename = tmp_path / filename

    with torch.no_grad():
        torch.onnx.export(
            multi_head_attn_local,
            dummy_input,
            filename,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
        )
