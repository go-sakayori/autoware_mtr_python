from pathlib import Path
from typing import Sequence

import pytest
import torch

from awml_pred.models import TransformerDecoderLayer
from awml_pred.test_utils import require_cuda

NUM_CENTER_OBJECTS: int = 10
NUM_OBJECTS: int = 15
NUM_POLYLINES: int = 768

AGENT_DECODER_CONFIG = dict(
    d_model=512,
    num_head=8,
    dim_feedforward=512 * 4,
    dropout=0.1,
    activation="relu",
    normalize_before=False,
    keep_query_pos=False,
    rm_self_attn_decoder=False,
    use_local_attn=False,
)

MAP_DECODER_CONFIG = dict(
    d_model=256,
    num_head=8,
    dim_feedforward=256 * 4,
    dropout=0.1,
    activation="relu",
    normalize_before=False,
    keep_query_pos=False,
    rm_self_attn_decoder=False,
    use_local_attn=True,
)

# ======================AGENT======================
# tgt: (torch.Size([64, 10, 512]), torch.float32, device(type='cuda', index=0))
# memory: (torch.Size([15, 10, 512]), torch.float32, device(type='cuda', index=0))
# tgt_mask: None
# memory_mask: None
# pos: (torch.Size([15, 10, 512]), torch.float32, device(type='cuda', index=0))
# query_pos: (torch.Size([64, 10, 512]), torch.float32, device(type='cuda', index=0))
# query_sine_embed: (torch.Size([64, 10, 512]), torch.float32, device(type='cuda', index=0))
# is_first: False
# memory_key_padding_mask: (torch.Size([10, 15]), torch.bool, device(type='cuda', index=0))
# key_batch_cnt: None
# index_pair: None
# index_pair_batch: None
# memory_valid_mask: None

# =======================MAP======================
# tgt: (torch.Size([64, 10, 256]), torch.float32, device(type='cuda', index=0))
# memory: (torch.Size([7680, 256]), torch.float32, device(type='cuda', index=0))
# tgt_mask: None
# memory_mask: None
# pos: (torch.Size([7680, 256]), torch.float32, device(type='cuda', index=0))
# query_pos: (torch.Size([64, 10, 256]), torch.float32, device(type='cuda', index=0))
# query_sine_embed: (torch.Size([64, 10, 256]), torch.float32, device(type='cuda', index=0))
# is_first: False
# memory_key_padding_mask: None
# key_batch_cnt: (torch.Size([10]), torch.int32, device(type='cuda', index=0))
# index_pair: (torch.Size([640, 384]), torch.int32, device(type='cuda', index=0))
# index_pair_batch: (torch.Size([640]), torch.int32, device(type='cuda', index=0))
# memory_valid_mask: (torch.Size([7680]), torch.bool, device(type='cuda', index=0))


def get_agent_dummy_input(is_first: bool, device="cuda") -> Sequence[torch.Tensor]:
    d_model = AGENT_DECODER_CONFIG["d_model"]

    tgt = torch.rand(64, NUM_CENTER_OBJECTS, d_model, device=device)
    memory = torch.rand(NUM_OBJECTS, NUM_CENTER_OBJECTS, d_model, device=device)
    tgt_mask = None
    memory_mask = None
    pos = torch.rand(NUM_OBJECTS, NUM_CENTER_OBJECTS, d_model, device=device)
    query_pos = torch.rand(64, NUM_CENTER_OBJECTS, d_model, device=device)
    query_sine_embed = torch.rand(64, NUM_CENTER_OBJECTS, d_model, device=device)
    memory_key_padding_mask = torch.randint(0, 2, (NUM_CENTER_OBJECTS, NUM_OBJECTS), dtype=torch.bool, device=device)
    key_batch_cnt = None
    index_pair = None
    index_pair_batch = None
    memory_valid_mask = None

    return (
        tgt,
        memory,
        tgt_mask,
        memory_mask,
        pos,
        query_pos,
        query_sine_embed,
        is_first,
        memory_key_padding_mask,
        key_batch_cnt,
        index_pair,
        index_pair_batch,
        memory_valid_mask,
    )


def get_map_dummy_input(is_first: bool, device="cuda") -> Sequence[torch.Tensor]:
    d_model: int = MAP_DECODER_CONFIG["d_model"]

    tgt = torch.ones(64, NUM_CENTER_OBJECTS, d_model, device=device)
    memory = torch.ones(NUM_POLYLINES * NUM_CENTER_OBJECTS, d_model, device=device)
    tgt_mask = None
    memory_mask = None
    pos = torch.ones(NUM_POLYLINES * NUM_CENTER_OBJECTS, d_model, device=device)
    query_pos = torch.ones(64, NUM_CENTER_OBJECTS, d_model, device=device)
    query_sine_embed = torch.ones(64, NUM_CENTER_OBJECTS, d_model, device=device)
    memory_key_padding_mask = None
    key_batch_cnt = torch.ones(NUM_CENTER_OBJECTS, dtype=torch.int, device=device)
    index_pair = torch.ones(64 * NUM_CENTER_OBJECTS, NUM_POLYLINES // 2, dtype=torch.int, device=device)
    index_pair_batch = torch.ones(64 * NUM_CENTER_OBJECTS, dtype=torch.int, device=device)
    memory_valid_mask = torch.randint(0, 2, (NUM_POLYLINES * NUM_CENTER_OBJECTS,), dtype=torch.bool, device=device)

    return (
        tgt,
        memory,
        tgt_mask,
        memory_mask,
        pos,
        query_pos,
        query_sine_embed,
        is_first,
        memory_key_padding_mask,
        key_batch_cnt,
        index_pair,
        index_pair_batch,
        memory_valid_mask,
    )


@require_cuda
@pytest.mark.parametrize(("mode", "is_first"), [("agent", True), ("agent", False), ("map", True), ("map", False)])
def test_transformer_decoder_layer(mode: str, is_first: bool) -> None:
    if mode == "agent":
        decoder_layer = TransformerDecoderLayer(**AGENT_DECODER_CONFIG)
        dummy_input = get_agent_dummy_input(is_first)
        output_shape = (64, NUM_CENTER_OBJECTS, AGENT_DECODER_CONFIG["d_model"])
    elif mode == "map":
        decoder_layer = TransformerDecoderLayer(**MAP_DECODER_CONFIG)
        dummy_input = get_map_dummy_input(is_first)
        output_shape = (64 * NUM_CENTER_OBJECTS, MAP_DECODER_CONFIG["d_model"])
    else:
        raise ValueError(f"mode: {mode}, is_first: {is_first}")

    decoder_layer = decoder_layer.eval().cuda()

    output = decoder_layer(*dummy_input)
    assert output.shape == output_shape


@require_cuda
@pytest.mark.parametrize(
    ("mode", "is_first", "filename"),
    [
        ("agent", True, "test_transformer_agent_decoder_layer_is_first.onnx"),
        ("agent", False, "test_transformer_agent_decoder_layer_non_first.onnx"),
        ("map", True, "test_transformer_map_decoder_layer_is_first.onnx"),
        ("map", False, "test_transformer_map_decoder_layer_non_first.onnx"),
    ],
)
def test_transformer_decoder_layer_export_onnx(
    tmp_path: Path,
    save_onnx: bool,
    mode: str,
    is_first: bool,
    filename: str,
) -> None:
    if mode == "agent":
        decoder_layer = TransformerDecoderLayer(**AGENT_DECODER_CONFIG)
        dummy_input = get_agent_dummy_input(is_first, "cpu")
    elif mode == "map":
        decoder_layer = TransformerDecoderLayer(**MAP_DECODER_CONFIG)
        dummy_input = get_map_dummy_input(is_first, "cpu")
    else:
        raise ValueError(f"mode: {mode}, is_first: {is_first}")

    decoder_layer = decoder_layer.eval()

    if not save_onnx:
        filename = tmp_path / filename

    with torch.no_grad():
        torch.onnx.export(
            decoder_layer,
            dummy_input,
            filename,
            opset_version=17,
        )
