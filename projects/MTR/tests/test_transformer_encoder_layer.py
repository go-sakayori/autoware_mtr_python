from pathlib import Path
from typing import Sequence

import torch

from awml_pred.models import TransformerEncoderLayer
from awml_pred.test_utils import require_cuda

NUM_POLYLINES: int = 768
NUM_CENTER_OBJECTS: int = 10

CONFIG = dict(
    d_model=256,
    num_head=8,
    dim_feedforward=1024,
    dropout=0.1,
    normalize_before=False,
    use_local_attn=False,
)

# src: (torch.Size([7830, 256]), torch.float32, device(type="cuda", index=0))
# src_key_padding_mask: None
# pos: (torch.Size([7830, 256]), torch.float32, device(type="cuda", index=0))
# index_pair: (torch.Size([7830, 16]), torch.int32, device(type="cuda", index=0))
# query_batch_cnt: (torch.Size([10]), torch.int32, device(type="cuda", index=0))
# key_batch_cnt: (torch.Size([10]), torch.int32, device(type="cuda", index=0))
# index_pair_batch: (torch.Size([7830]), torch.int32, device(type="cuda", index=0))


def get_dummy_input(device="cuda") -> Sequence[torch.Tensor]:
    d_model = CONFIG["d_model"]

    src = torch.ones(7830, d_model, device=device)
    src_key_padding_mask = None
    pos = torch.ones(7830, d_model, device=device)
    index_pair = torch.ones(7830, 16, dtype=torch.int32, device=device)
    query_batch_cnt = torch.arange(0, NUM_CENTER_OBJECTS, dtype=torch.int32, device=device)
    key_batch_cnt = torch.arange(0, NUM_CENTER_OBJECTS, dtype=torch.int32, device=device)
    index_pair_batch = torch.ones(7830, dtype=torch.int32, device=device)

    return src, src_key_padding_mask, pos, index_pair, query_batch_cnt, key_batch_cnt, index_pair_batch


@require_cuda
def test_transformer_encoder_layer() -> None:
    encoder_layer = TransformerEncoderLayer(**CONFIG)
    encoder_layer = encoder_layer.eval().cuda()

    dummy_input = get_dummy_input()
    output = encoder_layer(*dummy_input)

    assert output.shape == (7830, CONFIG["d_model"])


@require_cuda
def test_transformer_encoder_layer_export_onnx(tmp_path: Path, save_onnx: bool) -> None:
    encoder_layer = TransformerEncoderLayer(**CONFIG).eval()

    filename: str = "test_transformer_encoder_layer.onnx"
    if not save_onnx:
        filename = tmp_path / filename

    dummy_input = get_dummy_input("cpu")
    input_names = [
        "src",
        "src_key_padding_mask",
        "pos",
        "index_pair",
        "query_batch_cnt",
        "key_batch_cnt",
        "index_pair_batch",
    ]

    with torch.no_grad():
        torch.onnx.export(
            encoder_layer,
            dummy_input,
            filename,
            input_names=input_names,
            output_names=["output"],
            opset_version=17,
        )
