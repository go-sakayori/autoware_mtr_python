from pathlib import Path
from typing import Sequence

import pytest
import torch

from awml_pred.models import MTRDecoder
from awml_pred.test_utils import require_cuda

NUM_CENTER_OBJECTS: int = 10
NUM_OBJECTS: int = 15
NUM_FEATURE: int = 256
NUM_POLYLINES: int = 768
NUM_FUTURE_FRAMES: int = 80
NUM_MOTION_MODES: int = 6

CONFIG = dict(
    in_channels=NUM_FEATURE,
    num_future_frames=NUM_FUTURE_FRAMES,
    num_motion_modes=NUM_MOTION_MODES,
    d_model=512,
    num_decoder_layers=6,
    num_attn_head=8,
    map_center_offset=(30, 0),
    num_waypoint_map_polylines=128,
    num_base_map_polylines=256,
    dropout=0.1,
    map_d_model=256,
    nms_threshold=2.5,
    decode_loss=dict(
        name="MTRLoss",
        reg_cfg=dict(name="GMMLoss", weight=1.0, use_square_gmm=False),
        cls_cfg=dict(name="CrossEntropyLoss", weight=1.0),
        vel_cfg=dict(name="L1Loss", weight=0.2),
    ),
)


def get_dummy_input(device="cuda") -> Sequence[torch.Tensor]:
    obj_feature = torch.rand(NUM_CENTER_OBJECTS, NUM_OBJECTS, NUM_FEATURE, device=device)
    obj_mask = torch.randint(0, 2, (NUM_CENTER_OBJECTS, NUM_OBJECTS), dtype=bool, device=device)
    obj_pos = torch.rand(NUM_CENTER_OBJECTS, NUM_OBJECTS, 3, device=device)
    map_feature = torch.rand(NUM_CENTER_OBJECTS, NUM_POLYLINES, NUM_FEATURE, device=device)
    map_mask = torch.randint(0, 2, (NUM_CENTER_OBJECTS, NUM_POLYLINES), dtype=bool, device=device)
    map_pos = torch.rand(NUM_CENTER_OBJECTS, NUM_POLYLINES, 3, device=device)
    center_objects_feature = torch.rand(NUM_CENTER_OBJECTS, NUM_FEATURE, device=device)
    intention_points = torch.rand(NUM_CENTER_OBJECTS, 64, 2, dtype=torch.float32, device=device)

    return (obj_feature, obj_mask, obj_pos, map_feature, map_mask, map_pos, center_objects_feature, intention_points)


@require_cuda
def test_mtr_decoder() -> None:
    mtr_decoder = MTRDecoder(**CONFIG)

    mtr_decoder = mtr_decoder.eval().cuda()

    dummy_input = get_dummy_input()

    scores, trajs = mtr_decoder(*dummy_input)

    assert scores.shape == (NUM_CENTER_OBJECTS, NUM_MOTION_MODES)
    assert trajs.shape == (NUM_CENTER_OBJECTS, NUM_MOTION_MODES, NUM_FUTURE_FRAMES, 7)


@require_cuda
@pytest.mark.parametrize("dynamic", (True, False))
def test_mtr_decoder_export_onnx(tmp_path: Path, save_onnx: bool, dynamic: bool) -> None:
    mtr_decoder = MTRDecoder(**CONFIG)

    dummy_input = get_dummy_input("cpu")
    input_names = [
        "obj_feature",
        "obj_mask",
        "obj_pos",
        "map_feature",
        "map_mask",
        "map_pos",
        "center_objects_feature",
        "intention_points",
    ]
    output_names = ["pred_scores", "pred_trajs"]

    if dynamic:
        dynamic_axes = {
            "obj_feature": {0: "num_center_objects", 1: "num_objects"},
            "obj_mask": {0: "num_center_objects", 1: "num_objects"},
            "obj_pos": {0: "num_center_objects", 1: "num_objects"},
            "map_feature": {0: "num_center_objects"},
            "map_mask": {0: "num_center_objects"},
            "map_pos": {0: "num_center_objects"},
            "center_objects_feature": {0: "num_center_objects"},
            "intention_points": {0: "num_center_objects"},
            "pred_scores": {0: "num_center_objects"},
            "pred_trajs": {0: "num_center_objects"},
        }
        filename: str = "test_mtr_decoder_dynamic.onnx"
    else:
        dynamic_axes = None
        filename: str = "test_mtr_decoder_static.onnx"

    if not save_onnx:
        filename = tmp_path / filename

    with torch.no_grad():
        torch.onnx.export(
            mtr_decoder,
            dummy_input,
            filename,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
        )
