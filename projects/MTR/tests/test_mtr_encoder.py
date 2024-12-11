from pathlib import Path
from typing import Sequence

import pytest
import torch

from awml_pred.models import MTREncoder
from awml_pred.test_utils import require_cuda

NUM_CENTER_OBJECTS: int = 10
NUM_OBJECTS: int = 15
NUM_PAST_TIMESTEP: int = 11
NUM_AGENT_DIMS: int = 29
NUM_POLYLINES: int = 768
NUM_WAYPOINTS: int = 20
NUM_POLYLINE_DIMS: int = 9
NUM_FEATURE: int = 256

CONFIG = dict(
    agent_polyline_encoder=dict(
        name="PointNetPolylineEncoder",
        in_channels=NUM_AGENT_DIMS + 1,
        hidden_dim=256,
        num_layers=3,
        out_channels=NUM_FEATURE,
    ),
    map_polyline_encoder=dict(
        name="PointNetPolylineEncoder",
        in_channels=9,
        hidden_dim=64,
        num_layers=5,
        num_pre_layers=3,
        out_channels=NUM_FEATURE,
    ),
    attention_layer=dict(
        name="TransformerEncoderLayer",
        d_model=256,
        num_head=8,
        dim_feedforward=1024,
        dropout=0.1,
        normalize_before=False,
        num_layers=6,
    ),
    use_local_attn=True,
    num_attn_neighbors=16,
)


def get_dummy_input(device="cuda") -> Sequence[torch.Tensor]:
    obj_trajs = torch.ones(NUM_CENTER_OBJECTS, NUM_OBJECTS, NUM_PAST_TIMESTEP, NUM_AGENT_DIMS, device=device)
    obj_trajs_mask = torch.randint(
        0, 2, (NUM_CENTER_OBJECTS, NUM_OBJECTS, NUM_PAST_TIMESTEP), dtype=torch.bool, device=device,
    )
    map_polylines = torch.ones(NUM_CENTER_OBJECTS, NUM_POLYLINES, NUM_WAYPOINTS, NUM_POLYLINE_DIMS, device=device)
    map_polylines_mask = torch.ones(NUM_CENTER_OBJECTS, NUM_POLYLINES, NUM_WAYPOINTS, dtype=torch.bool, device=device)
    map_polylines_center = torch.ones(NUM_CENTER_OBJECTS, NUM_POLYLINES, 3, device=device)
    obj_trajs_pos = torch.ones(NUM_CENTER_OBJECTS, NUM_OBJECTS, 3, device=device)
    track_index_to_predict = torch.randint(0, 3, (NUM_CENTER_OBJECTS,), device=device)

    return (
        obj_trajs,
        obj_trajs_mask,
        map_polylines,
        map_polylines_mask,
        map_polylines_center,
        obj_trajs_pos,
        track_index_to_predict,
    )


@require_cuda
def test_mtr_encoder() -> None:
    mtr_encoder = MTREncoder(**CONFIG)
    mtr_encoder = mtr_encoder.eval().cuda()
    dummy_input = get_dummy_input()

    (
        obj_polylines_feature,
        obj_valid_mask,
        obj_trajs_last_pos,
        map_polylines_feature,
        map_valid_mask,
        map_polylines_center,
        center_objects_feature,
    ) = mtr_encoder(*dummy_input)

    assert obj_polylines_feature.shape == (NUM_CENTER_OBJECTS, NUM_OBJECTS, NUM_FEATURE)
    assert obj_valid_mask.shape == (NUM_CENTER_OBJECTS, NUM_OBJECTS)
    assert obj_trajs_last_pos.shape == (NUM_CENTER_OBJECTS, NUM_OBJECTS, 3)
    assert map_polylines_feature.shape == (NUM_CENTER_OBJECTS, NUM_POLYLINES, NUM_FEATURE)
    assert map_valid_mask.shape == (NUM_CENTER_OBJECTS, NUM_POLYLINES)
    assert map_polylines_center.shape == (NUM_CENTER_OBJECTS, NUM_POLYLINES, 3)
    assert center_objects_feature.shape == (NUM_CENTER_OBJECTS, NUM_FEATURE)


@require_cuda
@pytest.mark.parametrize("dynamic", [True, False])
def test_mtr_encoder_export_onnx(tmp_path: Path, save_onnx: bool, dynamic: bool) -> None:
    mtr_encoder = MTREncoder(**CONFIG).eval()

    dummy_input = get_dummy_input("cpu")
    input_names = [
        "obj_trajs",
        "obj_trajs_mask",
        "map_polylines",
        "map_polylines_mask",
        "map_polylines_center",
        "obj_trajs_pos",
        "track_index_to_predict",
    ]
    output_names = [
        "obj_polylines_feature",
        "obj_valid_mask",
        "obj_trajs_last_pos",
        "map_polylines_feature",
        "map_valid_mask",
        "map_polylines_center",
        "center_objects_feature",
    ]

    if dynamic:
        dynamic_axes = {
            "obj_trajs": {0: "num_center_objects", 1: "num_objects"},
            "obj_trajs_mask": {0: "num_center_objects", 1: "num_objects"},
            "map_polylines": {0: "num_center_objects"},
            "map_polylines_mask": {0: "num_center_objects"},
            "map_polylines_center": {0: "num_center_objects"},
            "obj_trajs_pos": {0: "num_center_objects", 1: "num_objects"},
            "track_index_to_predict": {0: "num_center_objects"},
            "obj_polylines_feature": {0: "num_center_objects", 1: "num_objects"},
            "obj_valid_mask": {0: "num_center_objects", 1: "num_objects"},
            "obj_trajs_last_pos": {0: "num_center_objects", 1: "num_objects"},
            "map_polylines_feature": {0: "num_center_objects"},
            "map_valid_mask": {0: "num_center_objects"},
            "center_objects_feature": {0: "num_center_objects"},
        }
        filename: str = "test_mtr_encoder_dynamic.onnx"
    else:
        dynamic_axes = None
        filename: str = "test_mtr_encoder_static.onnx"

    if not save_onnx:
        filename = tmp_path / filename

    with torch.no_grad():
        torch.onnx.export(
            mtr_encoder,
            dummy_input,
            filename,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
        )
