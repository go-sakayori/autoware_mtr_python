from pathlib import Path

import pytest
import torch

from awml_pred.models.layers import PointNetPolylineEncoder
from awml_pred.test_utils import require_cuda
from awml_pred.typing import Tensor


@require_cuda
@pytest.mark.parametrize(
    ("batch_size", "num_polylines", "num_points", "num_dim", "hidden_dim", "out_channels"),
    (
        (10, 768, 20, 9, 64, 256),
        (10, 768, 30, 9, 128, None),
    ),
)
def test_polyline_encoder(
    batch_size: int,
    num_polylines: int,
    num_points: int,
    num_dim: int,
    hidden_dim: int,
    out_channels: int | None,
):
    polylines = torch.ones(batch_size, num_polylines, num_points, num_dim)
    polylines_mask = torch.randint(0, 2, (batch_size, num_polylines, num_points), dtype=torch.bool)
    polyline_encoder = PointNetPolylineEncoder(
        in_channels=num_dim,
        hidden_dim=hidden_dim,
        num_layers=6,
        num_pre_layers=3,
        out_channels=out_channels,
    )

    out: Tensor = polyline_encoder(polylines, polylines_mask)
    if out_channels is None:
        assert out.shape == (batch_size, num_polylines, hidden_dim)
    else:
        assert out.shape == (batch_size, num_polylines, out_channels)


@require_cuda
@pytest.mark.parametrize(
    ("batch_size", "num_polylines", "num_points", "num_dim", "hidden_dim", "out_channels", "filename"),
    (
        (10, 768, 20, 9, 64, 256, "test_polyline_encoder_with_out_ch.onnx"),
        (10, 768, 30, 9, 128, None, "test_polyline_encoder_without_out_ch.onnx"),
    ),
)
def test_polyline_encoder_export_onnx(
    tmp_path: Path,
    save_onnx: bool,
    batch_size: int,
    num_polylines: int,
    num_points: int,
    num_dim: int,
    hidden_dim: int,
    out_channels: int | None,
    filename: str,
) -> None:
    polylines = torch.ones(batch_size, num_polylines, num_points, num_dim)
    polylines_mask = torch.ones(batch_size, num_polylines, num_points, dtype=bool)
    dummy_input = (polylines, polylines_mask)

    polyline_encoder = PointNetPolylineEncoder(
        in_channels=num_dim,
        hidden_dim=hidden_dim,
        num_layers=6,
        num_pre_layers=3,
        out_channels=out_channels,
    )

    input_names = ["polylines", "polylines_mask"]
    output_names = ["output"]
    dynamic_axes = {
        "polylines": {0: "batch_size"},
        "polylines_mask": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    if not save_onnx:
        filename = tmp_path / filename

    with torch.no_grad():
        torch.onnx.export(
            polyline_encoder,
            dummy_input,
            filename,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17
        )
