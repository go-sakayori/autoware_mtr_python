from typing import Sequence

import torch
from omegaconf import DictConfig

from awml_pred.deploy.rewriters import RewriterContext
from awml_pred.deploy.rewriters.constants import IR, Backend
from awml_pred.typing import Module, Tensor


def export(
    model: Module,
    deploy_cfg: DictConfig,
    filename: str,
    backend: str | Backend = Backend.TENSORRT,
    opset_version: int = 17,
    *,
    verbose: bool = False,
) -> None:
    """Export a model into onnx format.

    Args:
    ----
        model (Module): `Module` instance.
        deploy_cfg (DictConfig): Deployment configuration.
        filename (str): Path to save exported onnx file.
        backend (str | Backend, optional): Name of backend. Defaults to Backend.TENSORRT.
        opset_version (int, optional): Opset version. Defaults to 17.
        verbose (bool, optional): Indicates whether to display logs. Defaults to False.

    """
    input_names: list[str] = list(deploy_cfg.input_names)
    output_names: list[str] = list(deploy_cfg.output_names)
    dummy_input = _load_inputs(deploy_cfg.input_shapes)
    dynamic_axes: DictConfig | None = deploy_cfg.get("dynamic_axes", None)

    if dynamic_axes is not None:
        dynamic_axes: dict[str, dict[int, str]] = _check_dynamic_axes(input_names, output_names, dynamic_axes)

    if isinstance(backend, str):
        backend = Backend.get(backend)

    with RewriterContext(backend=backend, ir=IR.ONNX), torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            filename,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            verbose=verbose,
        )


def _check_dynamic_axes(
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamic_axes: DictConfig,
) -> dict[str, dict[int, str]]:
    """Check all keys of dynamic axes are covered by input and output names, and convert its type.

    Args:
    ----
        input_names (Sequence[str]): Sequence of input names.
        output_names (Sequence[str]): Sequence of output names.
        dynamic_axes (DicConfig): Dict whose axes are dynamic.

    Raises:
    ------
        ValueError: Raises if a set of input and output names does not comprise a set of keys contained `dynamic_axes`.

    Returns:
    -------
        dict[str, dict[int, str]]: Converted from `DictConfig`.

    """
    parameters = (*input_names, *output_names)
    if not set(dynamic_axes.keys()).issubset(parameters):
        raise ValueError(
            "Key names between dynamic axes and inputs/outputs must be same, "
            f"\ninputs: {input_names}\noutputs: {output_names}\ndynamic axes: {list(dynamic_axes.keys())}",
        )

    return {key: dict(items) for key, items in dynamic_axes.items()}


def _load_inputs(input_shapes: DictConfig) -> tuple[Tensor]:
    def _to_dtype(dtype: str) -> torch.dtype:
        if dtype == "float":
            return torch.float32
        elif dtype == "int":
            return torch.int32
        elif dtype == "bool":
            return torch.bool
        else:
            raise ValueError(f"Unexpected dtype: {dtype}")

    inputs: list[Tensor] = []
    for _, items in input_shapes.items():
        shape: list[int] = list(items.shape)
        dtype = _to_dtype(items.dtype)

        value_type: str = items.value
        if value_type == "ones":
            value = torch.ones(shape, dtype=dtype)
        elif value_type == "zeros":
            value = torch.zeros(shape, dtype=dtype)
        elif value_type == "arange":
            assert len(shape) == 1
            value = torch.arange(0, shape[0], dtype=dtype)
        else:
            raise ValueError(f"Unexpected value type: {value_type}")

        inputs.append(value)

    return tuple(inputs)
