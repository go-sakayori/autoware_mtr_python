from __future__ import annotations

from omegaconf import DictConfig

from awml_pred.common import DECODERS, DETECTORS, ENCODERS, LAYERS, LOSSES
from awml_pred.typing import Module


def build_model(cfg: DictConfig) -> Module:
    return DETECTORS.build(cfg)


def build_encoder(cfg: DictConfig) -> Module:
    return ENCODERS.build(cfg)


def build_decoder(cfg: DictConfig) -> Module:
    return DECODERS.build(cfg)


def build_loss(cfg: DictConfig) -> Module:
    return LOSSES.build(cfg)


def build_layer(cfg: DictConfig) -> Module:
    return LAYERS.build(cfg)
