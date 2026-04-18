"""Model dispatch + freeze/unfreeze helpers.

To add a model: write a builder, register it in `build_model`, and if it's a
transfer-learning model add it to `TRANSFER_MODELS` and (optionally) the
`_BACKBONE_HEAD` table so the freeze helpers work.
"""

from __future__ import annotations

import torch.nn as nn

from .mlp import MLP
from .simple_cnn import SimpleCNN
from .transfer import (
    build_efficientnet_b0,
    build_mobilenet_v2,
    build_mobilenet_v2_custom,
    build_mobilenet_v2_stem,
    build_squeezenet,
)


def build_model(model_cfg: dict) -> nn.Module:
    t = model_cfg["type"]
    out = int(model_cfg["output_dim"])
    dropout = float(model_cfg.get("dropout", 0.0))

    if t == "mlp":                      return MLP(output_dim=out)
    if t == "simple_cnn":               return SimpleCNN(output_dim=out, dropout=dropout)
    if t == "squeezenet":               return build_squeezenet(output_dim=out, dropout=dropout or 0.3)
    if t == "mobilenet_v2":             return build_mobilenet_v2(output_dim=out, dropout=dropout or 0.2)
    if t == "mobilenet_v2_stem":        return build_mobilenet_v2_stem(output_dim=out, dropout=dropout or 0.2)
    if t == "mobilenet_v2_custom":      return build_mobilenet_v2_custom(output_dim=out, dropout=dropout or 0.4)
    if t == "efficientnet_b0":          return build_efficientnet_b0(output_dim=out, dropout=dropout or 0.2)
    raise ValueError(f"unknown model type: {t}")


TRANSFER_MODELS: set[str] = {
    "squeezenet", "mobilenet_v2", "mobilenet_v2_stem",
    "mobilenet_v2_custom", "efficientnet_b0",
}


def get_backbone_and_head(model: nn.Module, model_type: str) -> tuple[nn.Module, nn.Module]:
    if model_type == "mobilenet_v2_custom":
        return model.features, model.head
    if model_type in {"mobilenet_v2", "mobilenet_v2_stem",
                      "efficientnet_b0", "squeezenet"}:
        return model.features, model.classifier
    raise ValueError(f"no backbone/head split known for {model_type}")


def freeze_backbone(model: nn.Module, model_type: str) -> None:
    backbone, head = get_backbone_and_head(model, model_type)
    for p in backbone.parameters(): p.requires_grad = False
    for p in head.parameters():     p.requires_grad = True


def unfreeze_last_n_blocks(model: nn.Module, model_type: str, n: int) -> list:
    """Unfreeze last n blocks of `.features`. Returns the newly-trainable params."""
    if n <= 0: return []
    backbone, _ = get_backbone_and_head(model, model_type)
    blocks = list(backbone.children())
    unfrozen: list = []
    for blk in blocks[-n:]:
        for p in blk.parameters():
            p.requires_grad = True
            unfrozen.append(p)
    return unfrozen
