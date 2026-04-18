"""Transfer-learning builders for Stages 4 and 5.

Each `build_*` function returns an `nn.Module` whose `.features` attribute is
the pretrained backbone (to be frozen) and whose `.classifier` or `.head`
attribute is the replaced regression head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    SqueezeNet1_1_Weights,
    efficientnet_b0,
    mobilenet_v2,
    squeezenet1_1,
)


# ---------------------------------------------------------------------------- SqueezeNet


def build_squeezenet(output_dim: int, dropout: float = 0.3) -> nn.Module:
    m = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Conv2d(512, output_dim, kernel_size=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    m.num_classes = output_dim
    return m


# ---------------------------------------------------------------------------- MobileNetV2


def _replace_stem_stride1(m: nn.Module) -> None:
    """Replace MobileNetV2 features[0][0] with a stride=1 conv, copying weights."""
    old = m.features[0][0]
    new = nn.Conv2d(
        old.in_channels, old.out_channels,
        kernel_size=old.kernel_size, stride=1,
        padding=old.padding, bias=(old.bias is not None),
    )
    with torch.no_grad():
        new.weight.copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    m.features[0][0] = new


def build_mobilenet_v2(output_dim: int, dropout: float = 0.2) -> nn.Module:
    m = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, output_dim))
    return m


def build_mobilenet_v2_stem(output_dim: int, dropout: float = 0.2) -> nn.Module:
    m = build_mobilenet_v2(output_dim=output_dim, dropout=dropout)
    _replace_stem_stride1(m)
    return m


class MobileNetV2Custom(nn.Module):
    """Stride-1 stem + custom regression head with BN + Dropout.

    Attributes `.features` (backbone) and `.head` (replacement head) so the
    freeze/unfreeze helpers in `registry.py` can find them generically.
    """

    def __init__(self, output_dim: int, dropout: float = 0.4):
        super().__init__()
        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        _replace_stem_stride1(base)
        self.features = base.features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_mobilenet_v2_custom(output_dim: int, dropout: float = 0.4) -> nn.Module:
    return MobileNetV2Custom(output_dim=output_dim, dropout=dropout)


# ---------------------------------------------------------------------------- EfficientNet-B0


def build_efficientnet_b0(output_dim: int, dropout: float = 0.2) -> nn.Module:
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, output_dim))
    return m
