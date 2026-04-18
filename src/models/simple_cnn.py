"""Task 1.1 baseline: two conv+pool blocks, FC head."""

from __future__ import annotations

import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, output_dim: int = 8, dropout: float = 0.0, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # input 96x96 -> 48x48 -> 24x24 after two MaxPool(2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 24 * 24, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
