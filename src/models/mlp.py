"""Task 1.0 baseline: 2-layer MLP over flattened 96x96 grayscale."""

from __future__ import annotations

import torch.nn as nn

from src.dataset import IMG_SIZE


class MLP(nn.Module):
    def __init__(self, output_dim: int = 8, hidden: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMG_SIZE * IMG_SIZE, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)
