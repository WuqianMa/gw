"""KeypointsDataset + image parsing + coord-column constants.

Contracts (read once, honoured everywhere):
    - img tensor:   float32, shape (C, 96, 96), values in [0, 1].
                    C=1 default, C=3 when three_channel=True (grayscale repeated).
    - target tensor: float32, length K = 8 (Dataset A) or 30 (Dataset B),
                    normalised to [-1, 1] via (coord - 48) / 48.
    - augmenter:    callable (img, y_px) -> (img, y_px) applied in PIXEL space
                    BEFORE target normalisation so label transforms stay easy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


COORD_COLS_A: list[str] = [
    "left_eye_center_x", "left_eye_center_y",
    "right_eye_center_x", "right_eye_center_y",
    "nose_tip_x", "nose_tip_y",
    "mouth_center_bottom_lip_x", "mouth_center_bottom_lip_y",
]

COORD_COLS_B: list[str] = [
    "left_eye_center_x", "left_eye_center_y",
    "right_eye_center_x", "right_eye_center_y",
    "left_eye_inner_corner_x", "left_eye_inner_corner_y",
    "left_eye_outer_corner_x", "left_eye_outer_corner_y",
    "right_eye_inner_corner_x", "right_eye_inner_corner_y",
    "right_eye_outer_corner_x", "right_eye_outer_corner_y",
    "left_eyebrow_inner_end_x", "left_eyebrow_inner_end_y",
    "left_eyebrow_outer_end_x", "left_eyebrow_outer_end_y",
    "right_eyebrow_inner_end_x", "right_eyebrow_inner_end_y",
    "right_eyebrow_outer_end_x", "right_eyebrow_outer_end_y",
    "nose_tip_x", "nose_tip_y",
    "mouth_left_corner_x", "mouth_left_corner_y",
    "mouth_right_corner_x", "mouth_right_corner_y",
    "mouth_center_top_lip_x", "mouth_center_top_lip_y",
    "mouth_center_bottom_lip_x", "mouth_center_bottom_lip_y",
]


def coord_cols_for(coord_set: str) -> list[str]:
    if coord_set == "A": return COORD_COLS_A
    if coord_set == "B": return COORD_COLS_B
    raise ValueError(f"coord_set must be 'A' or 'B', got {coord_set!r}")


IMG_SIZE = 96
CENTER = 48.0


def parse_image(s: str) -> torch.Tensor:
    """Parse the space-separated pixel string from the CSV.
    Returns float32 (1, 96, 96), values in [0, 1]."""
    arr = np.fromstring(s, sep=" ", dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def normalize_target(y_px: torch.Tensor) -> torch.Tensor:
    return (y_px - CENTER) / CENTER


def denormalize_target(y_norm: torch.Tensor) -> torch.Tensor:
    return y_norm * CENTER + CENTER


class KeypointsDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        coord_cols: Sequence[str],
        three_channel: bool = False,
        augmenter: Callable | None = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.coord_cols = list(coord_cols)
        self.three_channel = three_channel
        self.augmenter = augmenter
        missing = [c for c in self.coord_cols if c not in self.df.columns]
        if missing:
            raise KeyError(f"{csv_path} missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = parse_image(row["Image"])
        y_px = torch.tensor(
            np.asarray(row[self.coord_cols].values, dtype=np.float32)
        )
        if self.augmenter is not None:
            img, y_px = self.augmenter(img, y_px)
        y = normalize_target(y_px)
        if self.three_channel:
            img = img.repeat(3, 1, 1)
        return img, y
