"""Data exploration: null-value analysis for training.csv.

Usage (from repo root):
    uv run python data/explore_data.py

Outputs:
    data/null_analysis.txt  - per-feature null counts, percentages, and summary
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "training.csv"
OUT_PATH = HERE / "null_analysis.txt"

# The 4 "core" keypoints used for Dataset A (8 coordinate columns).
CORE_KEYPOINTS = [
    "left_eye_center_x", "left_eye_center_y",
    "right_eye_center_x", "right_eye_center_y",
    "nose_tip_x", "nose_tip_y",
    "mouth_center_bottom_lip_x", "mouth_center_bottom_lip_y",
]


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Expected dataset at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    n_rows, n_cols = df.shape
    coord_cols = [c for c in df.columns if c != "Image"]
    all_coords_mask = df[coord_cols].notna().all(axis=1)
    core_mask = df[CORE_KEYPOINTS].notna().all(axis=1)

    null_counts = df.isna().sum()
    null_pct = (null_counts / n_rows * 100).round(2)

    # Image column: parse first row to confirm 96x96 length.
    first_img = df["Image"].iloc[0]
    img_pixels = len(first_img.split())
    img_side = int(np.sqrt(img_pixels)) if img_pixels > 0 else 0

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("TRAINING.CSV - NULL VALUE ANALYSIS")
    lines.append("=" * 72)
    lines.append(f"Rows: {n_rows}")
    lines.append(f"Columns: {n_cols}")
    lines.append(f"Image column: {img_pixels} pixels per row (~{img_side}x{img_side})")
    lines.append("")
    lines.append("-" * 72)
    lines.append(f"{'feature':40s}  {'null_count':>10s}  {'null_pct':>9s}")
    lines.append("-" * 72)
    for col in df.columns:
        lines.append(f"{col:40s}  {int(null_counts[col]):>10d}  {null_pct[col]:>8.2f}%")

    lines.append("")
    lines.append("-" * 72)
    lines.append("SAMPLE AVAILABILITY")
    lines.append("-" * 72)
    lines.append(f"Rows with ALL 30 coordinates present (Dataset B candidates): "
                 f"{int(all_coords_mask.sum())}")
    lines.append(f"Rows with 4 core keypoints present (Dataset A candidates):   "
                 f"{int(core_mask.sum())}")
    lines.append("")
    lines.append("Core keypoints used for Dataset A:")
    for c in CORE_KEYPOINTS:
        lines.append(f"  - {c}")

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote null analysis to {OUT_PATH}")


if __name__ == "__main__":
    main()
