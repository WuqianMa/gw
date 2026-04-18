"""Sanity-check visualization: 12 random training images with keypoints overlaid.

Usage (from repo root):
    uv run python data/visualize_samples.py                   # uses training.csv
    uv run python data/visualize_samples.py --csv data/dataset_b/dataset_b_train.csv

Outputs:
    data/samples_preview.png  - 3x4 grid, image + red keypoint dots
    data/keypoint_stats.txt   - per-coord mean/std in pixels (useful before aug)

Why run this first: the #1 rule of practical deep learning is "look at your
data before you train anything." This script enforces that for the team.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "training.csv"
DEFAULT_IMG = HERE / "samples_preview.png"
DEFAULT_STATS = HERE / "keypoint_stats.txt"

SEED = 42
N_ROWS, N_COLS = 3, 4   # 12 samples


def parse_image(s: str) -> np.ndarray:
    return np.fromstring(s, sep=" ", dtype=np.uint8).reshape(96, 96)


def sample_grid(df: pd.DataFrame, coord_cols: list[str], out_path: Path) -> None:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=N_ROWS * N_COLS, replace=False)
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 2.3, N_ROWS * 2.3))
    for ax, i in zip(axes.flat, idx):
        row = df.iloc[i]
        img = parse_image(row["Image"])
        ax.imshow(img, cmap="gray")
        xs = [row[c] for c in coord_cols if c.endswith("_x") and not pd.isna(row[c])]
        ys = [row[c] for c in coord_cols if c.endswith("_y") and not pd.isna(row[c])]
        ax.scatter(xs, ys, s=10, c="red", marker="o", edgecolors="none")
        ax.set_title(f"row {i}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{len(coord_cols)//2} keypoints over {out_path.parent.name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def coord_stats(df: pd.DataFrame, coord_cols: list[str], out_path: Path) -> None:
    stats = df[coord_cols].agg(["count", "mean", "std", "min", "max"]).T.round(2)
    lines = [
        "=" * 72,
        f"KEYPOINT COORDINATE STATS (pixel space, image is 96x96)",
        "=" * 72,
        stats.to_string(),
        "",
        "Useful sanity checks:",
        "  * means should cluster in [20, 75] - keypoints are on a face,",
        "    not at image edges.",
        "  * max should never exceed 96; min should never be below 0.",
        "  * std tells you how much a keypoint moves between faces; small std",
        "    (e.g. mouth_center_*) means the position is easier to predict.",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out-image", type=Path, default=DEFAULT_IMG)
    p.add_argument("--out-stats", type=Path, default=DEFAULT_STATS)
    args = p.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)
    coord_cols = [c for c in df.columns if c != "Image"]
    sample_grid(df, coord_cols, args.out_image)
    coord_stats(df, coord_cols, args.out_stats)
    print(f"Wrote {args.out_image}")
    print(f"Wrote {args.out_stats}")


if __name__ == "__main__":
    main()
