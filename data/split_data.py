"""Build Dataset A and Dataset B and write train/val/test CSV splits.

Usage (from repo root):
    uv run python data/split_data.py

Produces:
    data/dataset_a/dataset_a_{train,val,test}.csv
    data/dataset_b/dataset_b_{train,val,test}.csv
    data/split_summary.txt

Split ratio: 70% train / 15% val / 15% test. Seed = 42.
The Image column is preserved as a space-separated string in every split
so downstream Dataset classes can parse it once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "training.csv"
SUMMARY_PATH = HERE / "split_summary.txt"
DIR_A = HERE / "dataset_a"
DIR_B = HERE / "dataset_b"

SEED = 42
VAL_FRAC = 0.15   # of full dataset
TEST_FRAC = 0.15  # of full dataset

CORE_KEYPOINTS = [
    "left_eye_center_x", "left_eye_center_y",
    "right_eye_center_x", "right_eye_center_y",
    "nose_tip_x", "nose_tip_y",
    "mouth_center_bottom_lip_x", "mouth_center_bottom_lip_y",
]


def three_way_split(
    df: pd.DataFrame,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministic 70/15/15 split via two successive train_test_split calls."""
    assert 0 < val_frac < 1 and 0 < test_frac < 1
    assert val_frac + test_frac < 1
    # First carve off test.
    train_val, test = train_test_split(df, test_size=test_frac, random_state=seed)
    # Then carve val out of the remainder so val is val_frac of the ORIGINAL set.
    val_relative = val_frac / (1 - test_frac)
    train, val = train_test_split(train_val, test_size=val_relative, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_dataset_a(df: pd.DataFrame) -> pd.DataFrame:
    """4 core keypoints + Image; drop rows missing any of those columns."""
    cols = CORE_KEYPOINTS + ["Image"]
    return df[cols].dropna(subset=CORE_KEYPOINTS).reset_index(drop=True)


def build_dataset_b(df: pd.DataFrame) -> pd.DataFrame:
    """All 15 keypoints (30 coord cols) + Image; drop rows missing any coord."""
    coord_cols = [c for c in df.columns if c != "Image"]
    return df.dropna(subset=coord_cols).reset_index(drop=True)


def write_splits(name: str, frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> list[str]:
    train, val, test = frames
    out_dir = HERE / f"dataset_{name}"
    out_dir.mkdir(exist_ok=True)
    paths = []
    for split_name, frame in (("train", train), ("val", val), ("test", test)):
        out = out_dir / f"dataset_{name}_{split_name}.csv"
        frame.to_csv(out, index=False)
        paths.append(f"  {out.relative_to(HERE.parent).as_posix():45s} rows={len(frame)}")
    return paths


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Expected dataset at {CSV_PATH}")

    np.random.seed(SEED)
    df = pd.read_csv(CSV_PATH)

    dataset_a = build_dataset_a(df)
    dataset_b = build_dataset_b(df)

    a_splits = three_way_split(dataset_a)
    b_splits = three_way_split(dataset_b)

    a_lines = write_splits("a", a_splits)
    b_lines = write_splits("b", b_splits)

    summary = [
        "=" * 72,
        "DATASET SPLIT SUMMARY",
        "=" * 72,
        f"Source: {CSV_PATH.name} (rows={len(df)})",
        f"Seed: {SEED}  |  Split: 70% train / 15% val / 15% test",
        "",
        f"Dataset A (4 core keypoints, target dim = 8)  rows={len(dataset_a)}",
        *a_lines,
        "",
        f"Dataset B (15 full keypoints, target dim = 30) rows={len(dataset_b)}",
        *b_lines,
    ]
    SUMMARY_PATH.write_text("\n".join(summary) + "\n")
    print("\n".join(summary))
    print(f"\nWrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
