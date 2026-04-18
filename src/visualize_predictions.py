"""Stage 8 error-analysis plots for a trained model.

Inputs:
    results/predictions_{run_name}_{split}.csv (written by src.evaluate)
    data/{dataset_a|dataset_b}/dataset_{a|b}_{split}.csv (for the raw images)
    logs/{run_name}/config.yaml (for coord_set + three_channel)

Outputs (under reports/):
    {run}_{split}_per_keypoint_rmse.png   bar chart of per-coord pixel RMSE
    {run}_{split}_error_hist.png          histogram of per-sample total RMSE_px
    {run}_{split}_worst12.png             3x4 gallery, gt (green) vs pred (red)
    {run}_{split}_best12.png              3x4 gallery, same overlay
    {run}_{split}_residual_heatmap.png    scatter of (pred - gt) residuals

Usage:
    uv run python -m src.visualize_predictions --run phase2_cnn
    uv run python -m src.visualize_predictions --run arch_winner --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataset import KeypointsDataset, coord_cols_for
from src.utils import LOGS_DIR, REPORTS_DIR, RESULTS_DIR, load_config


def _load_predictions(run_name: str, split: str, coord_cols: list[str]):
    path = RESULTS_DIR / f"predictions_{run_name}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `uv run python -m src.evaluate --run {run_name} --split {split}` first"
        )
    df = pd.read_csv(path)
    n = len(df)
    K = len(coord_cols)
    preds = np.stack([df[f"pred_{c}"].to_numpy() for c in coord_cols], axis=1)
    targets = np.stack([df[f"target_{c}"].to_numpy() for c in coord_cols], axis=1)
    errs = preds - targets
    assert preds.shape == (n, K)
    per_sample_rmse = df["rmse_px"].to_numpy()
    return preds, targets, errs, per_sample_rmse


def _keypoint_names(coord_cols: list[str]) -> list[str]:
    """Collapse *_x/_y to one name per keypoint."""
    names: list[str] = []
    for c in coord_cols:
        if c.endswith("_x"):
            names.append(c[:-2])
    return names


def _plot_per_keypoint_rmse(errs: np.ndarray, coord_cols: list[str], out: Path) -> None:
    """Per-keypoint RMSE = sqrt(mean(err_x^2 + err_y^2)) over all samples."""
    names = _keypoint_names(coord_cols)
    K = len(names)
    per_kp = np.zeros(K)
    for i in range(K):
        dx = errs[:, 2 * i]
        dy = errs[:, 2 * i + 1]
        per_kp[i] = float(np.sqrt((dx ** 2 + dy ** 2).mean()))

    order = np.argsort(per_kp)[::-1]
    names_sorted = [names[i] for i in order]
    vals_sorted = per_kp[order]

    fig, ax = plt.subplots(figsize=(max(6, 0.35 * K + 3), 4.5))
    ax.bar(range(K), vals_sorted, color="steelblue")
    ax.set_xticks(range(K))
    ax.set_xticklabels(names_sorted, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RMSE (px)")
    ax.set_title("Per-keypoint RMSE (hardest → easiest)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_error_hist(per_sample_rmse: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(per_sample_rmse, bins=40, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(float(per_sample_rmse.mean()), color="red", ls="--",
               label=f"mean = {per_sample_rmse.mean():.2f}")
    ax.axvline(float(np.median(per_sample_rmse)), color="green", ls="--",
               label=f"median = {np.median(per_sample_rmse):.2f}")
    ax.set_xlabel("per-sample RMSE (px)")
    ax.set_ylabel("count")
    ax.set_title("Error distribution over test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_gallery(
    ds: KeypointsDataset,
    preds_px: np.ndarray,
    targets_px: np.ndarray,
    per_sample_rmse: np.ndarray,
    indices: np.ndarray,
    title: str,
    out: Path,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for ax, idx in zip(axes.flat, indices):
        img, _ = ds[int(idx)]
        arr = img[0].cpu().numpy() if img.shape[0] == 1 else img.mean(0).cpu().numpy()
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        t = targets_px[idx].reshape(-1, 2)
        p = preds_px[idx].reshape(-1, 2)
        ax.scatter(t[:, 0], t[:, 1], c="lime", s=12, marker="o",
                   edgecolors="black", linewidths=0.3, label="gt")
        ax.scatter(p[:, 0], p[:, 1], c="red", s=12, marker="x", label="pred")
        ax.set_title(f"#{idx}  rmse={per_sample_rmse[idx]:.2f}px", fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_residual_heatmap(errs: np.ndarray, out: Path) -> None:
    """Scatter of (err_x, err_y) across every keypoint of every sample."""
    dx = errs[:, 0::2].reshape(-1)
    dy = errs[:, 1::2].reshape(-1)
    lim = float(max(np.abs(dx).max(), np.abs(dy).max(), 1.0))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.hexbin(dx, dy, gridsize=50, cmap="viridis", mincnt=1)
    ax.axhline(0, color="white", lw=0.5); ax.axvline(0, color="white", lw=0.5)
    ax.scatter([0], [0], marker="+", c="red", s=80)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("x residual (pred - gt, px)")
    ax.set_ylabel("y residual (pred - gt, px)")
    ax.set_aspect("equal")
    ax.set_title(f"Residual heatmap — mean=({dx.mean():+.2f}, {dy.mean():+.2f})")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=str,
                   help="run_name; reads logs/{run}/config.yaml + results/predictions_{run}_{split}.csv")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = p.parse_args()

    cfg_path = LOGS_DIR / args.run / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"{cfg_path} not found — was this run trained?")
    cfg = load_config(cfg_path)
    coord_cols = coord_cols_for(cfg["data"]["coord_set"])
    three_channel = bool(cfg["data"].get("three_channel", False))
    split_csv = cfg["data"][f"{args.split}_csv"]

    preds_px, targets_px, errs, per_sample_rmse = _load_predictions(
        args.run, args.split, coord_cols,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.run}_{args.split}"

    _plot_per_keypoint_rmse(errs, coord_cols, REPORTS_DIR / f"{stem}_per_keypoint_rmse.png")
    _plot_error_hist(per_sample_rmse, REPORTS_DIR / f"{stem}_error_hist.png")
    _plot_residual_heatmap(errs, REPORTS_DIR / f"{stem}_residual_heatmap.png")

    ds = KeypointsDataset(split_csv, coord_cols,
                          three_channel=three_channel, augmenter=None)
    order = np.argsort(per_sample_rmse)
    n = min(12, len(per_sample_rmse))
    best = order[:n]
    worst = order[-n:][::-1]
    _plot_gallery(ds, preds_px, targets_px, per_sample_rmse, worst,
                  f"Worst {n} predictions — {args.run} ({args.split})",
                  REPORTS_DIR / f"{stem}_worst12.png")
    _plot_gallery(ds, preds_px, targets_px, per_sample_rmse, best,
                  f"Best {n} predictions — {args.run} ({args.split})",
                  REPORTS_DIR / f"{stem}_best12.png")

    print(f"Wrote 5 plots to {REPORTS_DIR}/ with prefix {stem}_")
    print(f"  mean per-sample RMSE = {per_sample_rmse.mean():.3f} px")
    print(f"  median               = {np.median(per_sample_rmse):.3f} px")
    print(f"  p90                  = {np.percentile(per_sample_rmse, 90):.3f} px")
    print(f"  max                  = {per_sample_rmse.max():.3f} px")


if __name__ == "__main__":
    main()
