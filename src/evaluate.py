"""Test-set evaluator.

Usage:
    uv run python -m src.evaluate --run phase2_cnn                      # defaults to --split test
    uv run python -m src.evaluate --run phase2_cnn --split val
    uv run python -m src.evaluate --config path.yaml --checkpoint path.pt --split test

Reports overall and per-coord RMSE_px, writes a per-sample prediction CSV
into results/ and updates {split}_rmse_px in results/results.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import KeypointsDataset, coord_cols_for, denormalize_target
from src.models import build_model
from src.utils import (
    CHECKPOINTS_DIR, LOGS_DIR, RESULTS_DIR,
    get_device, load_checkpoint, load_config, set_seed,
)


def _update_results_row(run_name: str, split: str, rmse_px: float) -> None:
    path = RESULTS_DIR / "results.csv"
    if not path.exists():
        print(f"[warn] {path} not found; nothing to update")
        return
    df = pd.read_csv(path)
    col = f"{split}_rmse_px"
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].astype(object)
    mask = df["run_name"] == run_name
    if not mask.any():
        print(f"[warn] no row for run_name={run_name!r} in results.csv; skipping update")
        return
    df.loc[mask, col] = f"{rmse_px:.4f}"
    df.to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default=None,
                   help="run_name; reads logs/{run}/config.yaml and checkpoints/{run}_best.pt")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = p.parse_args()

    if args.run:
        cfg_path = LOGS_DIR / args.run / "config.yaml"
        ckpt_path = args.checkpoint or CHECKPOINTS_DIR / f"{args.run}_best.pt"
    else:
        if not args.config:
            raise SystemExit("provide --run or --config")
        cfg_path = args.config
        ckpt_path = args.checkpoint or CHECKPOINTS_DIR / f"{cfg_path.stem}_best.pt"

    cfg = load_config(cfg_path)
    run_name = cfg["run_name"]
    set_seed(int(cfg.get("seed", 42)))
    device = get_device()

    coord_cols = coord_cols_for(cfg["data"]["coord_set"])
    three_channel = bool(cfg["data"].get("three_channel", False))
    split_csv = cfg["data"][f"{args.split}_csv"]
    ds = KeypointsDataset(split_csv, coord_cols, three_channel=three_channel, augmenter=None)
    loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]),
                        shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(cfg["model"])
    load_checkpoint(ckpt_path, model)
    model = model.to(device).eval()

    preds_px_all, targets_px_all = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device, non_blocking=True)
            pred = model(imgs).cpu()
            preds_px_all.append(denormalize_target(pred).numpy())
            targets_px_all.append(denormalize_target(targets).numpy())
    preds_px = np.concatenate(preds_px_all, axis=0)
    targets_px = np.concatenate(targets_px_all, axis=0)
    errs = preds_px - targets_px

    rmse_total = float(np.sqrt((errs ** 2).mean()))
    rmse_per_coord = np.sqrt((errs ** 2).mean(axis=0))

    print(f"[{run_name}] {args.split} RMSE_px total = {rmse_total:.4f}")
    print("Per-coord RMSE_px:")
    for name, v in zip(coord_cols, rmse_per_coord):
        print(f"  {name:35s} {v:7.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / f"predictions_{run_name}_{args.split}.csv"
    header = (["sample_idx"]
              + [f"pred_{c}" for c in coord_cols]
              + [f"target_{c}" for c in coord_cols]
              + [f"err_{c}" for c in coord_cols]
              + ["rmse_px"])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for i in range(len(ds)):
            sample_rmse = float(np.sqrt((errs[i] ** 2).mean()))
            row = ([i] + preds_px[i].tolist() + targets_px[i].tolist()
                   + errs[i].tolist() + [sample_rmse])
            w.writerow(row)
    print(f"Wrote {out_csv}")

    _update_results_row(run_name, args.split, rmse_total)


if __name__ == "__main__":
    main()
