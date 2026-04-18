"""Random-search hyperparameter tuner for Stage 7.

Search config (YAML) schema:

    base_config: configs/phase5_arch_tweaks/task5_2_custom_stage2.yaml
    run_name_prefix: phase7_tune
    n_trials: 30
    space:
      train.lr:           {log_uniform: [1.0e-5, 5.0e-3]}
      train.optimizer:    {choice: [adam, adamw, sgd]}
      train.weight_decay: {log_uniform: [1.0e-6, 1.0e-3]}
      train.batch_size:   {choice: [16, 32, 64]}
      model.dropout:      {uniform: [0.1, 0.5]}
      data.augmentation.flip_prob: {choice: [0.0, 0.3, 0.5, 0.8]}

Each trial:
    - resolves a sampled config, dumps it to {search_cfg.parent}/_trials/{run_name}.yaml
    - runs src.train.run_training(cfg) in-process
    - appends a row to results/tune_results.csv including the sampled hyperparams
      and the trial's best val_rmse_px

Parallelise by launching multiple processes with disjoint `--start/--end` ranges.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.train import run_training
from src.utils import RESULTS_DIR, load_config, save_config


def _sample_value(rng: np.random.Generator, spec: dict) -> Any:
    if "choice" in spec:
        choices = spec["choice"]
        return choices[int(rng.integers(0, len(choices)))]
    if "uniform" in spec:
        lo, hi = spec["uniform"]
        return float(rng.uniform(float(lo), float(hi)))
    if "log_uniform" in spec:
        lo, hi = spec["log_uniform"]
        u = rng.uniform(math.log(float(lo)), math.log(float(hi)))
        return float(math.exp(u))
    raise ValueError(f"unknown spec {spec}; expected choice | uniform | log_uniform")


def _set_nested(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    args = p.parse_args()

    search_cfg = load_config(args.config)
    base_cfg = load_config(Path(search_cfg["base_config"]))
    prefix = search_cfg["run_name_prefix"]
    n_trials = int(search_cfg["n_trials"])
    start = args.start
    end = n_trials if args.end is None else args.end
    space: dict = search_cfg["space"]

    trials_dir = args.config.parent / "_trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tune_csv = RESULTS_DIR / "tune_results.csv"

    for t in range(start, end):
        rng = np.random.default_rng(12345 + t)
        cfg = copy.deepcopy(base_cfg)
        sampled: dict[str, Any] = {}
        for key, spec in space.items():
            v = _sample_value(rng, spec)
            if isinstance(v, np.generic): v = v.item()
            _set_nested(cfg, key, v)
            sampled[key] = v

        run_name = f"{prefix}_{t:03d}"
        cfg["run_name"] = run_name
        cfg["seed"] = 42 + t
        trial_cfg_path = trials_dir / f"{run_name}.yaml"
        save_config(cfg, trial_cfg_path)

        print(f"\n=== trial {t:03d}/{n_trials - 1} ===")
        for k, v in sampled.items():
            print(f"  {k} = {v}")
        summary = run_training(cfg)

        row: dict[str, Any] = {
            "trial_idx": t,
            "run_name": run_name,
            "val_rmse_px": f"{summary['best_val_rmse_px']:.4f}",
            "best_epoch": summary["best_epoch"],
            "epochs_trained": summary["epochs_trained"],
            "total_time_sec": f"{summary['total_time_sec']:.2f}",
        }
        for k, v in sampled.items():
            row[f"p_{k}"] = v

        exists = tune_csv.exists()
        with open(tune_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists: w.writeheader()
            w.writerow(row)

    if tune_csv.exists():
        import pandas as pd
        df = pd.read_csv(tune_csv)
        df["val_rmse_px"] = pd.to_numeric(df["val_rmse_px"], errors="coerce")
        print("\nTop-5 trials by val_rmse_px:")
        print(df.sort_values("val_rmse_px").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
