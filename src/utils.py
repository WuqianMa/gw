"""Shared utilities: seeding, device selection, config I/O, checkpointing, logging.

Everything that is not model- or task-specific lives here so every script in
`src/` can import the same helpers.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"


# ---------------------------------------------------------------------------- seeding + device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    # Colab T4 / Linux CUDA: uncomment the next line
    # if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------- config I/O


def _parse_cli_value(v: str) -> Any:
    low = v.lower()
    if low in {"true", "false"}: return low == "true"
    if low in {"none", "null"}: return None
    try: return int(v)
    except ValueError: pass
    try: return float(v)
    except ValueError: pass
    return v


def _set_nested(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def apply_cli_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Each override is 'dotted.key=value'; value is parsed to int/float/bool/str."""
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        key, raw = ov.split("=", 1)
        _set_nested(cfg, key, _parse_cli_value(raw))
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        apply_cli_overrides(cfg, overrides)
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------- logging + results


def make_run_dir(run_name: str) -> Path:
    run_dir = LOGS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_metrics_row(run_dir: Path, row: dict) -> None:
    path = Path(run_dir) / "metrics.csv"
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists: w.writeheader()
        w.writerow(row)


def append_results_row(row: dict) -> None:
    """Append to results/results.csv, preserving any existing columns."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "results.csv"
    existing: list[dict] = []
    header: list[str] = list(row.keys())
    if path.exists():
        with open(path, "r", newline="") as f:
            r = csv.DictReader(f)
            header = list(r.fieldnames or header)
            existing = list(r)
    for k in row:
        if k not in header: header.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for er in existing: w.writerow({k: er.get(k, "") for k in header})
        w.writerow({k: row.get(k, "") for k in header})


# ---------------------------------------------------------------------------- checkpointing


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    val_rmse_px: float,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "val_rmse_px": val_rmse_px,
        "extra": extra or {},
    }
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
) -> dict:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"], strict=strict)
    if optimizer is not None and state.get("optimizer_state"):
        optimizer.load_state_dict(state["optimizer_state"])
    return state
