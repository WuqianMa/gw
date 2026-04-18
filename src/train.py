"""Config-driven training entry point.

Usage:
    uv run python -m src.train --config configs/phase1/task1_0_mlp.yaml
    uv run python -m src.train --config cfg.yaml --override train.epochs=5 seed=7
    uv run python -m src.train --config cfg.yaml --dry-run
    uv run python -m src.train --config stage2.yaml --resume checkpoints/run_stage1_best.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.augmentation import build_augmenter, summarise_augmentation
from src.dataset import KeypointsDataset, coord_cols_for, denormalize_target
from src.models import build_model
from src.models.registry import (
    TRANSFER_MODELS, freeze_backbone, get_backbone_and_head, unfreeze_last_n_blocks,
)
from src.utils import (
    CHECKPOINTS_DIR, REPORTS_DIR,
    append_metrics_row, append_results_row,
    get_device, load_checkpoint, load_config, make_run_dir,
    save_checkpoint, save_config, set_seed,
)


# ---------------------------------------------------------------------------- builders


def _build_loaders(cfg: dict):
    coord_set = cfg["data"]["coord_set"]
    coord_cols = coord_cols_for(coord_set)
    three_channel = bool(cfg["data"].get("three_channel", False))
    augmenter = build_augmenter(
        cfg["data"].get("augmentation"),
        coord_cols=coord_cols,
        coord_set=coord_set,
    )
    train_ds = KeypointsDataset(
        cfg["data"]["train_csv"], coord_cols,
        three_channel=three_channel, augmenter=augmenter,
    )
    val_ds = KeypointsDataset(
        cfg["data"]["val_csv"], coord_cols,
        three_channel=three_channel, augmenter=None,
    )
    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["data"].get("num_workers", 2))
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True,
    )
    return train_loader, val_loader, coord_cols


def _build_optimizer(params_or_groups, name: str, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":  return torch.optim.Adam(params_or_groups, lr=lr, weight_decay=weight_decay)
    if name == "adamw": return torch.optim.AdamW(params_or_groups, lr=lr, weight_decay=weight_decay)
    if name == "sgd":   return torch.optim.SGD(params_or_groups, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {name}")


def _prepare_model_and_param_groups(cfg: dict, resume_path: str | None):
    model = build_model(cfg["model"])
    if resume_path:
        load_checkpoint(Path(resume_path), model, optimizer=None, strict=True)

    model_type = cfg["model"]["type"]
    freeze = bool(cfg["model"].get("freeze_backbone", False))
    unfreeze_n = int(cfg["model"].get("unfreeze_last_n_blocks", 0))

    use_two_groups = False
    head_params: list = []
    backbone_params: list = []

    if freeze and model_type in TRANSFER_MODELS:
        freeze_backbone(model, model_type)
        _, head = get_backbone_and_head(model, model_type)
        if unfreeze_n > 0:
            backbone_params = unfreeze_last_n_blocks(model, model_type, unfreeze_n)
            head_params = [p for p in head.parameters() if p.requires_grad]
            use_two_groups = True
        else:
            head_params = [p for p in head.parameters() if p.requires_grad]

    return model, head_params, backbone_params, use_two_groups


# ---------------------------------------------------------------------------- eval


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Return (avg_loss, rmse_px) on the given loader."""
    model.eval()
    criterion = nn.MSELoss()
    total_sq = 0.0
    total_elems = 0
    total_loss = 0.0
    n_batches = 0
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        pred = model(imgs)
        total_loss += criterion(pred, targets).item()
        n_batches += 1
        total_sq += ((pred - targets) ** 2).sum().item()
        total_elems += targets.numel()
    mse = total_sq / max(total_elems, 1)
    rmse_px = (mse ** 0.5) * 48.0
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, rmse_px


# ---------------------------------------------------------------------------- public entry points


def run_training(cfg: dict, resume_path: str | None = None) -> dict:
    """Run a single training job. Returns a summary dict of final metrics."""
    run_name = cfg["run_name"]
    set_seed(int(cfg.get("seed", 42)))
    device = get_device()

    run_dir = make_run_dir(run_name)
    save_config(cfg, run_dir / "config.yaml")

    resume_path = resume_path or cfg.get("train", {}).get("resume")
    model, head_params, backbone_params, two_groups = _prepare_model_and_param_groups(cfg, resume_path)
    model = model.to(device)

    tcfg = cfg["train"]
    lr = float(tcfg["lr"])
    bb_lr = float(tcfg.get("backbone_lr", lr * 0.1))
    wd = float(tcfg.get("weight_decay", 0.0))
    opt_name = str(tcfg.get("optimizer", "adam"))

    if two_groups:
        optimizer = _build_optimizer(
            [{"params": head_params, "lr": lr},
             {"params": backbone_params, "lr": bb_lr}],
            opt_name, lr=lr, weight_decay=wd,
        )
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = _build_optimizer(trainable, opt_name, lr=lr, weight_decay=wd)

    sch_cfg = tcfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=float(sch_cfg.get("factor", 0.5)),
        patience=int(sch_cfg.get("patience", 5)),
        min_lr=float(sch_cfg.get("min_lr", 1e-6)),
    )

    train_loader, val_loader, _ = _build_loaders(cfg)
    criterion = nn.MSELoss()

    epochs = int(tcfg["epochs"])
    grad_clip = float(tcfg.get("grad_clip", 0.0))
    es_patience = int(tcfg.get("early_stopping", {}).get("patience", 15))

    best_rmse = float("inf")
    best_epoch = -1
    best_train_rmse = float("inf")
    epochs_no_improve = 0
    epoch_times: list[float] = []
    t_total = time.time()
    last_epoch = 0

    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        model.train()
        t0 = time.time()
        train_sq = 0.0; train_elems = 0; train_loss = 0.0; n_batches = 0
        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, targets)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip,
                )
            optimizer.step()
            train_loss += loss.item(); n_batches += 1
            with torch.no_grad():
                train_sq += ((pred - targets) ** 2).sum().item()
                train_elems += targets.numel()

        train_rmse_px = ((train_sq / max(train_elems, 1)) ** 0.5) * 48.0
        avg_train_loss = train_loss / max(n_batches, 1)
        val_loss, val_rmse_px = _evaluate(model, val_loader, device)
        dt = time.time() - t0
        epoch_times.append(dt)
        lr_now = optimizer.param_groups[0]["lr"]

        append_metrics_row(run_dir, {
            "epoch": epoch,
            "lr": f"{lr_now:.2e}",
            "train_loss": f"{avg_train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "train_rmse_px": f"{train_rmse_px:.4f}",
            "val_rmse_px": f"{val_rmse_px:.4f}",
            "epoch_time_sec": f"{dt:.2f}",
        })
        print(f"[{run_name}] ep {epoch:3d}/{epochs}  "
              f"train {train_rmse_px:6.3f}  val {val_rmse_px:6.3f}  "
              f"lr {lr_now:.2e}  ({dt:.1f}s)")

        save_checkpoint(CHECKPOINTS_DIR / f"{run_name}_last.pt",
                        model, optimizer, epoch, val_rmse_px)
        if val_rmse_px < best_rmse:
            best_rmse = val_rmse_px
            best_epoch = epoch
            best_train_rmse = train_rmse_px
            epochs_no_improve = 0
            save_checkpoint(CHECKPOINTS_DIR / f"{run_name}_best.pt",
                            model, optimizer, epoch, val_rmse_px)
        else:
            epochs_no_improve += 1

        scheduler.step(val_rmse_px)
        if epochs_no_improve >= es_patience:
            print(f"[{run_name}] early stop at epoch {epoch} (best={best_rmse:.3f} @ {best_epoch})")
            break

    total_time = time.time() - t_total
    results_row = {
        "run_name": run_name,
        "task_id": cfg.get("task_id", ""),
        "phase": cfg.get("phase", ""),
        "model_type": cfg["model"]["type"],
        "dataset": cfg["data"]["coord_set"],
        "output_dim": cfg["model"]["output_dim"],
        "stage": tcfg.get("stage", "single"),
        "epochs_trained": last_epoch,
        "best_epoch": best_epoch,
        "train_rmse_px": f"{best_train_rmse:.4f}",
        "val_rmse_px": f"{best_rmse:.4f}",
        "test_rmse_px": "",
        "epoch_time_sec_mean": f"{(sum(epoch_times) / max(len(epoch_times), 1)):.2f}",
        "total_time_sec": f"{total_time:.2f}",
        "optimizer": opt_name,
        "lr": lr,
        "backbone_lr": bb_lr if two_groups else "",
        "weight_decay": wd,
        "batch_size": tcfg["batch_size"],
        "dropout": cfg["model"].get("dropout", 0.0),
        "augmentation": summarise_augmentation(cfg["data"].get("augmentation")),
        "unfreeze_last_n_blocks": cfg["model"].get("unfreeze_last_n_blocks", 0),
        "seed": cfg.get("seed", 42),
        "notes": cfg.get("notes", ""),
    }
    append_results_row(results_row)
    print(f"[{run_name}] DONE  best val_rmse_px = {best_rmse:.3f}  @ epoch {best_epoch}")
    return {"best_val_rmse_px": best_rmse, "best_epoch": best_epoch,
            "epochs_trained": last_epoch, "total_time_sec": total_time}


def run_dry_run(cfg: dict) -> None:
    """Pull one batch through the augmentation pipeline and save a preview PNG.
    Use this BEFORE training any geometric-aug config to catch label bugs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_seed(int(cfg.get("seed", 42)))
    train_loader, _, _ = _build_loaders(cfg)
    imgs, targets = next(iter(train_loader))
    imgs = imgs[:8]; targets = targets[:8]
    targets_px = denormalize_target(targets)

    fig, axes = plt.subplots(2, 4, figsize=(10, 5.5))
    for ax, img, y in zip(axes.flat, imgs, targets_px):
        arr = img[0].cpu().numpy()
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        y2 = y.reshape(-1, 2).cpu().numpy()
        ax.scatter(y2[:, 0], y2[:, 1], c="red", s=10, marker="o", edgecolors="none")
        ax.axis("off")
    fig.suptitle(f"dry-run preview: {cfg['run_name']}", fontsize=10)
    fig.tight_layout()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"aug_{cfg['run_name']}_preview.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[dry-run] wrote {out}")


# ---------------------------------------------------------------------------- CLI


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    if args.dry_run:
        run_dry_run(cfg)
        return
    run_training(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
