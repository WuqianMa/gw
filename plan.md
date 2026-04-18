# EXPERIMENT OPERATION PLAN — FACIAL KEYPOINTS DETECTION

> **Version:** 3 — rewritten from a practical deep-learning tutor's perspective.
> Every stage is self-contained, lists its inputs and outputs, and can be
> resumed cold by a new teammate without rereading the whole document.
>
> **What changed from v2:**
> - Added §0 with explicit learning outcomes (what the tutor wants the team to walk away understanding).
> - Promoted **data augmentation** to its own Stage (Stage 3) with an 8-way ablation, because aug is the single biggest lever on datasets this small.
> - Added **Stage 0** visualization step — "look at your data" is non-negotiable before any training.
> - Added **Stage 8** error analysis (per-keypoint RMSE, worst/best predictions, error histograms) because quantitative metrics alone mislead.
> - Every stage now has the same fixed structure: **Learning goal / Prerequisites / Commands / Outputs / Success criteria / Handoff**.
> - Folder skeleton is pre-created so runs land in consistent locations.
>
> **One rule that drives everything below:** any operation that touches data — loading, splitting, augmenting, predicting — lives in a Python file on disk and is invoked by an exact command. No notebook shortcuts. The cost of enforcing this is low; the cost of irreproducible results in a group project is high.

---

## 0. Learning Outcomes (tutor's expectation)

By the end of this project the team should be able to answer, with evidence
from their own runs, these questions:

1. **Pipeline fluency.** Build a full train/val/test pipeline in PyTorch that loads raw CSV, parses images, normalizes targets, trains with early stopping, saves checkpoints, logs per-epoch metrics. (Stages 0–2.)
2. **Inductive bias.** Why does a CNN beat an MLP on image regression even at identical parameter counts? What does the first conv layer "see"? (Stage 2.)
3. **Regularization via augmentation.** For keypoint regression, which augmentations help, which hurt, and why do *geometric* augmentations require transforming the labels? Produce an ablation table. (Stage 3.)
4. **Transfer learning.** What does "freeze the backbone" mean mechanically, and why do we use a *lower* learning rate when we unfreeze it? (Stage 4.)
5. **Architectural inductive-bias tweaks.** Why does making the stem `stride=1` help at low input resolution? What does adding BatchNorm + Dropout in the head buy you? (Stage 5.)
6. **Data scarcity.** Dataset B has 2,140 rows. What changes about the training recipe when you move there? (Stage 6.)
7. **Hyperparameter search discipline.** Random vs grid, seed sweeps for stability, validation vs test separation, and what "well-tuned" actually means. (Stage 7.)
8. **Error analysis.** Which keypoints is the model worst at? Look at the 12 worst predictions — what do they have in common? A number in a results table is not a finding; a failure mode is. (Stage 8.)

Every stage below produces artefacts that go into the final report. If a stage finishes without producing numbers/plots a teammate could paste into the report, it is not done.

---

## 1. Repo Layout (skeleton is already created)

```
GROUPPROJECT/
├── pyproject.toml                uv project config
├── uv.lock                       pinned dep lockfile — commit it
├── plan.md                       this file
├── .gitignore
├── data/
│   ├── training.csv              raw Kaggle file (7,049 × 31)             [INPUT]
│   ├── explore_data.py           null analysis                             [WRITTEN]
│   ├── visualize_samples.py      plot 12 random samples                    [WRITTEN]
│   ├── split_data.py             70/15/15 splits                           [WRITTEN]
│   ├── null_analysis.txt         generated
│   ├── keypoint_stats.txt        generated
│   ├── samples_preview.png       generated
│   ├── split_summary.txt         generated
│   ├── dataset_a/                generated — 8 coords, 7,000 rows
│   │   ├── dataset_a_train.csv    (4,900)
│   │   ├── dataset_a_val.csv      (1,050)
│   │   └── dataset_a_test.csv     (1,050)
│   └── dataset_b/                generated — 30 coords, 2,140 rows
│       ├── dataset_b_train.csv    (1,498)
│       ├── dataset_b_val.csv      (  321)
│       └── dataset_b_test.csv     (  321)
├── src/                          (team writes these — spec in §4)
│   ├── __init__.py
│   ├── utils.py                    seed, device, logging, checkpoints, YAML
│   ├── dataset.py                  KeypointsDataset, parse_image
│   ├── augmentation.py             ALL augmentation transforms (§4.3)
│   ├── train.py                    config-driven training entry
│   ├── tune.py                     random-search driver (Stage 7)
│   ├── evaluate.py                 test-set evaluator
│   ├── visualize_predictions.py    error analysis plots (Stage 8)
│   └── models/
│       ├── __init__.py
│       ├── registry.py             build_model(cfg.model) dispatch
│       ├── mlp.py
│       ├── simple_cnn.py
│       └── transfer.py
├── configs/
│   ├── phase1/                  Stage 1 — MLP baseline
│   ├── phase2/                  Stage 2 — CNN baseline
│   ├── phase3_augmentation/     Stage 3 — aug ablation (8 configs)
│   ├── phase4_transfer/         Stage 4 — transfer learning
│   ├── phase5_arch_tweaks/      Stage 5 — stem + custom head
│   ├── phase6_full_output/      Stage 6 — Dataset B, 30 coords
│   └── phase7_tuning/           Stage 7 — random search
├── logs/{run_name}/             generated — config.yaml + metrics.csv
├── checkpoints/                 generated — {run_name}_best.pt / _last.pt
├── results/
│   ├── results.csv              aggregated across all runs
│   └── tune_results.csv         Stage 7 trials
└── reports/
    └── (figures, tables, writeup drafts)
```

Skeleton folders (`src/`, `src/models/`, `configs/phase*`, `logs/`, `checkpoints/`, `results/`, `reports/`) are already created and kept under git with `.gitkeep` stubs.

---

## 2. Environment & Reproducibility

### 2.1 uv package manager
```bash
# one-time, from repo root
uv sync                     # installs .venv from pyproject.toml + uv.lock
uv add <package>            # use this for every new dependency — never pip install
uv run python <script>      # preferred invocation (no activate needed)
```

### 2.2 Device (inside `src/utils.py::get_device()`)
```python
import torch
def get_device() -> torch.device:
    if torch.backends.mps.is_available(): return torch.device("mps")
    # Colab / Linux CUDA: uncomment the next line
    # if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")
```

### 2.3 Seeding (inside `src/utils.py::set_seed(seed)`)
```python
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
**Every script that can train a model calls `set_seed(cfg.seed)` before building the dataset.**

---

## 3. Stage 0 — Explore the Data

**Learning goal.** Learn the shape of the problem before modelling. Missing data, label distribution, and what an input actually looks like should all be in your head before you train anything.

### Step 0.1 · Null analysis

| | |
|---|---|
| **Prerequisites** | `data/training.csv` exists. Environment installed via `uv sync`. |
| **Command** | `uv run python data/explore_data.py` |
| **Outputs** | `data/null_analysis.txt` |
| **Success criteria** | File contains: `Rows: 7049`, `Image column: 9216 pixels per row`, Dataset A candidates = **7,000**, Dataset B candidates = **2,140**. |
| **Handoff** | Read the file. Confirm the 11 "hard" keypoints (inner/outer corners, eyebrows, mouth corners, top-lip) are all 67–68% missing — this is *why* we build two datasets. |

### Step 0.2 · Visual EDA

| | |
|---|---|
| **Prerequisites** | Step 0.1 done. |
| **Command** | `uv run python data/visualize_samples.py` |
| **Outputs** | `data/samples_preview.png` (3×4 grid), `data/keypoint_stats.txt` (per-coord mean/std). |
| **Success criteria** | Open the PNG. Red dots should sit on actual facial landmarks. Coord means in the stats file should lie in `[20, 75]` (faces are centered); min ≥ 0 and max ≤ 96 for every column. |
| **Handoff** | Put `samples_preview.png` into `reports/` — it goes in the writeup. Note which keypoint has the smallest std (it'll be the easiest to predict; a useful baseline sanity check later). |

### Step 0.3 · Train/val/test splits

| | |
|---|---|
| **Prerequisites** | Steps 0.1, 0.2 done. |
| **Command** | `uv run python data/split_data.py` |
| **Outputs** | `data/dataset_a/dataset_a_{train,val,test}.csv`, `data/dataset_b/dataset_b_{train,val,test}.csv`, `data/split_summary.txt` |
| **Success criteria** | Summary shows A split 4,900/1,050/1,050 and B split 1,498/321/321. Seed = 42. |
| **Handoff** | **Do not touch the test CSVs until Stage 8.** Any accidental fit-on-test — even by spot-checking — invalidates the reported numbers. |

---

## 4. Shared Code Contracts (`src/`)

Two people write, one reviews. Ship this **before** Stage 1 begins so nobody forks their own trainer.

### 4.1 `src/dataset.py`
```python
COORD_COLS_A = [
    "left_eye_center_x", "left_eye_center_y",
    "right_eye_center_x", "right_eye_center_y",
    "nose_tip_x", "nose_tip_y",
    "mouth_center_bottom_lip_x", "mouth_center_bottom_lip_y",
]
# COORD_COLS_B = every column except "Image".

def parse_image(s: str) -> torch.Tensor:
    """'238 236 ...' -> float32 tensor (1, 96, 96), values in [0, 1]."""
    arr = np.fromstring(s, sep=" ", dtype=np.float32).reshape(96, 96) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

class KeypointsDataset(Dataset):
    """Returns (img, target) per item.
       - img: float32 (C, 96, 96), C=1 default, C=3 if three_channel=True
       - target: float32 length-K vector normalized to [-1, 1]: (coord-48)/48
       - augmenter: callable (img, target_px) -> (img, target_px) applied when provided
    """
    def __init__(self, csv_path, coord_cols, three_channel=False, augmenter=None):
        ...
```
**Contract.** Every model, every phase, consumes the same `(img, target)` shape. Augmentation is injected, never baked in.

### 4.2 `src/utils.py`
Must expose:
```python
set_seed(seed)
get_device() -> torch.device
load_config(path, overrides=None) -> dict            # YAML + CLI overrides
save_checkpoint(path, model, optimizer, epoch, val_rmse_px)
load_checkpoint(path, model, optimizer=None) -> dict
make_run_dir(run_name) -> Path                       # logs/{run_name}/
append_metrics_row(run_dir, row_dict)                # logs/{run_name}/metrics.csv
append_results_row(row_dict)                         # results/results.csv
```
All paths resolved relative to the repo root so `uv run python -m src.train` works from anywhere.

### 4.3 `src/augmentation.py` — *this is the Stage-3 lesson made concrete*

Augmentations operate on `(img, y_px)` where `y_px` is the **un-normalized** coord tensor (length K). The training pipeline un-normalizes → augments → re-normalizes, so geometric augs can reason in pixel space.

Must expose:
```python
class Identity:                                    # no-op (control)
class HorizontalFlip(prob=0.5, flip_pairs=…):     # geometric; transforms labels
class RandomRotation(max_deg=10):                 # geometric; transforms labels
class RandomTranslate(max_px=5):                   # geometric; transforms labels
class RandomScale(low=0.9, high=1.1):             # geometric; transforms labels
class BrightnessContrast(b=0.2, c=0.2):           # photometric; labels unchanged
class GaussianNoise(sigma=0.01):                  # photometric; labels unchanged
class Cutout(size=16):                             # photometric; labels unchanged
class Compose([t1, t2, ...]):                     # sequential combinator
def build_augmenter(cfg: dict):                   # factory called from configs
```

**Geometric math** (students will implement these; the formulas are here so the check is mechanical):

For rotation by angle θ (radians) around image centre (48, 48):
- `x' = cos(θ)(x-48) - sin(θ)(y-48) + 48`
- `y' = sin(θ)(x-48) + cos(θ)(y-48) + 48`

For translation by (tx, ty):
- `x' = x + tx`, `y' = y + ty`

For scale by factor s around (48, 48):
- `x' = s(x-48) + 48`, `y' = s(y-48) + 48`

For horizontal flip:
- `x' = 95 - x`, `y' = y`, then **swap** symmetric pairs (see §4.3a).

Image-side transforms should use `torchvision.transforms.functional.affine` (rotation/translation/scale can all be expressed as one affine) or `torch.nn.functional.grid_sample`. Pad with zeros (fill=0).

### 4.3a Flip symmetric-pair table
```python
FLIP_PAIRS_B = [
    ("left_eye_center",        "right_eye_center"),
    ("left_eye_inner_corner",  "right_eye_inner_corner"),
    ("left_eye_outer_corner",  "right_eye_outer_corner"),
    ("left_eyebrow_inner_end", "right_eyebrow_inner_end"),
    ("left_eyebrow_outer_end", "right_eyebrow_outer_end"),
    ("mouth_left_corner",      "mouth_right_corner"),
]
FLIP_PAIRS_A = [("left_eye_center", "right_eye_center")]
```
Midline keypoints (`nose_tip`, `mouth_center_top_lip`, `mouth_center_bottom_lip`) do not swap; only their x flips.

### 4.4 `src/train.py` — training loop spec

Entry point: `uv run python -m src.train --config <path> [--override key=value ...]`

Flow:
```
1. load_config(); set_seed(cfg.seed); device = get_device(); run_dir = make_run_dir(cfg.run_name)
2. augmenter = build_augmenter(cfg.data.augmentation)              # None for Stage 1 & 2
3. train_ds = KeypointsDataset(cfg.data.train_csv, cols, three_channel, augmenter)
   val_ds   = KeypointsDataset(cfg.data.val_csv,   cols, three_channel, augmenter=None)
4. DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
              num_workers=2, pin_memory=True)
5. model = build_model(cfg.model).to(device)
   if cfg.model.freeze_backbone: freeze backbone params.
   if cfg.train.stage == "stage2": unfreeze last-N blocks, build 2 param groups
       [{params: head, lr: cfg.train.lr}, {params: backbone, lr: cfg.train.backbone_lr}]
6. criterion = nn.MSELoss() on NORMALIZED targets
   optimizer  = {adam: Adam, adamw: AdamW, sgd: SGD(momentum=0.9)}[cfg.train.optimizer]
   scheduler  = ReduceLROnPlateau(factor=0.5, patience=cfg.train.scheduler.patience)
7. loop up to cfg.train.epochs with early stopping on val_rmse_px:
      forward -> loss.backward() -> clip_grad_norm_(params, cfg.train.grad_clip) -> step()
      after val epoch: append {epoch, lr, train_loss, val_loss, train_rmse_px,
                               val_rmse_px, epoch_time_sec} to logs/{run}/metrics.csv
8. save checkpoints/{run}_best.pt and {run}_last.pt
9. append_results_row(cfg + final metrics)
```

**RMSE reported in pixel space:**
```python
rmse_px = torch.sqrt(((pred - target) ** 2).mean()) * 48.0
```

### 4.5 Fixed defaults by task family

These are applied automatically unless the config overrides them. Written so Stages 1–6 are comparable; Stage 7 explicitly relaxes them.

| Knob               | Baseline (MLP/CNN, Stages 1–3, 6.0) | Transfer (Stages 4, 5, 6.1) |
|--------------------|--------------------------------------|------------------------------|
| Epochs             | 100                                  | 30 per stage                 |
| Batch size         | 64                                   | 32                           |
| Optimizer          | Adam                                 | Adam                         |
| LR (head)          | 1e-3                                 | 1e-3                         |
| LR (backbone)      | —                                    | 1e-4 (stage2 only)           |
| Weight decay       | 0                                    | 1e-4                         |
| Scheduler          | ReduceLROnPlateau(f=0.5, p=5)        | same                         |
| Early stopping     | patience 15 on val_rmse_px           | same                         |
| Gradient clip (L2) | 1.0                                  | 1.0                          |

---

## 5. Config Schema

Every run is one YAML. Inheritance is handled in code via `load_config(base=…)` — **do not** copy-paste large configs.

```yaml
# example: configs/phase2/cnn_baseline.yaml
run_name: phase2_cnn_baseline
task_id: "2.0"
phase: 2
seed: 42

data:
  train_csv: data/dataset_a/dataset_a_train.csv
  val_csv:   data/dataset_a/dataset_a_val.csv
  test_csv:  data/dataset_a/dataset_a_test.csv
  coord_set: A                 # A -> COORD_COLS_A, B -> all coords
  three_channel: false
  num_workers: 2
  augmentation:
    compose: []                # empty = no augmentation

model:
  type: simple_cnn             # mlp | simple_cnn | squeezenet |
                               # mobilenet_v2 | mobilenet_v2_stem | mobilenet_v2_custom |
                               # efficientnet_b0
  output_dim: 8                # 8 for A, 30 for B
  dropout: 0.0
  # transfer-only:
  freeze_backbone: false
  unfreeze_last_n_blocks: 0

train:
  stage: single                # single | stage1 | stage2
  epochs: 100
  batch_size: 64
  optimizer: adam
  lr: 1.0e-3
  backbone_lr: 1.0e-4           # used only if stage == stage2
  weight_decay: 0.0
  grad_clip: 1.0
  scheduler: {type: reduce_on_plateau, factor: 0.5, patience: 5, min_lr: 1.0e-6}
  early_stopping: {patience: 15, monitor: val_rmse_px}
```

Two-stage transfer runs are **two YAMLs** sharing the same `run_name`, one with `stage: stage1`, one with `stage: stage2` that loads the stage1 checkpoint via `--override train.resume=checkpoints/{run_name}_stage1_best.pt`.

---

## 6. Stage 1 — MLP Baseline (Dataset A)

**Learning goal.** Confirm the whole pipeline works end-to-end on the simplest possible model. Establish the floor that every subsequent model must beat.

Model (`src/models/mlp.py`):
```
Flatten(96*96) → Linear(9216, 100) → ReLU → Linear(100, 8)
```

| | |
|---|---|
| **Prerequisites** | §4 shared code merged. `data/dataset_a/*.csv` exist. |
| **Config** | `configs/phase1/task1_0_mlp.yaml` (defaults in §4.5 baseline column, no aug) |
| **Command** | `uv run python -m src.train --config configs/phase1/task1_0_mlp.yaml` |
| **Outputs** | `logs/phase1_mlp/{config.yaml, metrics.csv}`, `checkpoints/phase1_mlp_{best,last}.pt`, one row in `results/results.csv`. |
| **Success criteria** | `val_rmse_px` < 4.0 (sanity-check number on this dataset; if you see ≥ 8 something is wrong with parsing/normalization). Val loss curve monotonically trending down for ~20 epochs. |
| **Handoff** | Baseline recorded. Do not tune. Move to Stage 2. |

---

## 7. Stage 2 — CNN Baseline (Dataset A)

**Learning goal.** Observe the spatial-inductive-bias payoff. Two conv+pool blocks should significantly beat the MLP at similar or lower parameter count. Students should be able to articulate *why*.

Model (`src/models/simple_cnn.py`):
```
Conv2d(1, 32, k=3, pad=1) → ReLU → MaxPool2d(2)
Conv2d(32, 64, k=3, pad=1) → ReLU → MaxPool2d(2)
Flatten → Linear(64*24*24, 128) → ReLU → Linear(128, 8)
```

| | |
|---|---|
| **Prerequisites** | Stage 1 done. |
| **Config** | `configs/phase2/task2_0_cnn.yaml` (baseline defaults, no aug) |
| **Command** | `uv run python -m src.train --config configs/phase2/task2_0_cnn.yaml` |
| **Outputs** | `logs/phase2_cnn/*`, `checkpoints/phase2_cnn_*`, one row in `results/results.csv`. |
| **Success criteria** | `val_rmse_px` noticeably lower than Stage 1's. Gap between train and val RMSE begins to appear — this is the overfitting you will regularize against in Stage 3. |
| **Handoff** | `cnn_baseline = val_rmse_px from this run`. Used as the reference in Stage 3. |

---

## 8. Stage 3 — Data Augmentation Ablation (Dataset A) ⭐

**Learning goal.** Teach that (a) augmentation is free regularization that often beats architectural changes, and (b) *geometric* augs are non-trivial for keypoint regression because labels must transform consistently with pixels. Build intuition via a proper ablation — one knob at a time, then combined.

Architecture frozen: same `simple_cnn` as Stage 2. Only `data.augmentation` changes. Every run uses the baseline hyperparameters (§4.5) so differences are attributable to the augmentation alone.

### 8.1 Ablation grid

8 configs under `configs/phase3_augmentation/`:

| id   | config file                  | `data.augmentation.compose`                                             | transforms labels? |
|------|------------------------------|--------------------------------------------------------------------------|--------------------|
| A0   | `a0_none.yaml`               | `[]`                                                                     | —                  |
| A1   | `a1_flip.yaml`               | `[HorizontalFlip(p=0.5)]`                                                | yes                |
| A2   | `a2_rotate.yaml`             | `[RandomRotation(max_deg=10)]`                                           | yes                |
| A3   | `a3_translate.yaml`          | `[RandomTranslate(max_px=5)]`                                            | yes                |
| A4   | `a4_scale.yaml`              | `[RandomScale(low=0.9, high=1.1)]`                                       | yes                |
| A5   | `a5_brightness_contrast.yaml`| `[BrightnessContrast(b=0.2, c=0.2)]`                                     | no                 |
| A6   | `a6_noise.yaml`              | `[GaussianNoise(sigma=0.01)]`                                            | no                 |
| A7   | `a7_cutout.yaml`             | `[Cutout(size=16)]`                                                      | no                 |
| A8   | `a8_geom_combo.yaml`         | `[HorizontalFlip, RandomRotation, RandomTranslate, RandomScale]`         | yes                |
| A9   | `a9_full_combo.yaml`         | `A8 + BrightnessContrast + GaussianNoise + Cutout`                       | mixed              |

`A0` duplicates Stage 2 but is repeated here so the whole ablation is one coherent block of rows in `results.csv`.

### 8.2 Pre-flight visual check

**Before training any A* config, add a `--dry-run` flag to `src/train.py` that pulls one batch, applies the augmenter, and writes `reports/aug_{id}_preview.png` — 8 images with keypoints overlaid.** Red dots must still sit on facial landmarks after augmentation. If they don't, the geometric math is wrong and you fix it *before* training anything. This saves days.

```bash
uv run python -m src.train --config configs/phase3_augmentation/a2_rotate.yaml --dry-run
# inspect reports/aug_a2_rotate_preview.png — labels must stay on the face
```

### 8.3 Train

```bash
for c in configs/phase3_augmentation/*.yaml; do
    uv run python -m src.train --config "$c"
done
```

| | |
|---|---|
| **Prerequisites** | Stage 2 done. Dry-run verified for every geometric aug. |
| **Outputs** | 10 rows in `results/results.csv` (one per A0–A9), 10 preview PNGs in `reports/`. |
| **Success criteria** | `A1 (flip)` should beat `A0` by a meaningful margin. At least one of `A8`/`A9` should be the overall best. If a single-aug run is *worse* than `A0` by a large margin, investigate — usually a label-transform bug. |
| **Handoff** | Write a short ablation table into `reports/stage3_augmentation.md` (2–3 sentences per row explaining the delta). Record `aug_winner = argmin_val_rmse(A*)` and the winning augmentation recipe — it will be reused in Stages 4–6 as the default aug. |

---

## 9. Stage 4 — Transfer Learning (Dataset A)

**Learning goal.** Pretrained ImageNet features transfer surprisingly well even for grayscale keypoints. Two-stage fine-tune teaches the mechanics: freeze → train head → unfreeze last blocks → small LR.

All configs: `data.three_channel: true` (grayscale duplicated to 3 channels). Default augmentation = `aug_winner` from Stage 3.

### 9.1 Tasks

| id   | config file                                         | model                | training                                                 |
|------|-----------------------------------------------------|----------------------|----------------------------------------------------------|
| 4.1  | `squeezenet.yaml`                                   | `squeezenet`         | single-stage, head only (`freeze_backbone: true`), 30 ep |
| 4.2a | `mobilenet_v2_stage1.yaml`                          | `mobilenet_v2`       | stage1 (freeze features), 30 ep                          |
| 4.2b | `mobilenet_v2_stage2.yaml`                          | `mobilenet_v2`       | stage2 (unfreeze last 2 blocks), 30 ep                   |
| 4.3a | `efficientnet_b0_stage1.yaml`                       | `efficientnet_b0`    | stage1 (freeze features), 30 ep                          |
| 4.3b | `efficientnet_b0_stage2.yaml`                       | `efficientnet_b0`    | stage2 (unfreeze last 2 blocks), 30 ep                   |

### 9.2 Model builders (`src/models/transfer.py`)

```python
# SqueezeNet
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
m = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
m.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Conv2d(512, OUT, kernel_size=1),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
)

# MobileNetV2
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
m = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, OUT))

# EfficientNet-B0
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, OUT))
```

| | |
|---|---|
| **Prerequisites** | Stage 3 done (`aug_winner` known). |
| **Commands** | Run each YAML with `uv run python -m src.train --config <path>`. For two-stage runs, launch stage1 first; stage2 auto-loads stage1's `_best.pt` via `train.resume`. |
| **Outputs** | 5 rows in `results/results.csv`. |
| **Success criteria** | The best transfer model should beat `aug_winner`. If it doesn't, likely cause: augmentation is too aggressive on 3-channel inputs (try turning off photometric aug first, since pretrained models expect cleaner inputs). |
| **Handoff** | Record `transfer_winner`. |

---

## 10. Stage 5 — Architectural Tweaks (Dataset A)

**Learning goal.** Show that targeted architectural changes can outperform naive transfer learning. Each tweak is motivated by a specific weakness of the prior model.

Both tasks start from **fresh** ImageNet-pretrained MobileNetV2 weights (not Stage 4.2b's fine-tuned checkpoint). Two-stage recipe same as Stage 4.

### 10.1 Task 5.1 — Stem modification

Motivation: MobileNetV2's first conv has `stride=2`, which discards half the spatial resolution in the very first layer. For 96×96 inputs with tiny features (eye corners are ~4 px), we cannot afford that.

Change: `m.features[0][0]` — replace with `Conv2d(3, 32, k=3, stride=1, pad=1)` and copy the original kernel weights (shape-compatible).

Configs: `configs/phase5_arch_tweaks/task5_1_stem_{stage1,stage2}.yaml`. Model type: `mobilenet_v2_stem`.

### 10.2 Task 5.2 — Custom regression head

Build on Task 5.1 (stem already stride=1). Replace the classifier with:
```python
nn.Sequential(
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    nn.Linear(1280, 256), nn.BatchNorm1d(256), nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, OUT),
)
```
Provide a `MobileNetV2Custom(nn.Module)` wrapper in `src/models/transfer.py` whose `forward` is `features(x) → head(x)` (bypass MobileNet's built-in classifier entirely).

Configs: `configs/phase5_arch_tweaks/task5_2_custom_{stage1,stage2}.yaml`. Model type: `mobilenet_v2_custom`.

| | |
|---|---|
| **Prerequisites** | Stage 4 done. |
| **Commands** | 4 YAMLs (2 tasks × 2 stages). |
| **Outputs** | 4 rows in `results/results.csv`. |
| **Success criteria** | Task 5.2 should beat Task 5.1, and Task 5.1 should beat Stage 4.2b. Stage winner called `arch_winner`. |

---

## 11. Stage 6 — Full 15-Keypoint Output (Dataset B)

**Learning goal.** Observe the cost of label coverage: Dataset B has only 2,140 rows and 30 outputs. Overfitting risk is high. Students should notice the train/val gap widen and plan accordingly (stronger aug, earlier stopping).

### 11.1 Task 6.0 — Simple CNN, full output

Clone Stage 2 architecture with final `Linear(128, 30)`. Train on `data/dataset_b/*`. Default aug = `aug_winner` from Stage 3 but with **`FLIP_PAIRS_B`** (not A). If your augmenter uses `coord_set` to pick pairs this is automatic.

### 11.2 Task 6.1 — `arch_winner` with full output

Clone `arch_winner` architecture, change final `Linear(256, 8)` → `Linear(256, 30)`, rerun two-stage recipe on Dataset B.

| | |
|---|---|
| **Prerequisites** | Stage 5 done. |
| **Commands** | `configs/phase6_full_output/task6_0_cnn_full.yaml`, `…/task6_1_best_full_stage{1,2}.yaml`. |
| **Outputs** | 3 rows in `results/results.csv`. |
| **Success criteria** | Per-coord RMSE_px reported in `results/results.csv` note column (see §13.2). `phase6_winner = argmin_val_rmse`. |

---

## 12. Stage 7 — Hyperparameter Tuning

**Learning goal.** All comparisons so far held hyperparameters fixed so architectural differences were interpretable. Now we actually tune. Students learn (1) random search beats grid at equal budget, (2) a single "best trial" is unreliable, (3) stability across seeds matters.

### 12.1 Pick the overall winner to tune

```
overall_winner = argmin_val_rmse({cnn_baseline, aug_winner, transfer_winner, arch_winner, phase6_winner})
```
Its config becomes the `base_config` in the search.

### 12.2 Search config (`configs/phase7_tuning/tune_best.yaml`)

```yaml
base_config: configs/phase5_arch_tweaks/task5_2_custom_stage2.yaml   # or whichever is overall_winner
run_name_prefix: phase7_tune
n_trials: 30
sampler: random
space:
  train.optimizer:              {choice: [adam, adamw, sgd]}
  train.lr:                     {log_uniform: [1.0e-5, 5.0e-3]}
  train.backbone_lr:            {log_uniform: [1.0e-6, 1.0e-3]}
  train.weight_decay:           {log_uniform: [1.0e-6, 1.0e-3]}
  train.batch_size:             {choice: [16, 32, 64]}
  model.dropout:                {uniform: [0.1, 0.5]}
  model.unfreeze_last_n_blocks: {choice: [1, 2, 3, 4]}
  data.augmentation.flip_prob:  {choice: [0.0, 0.3, 0.5, 0.8]}
  train.scheduler.patience:     {choice: [3, 5, 8]}
```

Budget-shrink heuristic: for tuning trials, set `train.epochs: 25` (single-stage) or `15+15` (two-stage) so 30 trials fit in ~2–4 hours on an M3 Pro. Full-length re-runs happen only for the top-3 candidates.

### 12.3 `src/tune.py` behavior

```
For trial_idx in range(n_trials):
  rng = np.random.default_rng(12345 + trial_idx)
  sampled = sample_space(search.space, rng)
  cfg = deep_merge(load_config(base_config), sampled)
  cfg.run_name = f"{run_name_prefix}_{trial_idx:03d}"
  cfg.seed = 42 + trial_idx
  save resolved cfg to configs/phase7_tuning/_trials/{cfg.run_name}.yaml
  src.train.main(cfg)
  append (trial_idx, sampled values, train/val rmse, epoch_time)
         to results/tune_results.csv
print top-5 rows sorted by val_rmse_px.
```

Parallelise by launching multiple processes with `--start i --end j` (disjoint index ranges).

### 12.4 Stability re-run

Take the top-3 trials by val_rmse_px. Rerun each at **full** epoch budget with seeds `{42, 43, 44}`. Promote the candidate whose three-seed **mean** is best and whose **stddev** is reasonable (rule of thumb: stddev < 10% of mean) to `tuned_winner`.

### 12.5 Optional — 5-fold CV of `tuned_winner`

Split Dataset A/B **train+val** (keep test untouched) with `KFold(n_splits=5, shuffle=True, random_state=42)`. Train 5 models, report mean ± stddev val_rmse_px. This is the headline number in the final report.

| | |
|---|---|
| **Prerequisites** | All of Stages 1–6 done. `results/results.csv` has ≥ 20 rows. |
| **Commands** | `uv run python -m src.tune --config configs/phase7_tuning/tune_best.yaml` then stability runs. |
| **Outputs** | `results/tune_results.csv` (30 rows) + `configs/phase7_tuning/_trials/*.yaml`, plus 3×3 stability rows in `results/results.csv`. |
| **Success criteria** | Best-tuned val_rmse_px is ≤ `overall_winner`'s. If not, tuning failed (search space too narrow, or base already well-tuned). |

---

## 13. Stage 8 — Error Analysis & Final Test Evaluation

**Learning goal.** Numbers in a table are not a finding. Students have to look at the model's predictions, identify failure modes, and describe them in words.

### 13.1 Test-set evaluation

Only now do we touch the test CSVs. For each of the following, run
```bash
uv run python -m src.evaluate \
  --run {run_name} \
  --checkpoint checkpoints/{run_name}_best.pt \
  --split test
```
Runs to evaluate:
- Stage 1 `phase1_mlp` — Dataset A test
- Stage 2 `phase2_cnn` — Dataset A test
- Stage 3 `aug_winner` — Dataset A test
- Stage 4 `transfer_winner` — Dataset A test
- Stage 5 `arch_winner` — Dataset A test
- Stage 6 `task6_0` and `task6_1` — Dataset B test
- Stage 7 `tuned_winner` — A or B depending on base

`src/evaluate.py` fills the `test_rmse_px` column in the matching `results/results.csv` row and additionally writes `results/predictions_{run_name}.csv` (one row per test sample: image index, per-coord pred, per-coord target, per-coord error).

### 13.2 Error analysis (`src/visualize_predictions.py`)

For `tuned_winner` (and any other model the team wants to contrast), produce:

1. **Per-keypoint RMSE bar chart** — `reports/{run}_per_keypoint_rmse.png`. Which keypoints are hardest? The answer is usually mouth-corner and eyebrow-outer.
2. **Error distribution histogram** — `reports/{run}_error_hist.png`. Shows whether errors are gaussian-ish or long-tailed.
3. **Worst 12 predictions** — `reports/{run}_worst12.png`. 3×4 grid. Ground truth green, prediction red. Label each panel with total RMSE_px.
4. **Best 12 predictions** — `reports/{run}_best12.png`. Same format. Shows what the model handles well.
5. **Residual heatmap** — `reports/{run}_residual_heatmap.png`. Scatter of (pred_x - gt_x, pred_y - gt_y) for every keypoint of every test sample. Look for systematic bias (off-center cloud).

| | |
|---|---|
| **Prerequisites** | Stage 7 complete. All test runs in §13.1 done. |
| **Outputs** | Test RMSE numbers in `results/results.csv`, 5 plots per analysed model in `reports/`. |
| **Success criteria** | Team can name the 2 worst keypoints and show a qualitative example of the dominant failure mode (usually: profile-angled face, partial occlusion, or non-frontal lighting). |

---

## 14. Reporting

Final deliverables in `reports/`:

1. `reports/stage3_augmentation.md` — ablation table written during Stage 3.
2. `reports/main_results_table.md` — one table of val & test RMSE_px for Stages 1–7 winners.
3. `reports/error_analysis.md` — 1–2 pages: per-keypoint RMSE, worst-case gallery, 3-sentence discussion of failure modes.
4. `reports/plan_v3_final.md` — a copy of this plan with the actual winners and numbers filled in at the end of each stage, as evidence that the plan was followed.

---

## 15. Quick-Start Commands (fresh clone)

```bash
# Stage 0 — data
uv sync
uv run python data/explore_data.py
uv run python data/visualize_samples.py
uv run python data/split_data.py

# Stage 1 — MLP baseline
uv run python -m src.train --config configs/phase1/task1_0_mlp.yaml

# Stage 2 — CNN baseline
uv run python -m src.train --config configs/phase2/task2_0_cnn.yaml

# Stage 3 — augmentation ablation (dry-run every geometric aug first!)
for c in configs/phase3_augmentation/*.yaml; do
    uv run python -m src.train --config "$c" --dry-run   # preview
done
for c in configs/phase3_augmentation/*.yaml; do
    uv run python -m src.train --config "$c"             # train
done

# Stage 4 — transfer learning
uv run python -m src.train --config configs/phase4_transfer/squeezenet.yaml
uv run python -m src.train --config configs/phase4_transfer/mobilenet_v2_stage1.yaml
uv run python -m src.train --config configs/phase4_transfer/mobilenet_v2_stage2.yaml
uv run python -m src.train --config configs/phase4_transfer/efficientnet_b0_stage1.yaml
uv run python -m src.train --config configs/phase4_transfer/efficientnet_b0_stage2.yaml

# Stage 5 — architectural tweaks
uv run python -m src.train --config configs/phase5_arch_tweaks/task5_1_stem_stage1.yaml
uv run python -m src.train --config configs/phase5_arch_tweaks/task5_1_stem_stage2.yaml
uv run python -m src.train --config configs/phase5_arch_tweaks/task5_2_custom_stage1.yaml
uv run python -m src.train --config configs/phase5_arch_tweaks/task5_2_custom_stage2.yaml

# Stage 6 — Dataset B, 30-keypoint output
uv run python -m src.train --config configs/phase6_full_output/task6_0_cnn_full.yaml
uv run python -m src.train --config configs/phase6_full_output/task6_1_best_full_stage1.yaml
uv run python -m src.train --config configs/phase6_full_output/task6_1_best_full_stage2.yaml

# Stage 7 — tuning
uv run python -m src.tune --config configs/phase7_tuning/tune_best.yaml
# + stability re-runs of top-3 trials at full epoch budget

# Stage 8 — error analysis & test eval
# (one --run per winner from earlier stages)
uv run python -m src.evaluate --run arch_winner --split test
uv run python -m src.visualize_predictions --run tuned_winner

# Final: commit results/*.csv, reports/*, and this plan.md with winners filled in.
```

---

## 16. Division of Labour (suggested for 3 people)

| Person | Stages | Shared review responsibility       |
|--------|--------|------------------------------------|
| A      | §4 shared code, Stage 1, Stage 2, Stage 6.0 | reviews B's and C's configs       |
| B      | Stage 3 augmentation ablation, Stage 4 transfer learning | reviews A's trainer & C's tuning |
| C      | Stage 5 tweaks, Stage 6.1, Stage 7 tuning, Stage 8 error analysis, writeup | reviews A's shared code           |

The shared code in §4 must be written, reviewed, and merged **before** anyone starts Stage 1. Forking your own trainer is the single fastest way to get non-comparable numbers; it is forbidden.
