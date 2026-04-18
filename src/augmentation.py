"""Keypoint-aware augmentations.

Every transform is a callable (img, y_px) -> (img, y_px):
    - img: torch.Tensor, shape (C, 96, 96), float32, values in [0, 1].
    - y_px: torch.Tensor, length K, PIXEL-space coordinates (NOT normalised).

Geometric transforms (flip, rotate, translate, scale) rewrite labels.
Photometric transforms (brightness/contrast, noise, cutout) leave labels alone.

Configs declare augmentation in a flat schema so tuning can override each knob
by a single dotted key:

    data:
      augmentation:
        flip_prob: 0.5         # 0 disables HorizontalFlip
        rotation_deg: 10       # 0 disables RandomRotation
        translate_px: 5        # 0 disables RandomTranslate
        scale_low: 0.9         # scale_low >= scale_high disables RandomScale
        scale_high: 1.1
        brightness: 0.2
        contrast: 0.2
        noise_sigma: 0.01
        cutout_size: 16
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from src.dataset import CENTER, IMG_SIZE


# ---------------------------------------------------------------------------- helpers


def _affine_warp(img: torch.Tensor, theta_2x3: torch.Tensor) -> torch.Tensor:
    """Apply a 2x3 sampling affine to (C, H, W). Pads with zero.

    NOTE on `align_corners`: we use align_corners=False, so grid corners map to
    half-pixel outside the image. The normalised origin (0, 0) corresponds to
    pixel (47.5, 47.5) for a 96x96 image, while our keypoint normalisation
    uses CENTER = 48. The 0.5-pixel mismatch is negligible for <= 10 deg
    rotations at this resolution. If you make augmentation more aggressive,
    revisit this.
    """
    C, H, W = img.shape
    theta = theta_2x3.unsqueeze(0)
    grid = F.affine_grid(theta, [1, C, H, W], align_corners=False)
    out = F.grid_sample(
        img.unsqueeze(0), grid,
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )
    return out.squeeze(0)


# ---------------------------------------------------------------------------- geometric


class HorizontalFlip:
    """Pixel-mirror plus swap of symmetric keypoint pairs.

    `flip_pairs` is a list of (name_a, name_b). For every pair where both
    name_a_{x,y} and name_b_{x,y} are present in coord_cols, the two
    indices are swapped after x-coord mirroring.
    """
    def __init__(
        self,
        coord_cols: Sequence[str],
        flip_pairs: Sequence[tuple[str, str]],
        prob: float = 0.5,
    ):
        self.prob = prob
        self.coord_cols = list(coord_cols)
        self._x_idx = [i for i, c in enumerate(self.coord_cols) if c.endswith("_x")]
        self._swaps: list[tuple[int, int]] = []
        for a, b in flip_pairs:
            for suf in ("_x", "_y"):
                ca, cb = a + suf, b + suf
                if ca in self.coord_cols and cb in self.coord_cols:
                    self._swaps.append((self.coord_cols.index(ca),
                                        self.coord_cols.index(cb)))

    def __call__(self, img, y_px):
        if torch.rand(1).item() >= self.prob:
            return img, y_px
        img = img.flip(-1)
        y = y_px.clone()
        for i in self._x_idx:
            y[i] = (IMG_SIZE - 1) - y[i]
        for ia, ib in self._swaps:
            tmp = y[ia].clone()
            y[ia] = y[ib]
            y[ib] = tmp
        return img, y


class RandomRotation:
    def __init__(self, max_deg: float = 10.0, prob: float = 1.0):
        self.max_deg = float(max_deg); self.prob = prob

    def __call__(self, img, y_px):
        if torch.rand(1).item() >= self.prob:
            return img, y_px
        deg = (torch.rand(1).item() * 2 - 1) * self.max_deg
        theta = math.radians(deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        # Sampling grid rotates BACKWARDS of label rotation.
        M = torch.tensor(
            [[cos_t, sin_t, 0.0],
             [-sin_t, cos_t, 0.0]], dtype=img.dtype,
        )
        img = _affine_warp(img, M)
        y = y_px.clone().reshape(-1, 2)
        x = y[:, 0] - CENTER; v = y[:, 1] - CENTER
        y[:, 0] = cos_t * x - sin_t * v + CENTER
        y[:, 1] = sin_t * x + cos_t * v + CENTER
        return img, y.reshape(-1)


class RandomTranslate:
    def __init__(self, max_px: int = 5, prob: float = 1.0):
        self.max_px = int(max_px); self.prob = prob

    def __call__(self, img, y_px):
        if torch.rand(1).item() >= self.prob or self.max_px <= 0:
            return img, y_px
        tx = (torch.rand(1).item() * 2 - 1) * self.max_px
        ty = (torch.rand(1).item() * 2 - 1) * self.max_px
        # Sampling grid shift is opposite sign, normalised by W/2, H/2.
        M = torch.tensor(
            [[1.0, 0.0, -2.0 * tx / IMG_SIZE],
             [0.0, 1.0, -2.0 * ty / IMG_SIZE]], dtype=img.dtype,
        )
        img = _affine_warp(img, M)
        y = y_px.clone().reshape(-1, 2)
        y[:, 0] += tx; y[:, 1] += ty
        return img, y.reshape(-1)


class RandomScale:
    def __init__(self, low: float = 0.9, high: float = 1.1, prob: float = 1.0):
        self.low = float(low); self.high = float(high); self.prob = prob

    def __call__(self, img, y_px):
        if torch.rand(1).item() >= self.prob or self.low >= self.high:
            return img, y_px
        s = self.low + torch.rand(1).item() * (self.high - self.low)
        inv = 1.0 / s
        M = torch.tensor(
            [[inv, 0.0, 0.0],
             [0.0, inv, 0.0]], dtype=img.dtype,
        )
        img = _affine_warp(img, M)
        y = y_px.clone().reshape(-1, 2)
        y[:, 0] = (y[:, 0] - CENTER) * s + CENTER
        y[:, 1] = (y[:, 1] - CENTER) * s + CENTER
        return img, y.reshape(-1)


# ---------------------------------------------------------------------------- photometric


class BrightnessContrast:
    def __init__(self, b: float = 0.2, c: float = 0.2, prob: float = 1.0):
        self.b = float(b); self.c = float(c); self.prob = prob

    def __call__(self, img, y_px):
        if torch.rand(1).item() >= self.prob:
            return img, y_px
        contrast = 1.0 + (torch.rand(1).item() * 2 - 1) * self.c
        brightness = (torch.rand(1).item() * 2 - 1) * self.b
        return (img * contrast + brightness).clamp(0.0, 1.0), y_px


class GaussianNoise:
    def __init__(self, sigma: float = 0.01, prob: float = 1.0):
        self.sigma = float(sigma); self.prob = prob

    def __call__(self, img, y_px):
        if torch.rand(1).item() >= self.prob or self.sigma <= 0:
            return img, y_px
        return (img + torch.randn_like(img) * self.sigma).clamp(0.0, 1.0), y_px


class Cutout:
    def __init__(self, size: int = 16, prob: float = 1.0):
        self.size = int(size); self.prob = prob

    def __call__(self, img, y_px):
        if torch.rand(1).item() >= self.prob or self.size <= 0:
            return img, y_px
        _, H, W = img.shape
        cx = int(torch.randint(0, W, (1,)).item())
        cy = int(torch.randint(0, H, (1,)).item())
        half = self.size // 2
        x0 = max(0, cx - half); x1 = min(W, cx + half)
        y0 = max(0, cy - half); y1 = min(H, cy + half)
        img = img.clone()
        img[:, y0:y1, x0:x1] = 0.0
        return img, y_px


# ---------------------------------------------------------------------------- compose + factory


class Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)
    def __call__(self, img, y_px):
        for t in self.transforms:
            img, y_px = t(img, y_px)
        return img, y_px


FLIP_PAIRS_A: list[tuple[str, str]] = [("left_eye_center", "right_eye_center")]
FLIP_PAIRS_B: list[tuple[str, str]] = [
    ("left_eye_center",        "right_eye_center"),
    ("left_eye_inner_corner",  "right_eye_inner_corner"),
    ("left_eye_outer_corner",  "right_eye_outer_corner"),
    ("left_eyebrow_inner_end", "right_eyebrow_inner_end"),
    ("left_eyebrow_outer_end", "right_eyebrow_outer_end"),
    ("mouth_left_corner",      "mouth_right_corner"),
]


def flip_pairs_for(coord_set: str) -> list[tuple[str, str]]:
    return FLIP_PAIRS_A if coord_set == "A" else FLIP_PAIRS_B


def build_augmenter(
    aug_cfg: dict | None,
    coord_cols: Sequence[str],
    coord_set: str,
) -> Callable | None:
    """Return a Compose(...) reading the flat augmentation schema, or None."""
    if not aug_cfg:
        return None
    transforms: list[Callable] = []

    flip_prob = float(aug_cfg.get("flip_prob", 0.0))
    if flip_prob > 0:
        transforms.append(HorizontalFlip(
            coord_cols=coord_cols,
            flip_pairs=flip_pairs_for(coord_set),
            prob=flip_prob,
        ))

    rot = float(aug_cfg.get("rotation_deg", 0.0))
    if rot > 0:
        transforms.append(RandomRotation(max_deg=rot))

    trans = int(aug_cfg.get("translate_px", 0))
    if trans > 0:
        transforms.append(RandomTranslate(max_px=trans))

    lo = float(aug_cfg.get("scale_low", 1.0))
    hi = float(aug_cfg.get("scale_high", 1.0))
    if lo < hi:
        transforms.append(RandomScale(low=lo, high=hi))

    b = float(aug_cfg.get("brightness", 0.0))
    c = float(aug_cfg.get("contrast", 0.0))
    if b > 0 or c > 0:
        transforms.append(BrightnessContrast(b=b, c=c))

    sigma = float(aug_cfg.get("noise_sigma", 0.0))
    if sigma > 0:
        transforms.append(GaussianNoise(sigma=sigma))

    cutout = int(aug_cfg.get("cutout_size", 0))
    if cutout > 0:
        transforms.append(Cutout(size=cutout))

    return Compose(transforms) if transforms else None


def summarise_augmentation(aug_cfg: dict | None) -> str:
    """Short '+'-joined tag for results.csv."""
    if not aug_cfg: return "none"
    parts: list[str] = []
    if float(aug_cfg.get("flip_prob", 0)) > 0: parts.append("flip")
    if float(aug_cfg.get("rotation_deg", 0)) > 0: parts.append("rot")
    if int(aug_cfg.get("translate_px", 0)) > 0: parts.append("trans")
    if float(aug_cfg.get("scale_low", 1.0)) < float(aug_cfg.get("scale_high", 1.0)): parts.append("scale")
    if float(aug_cfg.get("brightness", 0)) > 0 or float(aug_cfg.get("contrast", 0)) > 0: parts.append("bc")
    if float(aug_cfg.get("noise_sigma", 0)) > 0: parts.append("noise")
    if int(aug_cfg.get("cutout_size", 0)) > 0: parts.append("cutout")
    return "+".join(parts) or "none"
