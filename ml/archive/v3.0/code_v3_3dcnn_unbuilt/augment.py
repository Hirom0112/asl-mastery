"""Pixel-level augmentation for the v3.x raw-RGB pipeline.

Per docs/MODEL.md §3 and docs/decisions/0005-classical-cv-allowed.md
(load-bearing again under ADR 0010). Applied training-time only,
inside the dataset loader, after MP4 decode and before tensor
normalization.

All augmentations are hand-coded over pixel arrays — no pretrained
component, no learned parameters. Classical CV (MOG2 background
swap) is permitted under ADR 0005's training-time-only boundary.

Operates on uint8 ndarrays of shape (T, 256, 256, 3) in RGB and
returns uint8 ndarrays of shape (T, H, W, 3). Composition of
augmentations is done by the `make_train_augment()` entry point.

The keypoint-level augmentations (coordinate jitter, temporal
stretch, keypoint dropout, etc.) that shipped under the superseded
ADR 0006 are removed — they have no analogue on raw RGB input.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # cv2 is in training/requirements.txt
    cv2 = None  # type: ignore


# -- 1. Random spatial crop -----------------------------------------


def random_spatial_crop(
    frames: np.ndarray, *, height: int, width: int, jitter_px: int = 8
) -> np.ndarray:
    """Crop a random (height, width) window from the source frames.

    All frames in the clip get the same crop offset (the model is
    learning a sign, not a tracking shot).
    """
    T, H, W, _ = frames.shape
    if H < height or W < width:
        raise ValueError(f"source frame {H}x{W} smaller than crop {height}x{width}")
    base_top = (H - height) // 2
    base_left = (W - width) // 2
    j = jitter_px
    top = max(0, min(H - height, base_top + random.randint(-j, j)))
    left = max(0, min(W - width, base_left + random.randint(-j, j)))
    return frames[:, top : top + height, left : left + width, :]


# -- 2. Color jitter / brightness / contrast ------------------------


def color_jitter(
    frames: np.ndarray,
    *,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
) -> np.ndarray:
    """Apply random brightness/contrast/saturation in HSV space.

    Returns uint8. All frames share the same multiplier so the clip
    looks like one continuous shot under one lighting condition.
    """
    if cv2 is None:
        return frames

    b_mult = 1.0 + random.uniform(-brightness, brightness)
    c_mult = 1.0 + random.uniform(-contrast, contrast)
    s_mult = 1.0 + random.uniform(-saturation, saturation)

    out = np.empty_like(frames)
    for t in range(frames.shape[0]):
        rgb = frames[t]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= s_mult  # saturation
        hsv[..., 2] *= b_mult  # brightness/value
        mean = hsv[..., 2].mean()
        hsv[..., 2] = (hsv[..., 2] - mean) * c_mult + mean
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
        out[t] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out


# -- 3. MOG2 background swap (classical CV under ADR 0005) ----------


def mog2_background_swap(
    frames: np.ndarray,
    *,
    bank: list[np.ndarray] | None = None,
    apply_prob: float = 0.5,
) -> np.ndarray:
    """Replace the source clip's background with one drawn from a bank.

    Steps:
      1. Run MOG2 over the clip to estimate the per-frame foreground
         mask. The signer is the moving foreground; the static parts
         of the frame are background.
      2. Pick a replacement background image (resized to frame shape)
         at random from `bank`. If `bank` is empty/None or `apply_prob`
         doesn't fire, return frames unchanged.
      3. Composite: `out = mask * foreground + (1 - mask) * bg`.

    The defense this provides is against the dorm-room overfitting
    failure mode that the superseded ADR 0001 named explicitly — under
    raw-RGB Path B the classifier can see the wall behind the learner,
    and we don't want it learning that the wall correlates with the
    sign.
    """
    if cv2 is None or bank is None or len(bank) == 0:
        return frames
    if random.random() > apply_prob:
        return frames

    T, H, W, _ = frames.shape
    bg = bank[random.randrange(len(bank))]
    if bg.shape[:2] != (H, W):
        bg = cv2.resize(bg, (W, H))

    sub = cv2.createBackgroundSubtractorMOG2(history=T, varThreshold=24, detectShadows=False)
    out = np.empty_like(frames)
    for t in range(T):
        gray_mask = sub.apply(cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR))
        gray_mask = cv2.dilate(gray_mask, np.ones((3, 3), np.uint8))
        gray_mask = cv2.GaussianBlur(gray_mask, (5, 5), 0)
        alpha = (gray_mask.astype(np.float32) / 255.0)[..., None]  # (H, W, 1)
        composited = alpha * frames[t].astype(np.float32) + (1 - alpha) * bg.astype(np.float32)
        out[t] = np.clip(composited, 0, 255).astype(np.uint8)
    return out


def load_background_bank(bank_dir: str | Path) -> list[np.ndarray]:
    """Read all .jpg/.png/.webp images from a directory into a bank.

    Empty / missing directory returns an empty list — the background
    swap then becomes a no-op. The bank is intended to be a small set
    of plain / textured / cluttered backgrounds the augmenter
    randomizes over.
    """
    if cv2 is None:
        return []
    path = Path(bank_dir)
    if not path.is_dir():
        return []
    out: list[np.ndarray] = []
    for p in sorted(path.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return out


# -- 4. Small affine (rotate / scale / translate) -------------------


def small_affine(
    frames: np.ndarray,
    *,
    max_deg: float = 10.0,
    max_scale: float = 0.05,
    max_translate: float = 0.03,
) -> np.ndarray:
    """Apply the same small affine transform to every frame in the clip."""
    if cv2 is None:
        return frames
    T, H, W, _ = frames.shape
    angle = random.uniform(-max_deg, max_deg)
    scale = 1.0 + random.uniform(-max_scale, max_scale)
    tx = random.uniform(-max_translate, max_translate) * W
    ty = random.uniform(-max_translate, max_translate) * H

    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    out = np.empty_like(frames)
    for t in range(T):
        out[t] = cv2.warpAffine(frames[t], M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    return out


# -- 5. Conditional horizontal flip ---------------------------------


def horizontal_flip(frames: np.ndarray, *, flippable: bool, prob: float = 0.5) -> np.ndarray:
    """Flip horizontally with given probability, only when the sign is
    `flippable: true` in docs/VOCABULARY.md.

    Two-handed asymmetric signs and signs whose handedness encodes
    meaning are never flipped.
    """
    if not flippable or random.random() > prob:
        return frames
    return frames[:, :, ::-1, :].copy()


# -- Composition entry point ----------------------------------------


def make_train_augment(
    *,
    input_height: int = 96,
    input_width: int = 96,
    background_bank: list[np.ndarray] | None = None,
    apply_bg_swap_prob: float = 0.5,
    flippable_lookup: dict[str, bool] | None = None,
) -> Callable[[np.ndarray, str], np.ndarray]:
    """Build the training-time augmentation callable.

    Returns a closure that takes `(frames_u8, sign_id)` and returns
    the augmented `(input_height, input_width)` frame stack. Sign-id
    drives the conditional horizontal flip via `flippable_lookup`.
    """

    def apply(frames: np.ndarray, sign_id: str) -> np.ndarray:
        frames = mog2_background_swap(frames, bank=background_bank, apply_prob=apply_bg_swap_prob)
        frames = color_jitter(frames)
        frames = small_affine(frames)
        flippable = flippable_lookup.get(sign_id, False) if flippable_lookup else False
        frames = horizontal_flip(frames, flippable=flippable)
        frames = random_spatial_crop(frames, height=input_height, width=input_width)
        return frames

    return apply


def make_val_transform(
    *, input_height: int = 96, input_width: int = 96
) -> Callable[[np.ndarray, str], np.ndarray]:
    """No-augmentation center-crop for val/test splits."""

    def apply(frames: np.ndarray, _sign_id: str) -> np.ndarray:
        T, H, W, _ = frames.shape
        top = max(0, (H - input_height) // 2)
        left = max(0, (W - input_width) // 2)
        return frames[:, top : top + input_height, left : left + input_width, :]

    return apply
