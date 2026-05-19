"""Keypoint-level augmentation per docs/MODEL.md §3.

Training-time only. All augmentations operate on the (T, K)
keypoint tensor produced by the cleaning pipeline. Pixel-level
augmentations from the superseded Path B are not used.
"""

from __future__ import annotations

import math

import numpy as np

from training.keypoints import (
    COORDS_PER_LANDMARK,
    LEFT_HAND_OFFSET,
    POSE_OFFSET,
    POSE_FLOATS,
    HAND_FLOATS,
    NUM_HAND_LANDMARKS,
    NUM_POSE_LANDMARKS,
    RIGHT_HAND_OFFSET,
    TEMPORAL_LENGTH,
    TOTAL_COORDS,
)


def jitter_coords(x: np.ndarray, sigma: float = 0.005) -> np.ndarray:
    """Add small Gaussian noise per coordinate. Models MediaPipe noise floor."""
    return x + np.random.normal(0, sigma, size=x.shape).astype(np.float32)


def temporal_stretch(x: np.ndarray, factor_range: tuple[float, float] = (0.85, 1.15)) -> np.ndarray:
    """Resample to a stretched length, then crop/pad back to TEMPORAL_LENGTH."""
    factor = np.random.uniform(*factor_range)
    new_T = max(2, int(round(x.shape[0] * factor)))
    # Linear-interp along temporal axis.
    src_idx = np.linspace(0, x.shape[0] - 1, num=new_T)
    stretched = np.stack([_lerp_row(x, idx) for idx in src_idx], axis=0)

    # Crop to TEMPORAL_LENGTH (random window) or pad with last row.
    if stretched.shape[0] >= TEMPORAL_LENGTH:
        start = np.random.randint(0, stretched.shape[0] - TEMPORAL_LENGTH + 1)
        return stretched[start : start + TEMPORAL_LENGTH]
    pad = np.tile(stretched[-1:], (TEMPORAL_LENGTH - stretched.shape[0], 1))
    return np.concatenate([stretched, pad], axis=0)


def _lerp_row(x: np.ndarray, idx: float) -> np.ndarray:
    lo = int(math.floor(idx))
    hi = min(lo + 1, x.shape[0] - 1)
    frac = idx - lo
    return ((1 - frac) * x[lo] + frac * x[hi]).astype(np.float32)


def keypoint_dropout(x: np.ndarray, p_landmark: float = 0.05, p_frame: float = 0.2) -> np.ndarray:
    """Zero a random subset of landmarks on a random subset of frames."""
    out = x.copy()
    T = out.shape[0]
    n_landmarks = (HAND_FLOATS * 2 + POSE_FLOATS) // COORDS_PER_LANDMARK
    for t in range(T):
        if np.random.rand() > p_frame:
            continue
        mask = np.random.rand(n_landmarks) < p_landmark
        for lm_idx in np.where(mask)[0]:
            base = lm_idx * COORDS_PER_LANDMARK
            out[t, base : base + COORDS_PER_LANDMARK] = 0.0
    return out


def rotate_xy(x: np.ndarray, max_deg: float = 10.0) -> np.ndarray:
    """Apply a small 2D rotation to all (x, y) pairs jointly. z untouched."""
    theta = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    c, s = math.cos(theta), math.sin(theta)
    out = x.copy()
    # Rotate around the frame center (0.5, 0.5) in normalized coords.
    cx, cy = 0.5, 0.5
    n_landmarks = TOTAL_COORDS // COORDS_PER_LANDMARK
    for lm in range(n_landmarks):
        base = lm * COORDS_PER_LANDMARK
        xs = out[:, base] - cx
        ys = out[:, base + 1] - cy
        out[:, base] = xs * c - ys * s + cx
        out[:, base + 1] = xs * s + ys * c + cy
    return out


def horizontal_flip_if_allowed(x: np.ndarray, allowed: bool) -> np.ndarray:
    """Flip via x-coordinate negation + hand-group swap. No-op if not allowed.

    `allowed` should be the value of `flippable` from
    `docs/VOCABULARY.md` for this clip's sign. Per ADR 0006, the
    classifier's invariance to handedness is encoded by which signs
    we allow to flip, not by the model.
    """
    if not allowed:
        return x
    out = x.copy()
    # Mirror x for every landmark (x → 1 - x in normalized coords).
    n_landmarks = TOTAL_COORDS // COORDS_PER_LANDMARK
    for lm in range(n_landmarks):
        base = lm * COORDS_PER_LANDMARK
        out[:, base] = 1.0 - out[:, base]

    # Swap left-hand and right-hand blocks so the model sees a
    # consistent "right-handed signer" input.
    left = out[:, LEFT_HAND_OFFSET : LEFT_HAND_OFFSET + HAND_FLOATS].copy()
    right = out[:, RIGHT_HAND_OFFSET : RIGHT_HAND_OFFSET + HAND_FLOATS].copy()
    out[:, LEFT_HAND_OFFSET : LEFT_HAND_OFFSET + HAND_FLOATS] = right
    out[:, RIGHT_HAND_OFFSET : RIGHT_HAND_OFFSET + HAND_FLOATS] = left
    return out


def augment(x: np.ndarray, flippable: bool) -> np.ndarray:
    """Compose the slice-1 augmentation stack."""
    out = x
    if np.random.rand() < 0.5 and flippable:
        out = horizontal_flip_if_allowed(out, allowed=True)
    out = temporal_stretch(out)
    out = rotate_xy(out)
    out = keypoint_dropout(out)
    out = jitter_coords(out)
    return out
