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


def jitter_coords(x: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """Add small Gaussian noise per coordinate. Models MediaPipe noise floor.

    Phase 9d.2: bumped from σ=0.005 → σ=0.01 to match the empirical noise
    scale measured on the v1 manifest (~1% of normalized-coord range).
    """
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


def keypoint_dropout(x: np.ndarray, p_landmark: float = 0.10, p_frame: float = 0.4) -> np.ndarray:
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


def rotate_xy(x: np.ndarray, max_deg: float = 20.0) -> np.ndarray:
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


def temporal_random_crop(x: np.ndarray, min_keep: float = 0.7) -> np.ndarray:
    """Phase 9d.2 — pure temporal random sub-window crop.

    Picks a random contiguous window covering at least ``min_keep`` of
    the original length, then resamples back to ``TEMPORAL_LENGTH``.
    Distinct from ``temporal_stretch`` (which interpolates speed), this
    simulates the case where the recorded clip starts before the sign
    onset or ends after the sign offset — the model should be robust to
    where in the 2-second window the sign actually fires.
    """
    T = x.shape[0]
    keep = max(int(round(min_keep * T)), 2)
    win = int(np.random.randint(keep, T + 1))
    start = int(np.random.randint(0, T - win + 1))
    cropped = x[start : start + win]
    if cropped.shape[0] == TEMPORAL_LENGTH:
        return cropped
    src_idx = np.linspace(0, cropped.shape[0] - 1, num=TEMPORAL_LENGTH)
    return np.stack([_lerp_row(cropped, float(i)) for i in src_idx], axis=0).astype(np.float32)


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
    """v1-style composer + 9d.2 defaults + 9e.4 mediapipe-noise (v2-009 build).

    The 9e.4 ``mediapipe_noise_apply`` (15% per-frame chance to drop one
    hand to zero) is now ALWAYS applied. It disrupts the source-mask
    shortcut: Sem-Lex's 0% miss rate would otherwise let the model use
    "ratio of zero frames" as a near-perfect source classifier, ignoring
    the actual sign content. Forcing all training clips through the same
    realistic noise model removes that shortcut.
    """
    out = x
    if flippable and np.random.rand() < 0.5:
        out = horizontal_flip_if_allowed(out, allowed=True)
    out = temporal_stretch(out)
    out = rotate_xy(out)
    out = keypoint_dropout(out)
    out = mediapipe_noise_apply(out, miss_rate=0.15)  # 9e.4 — source-shortcut breaker
    out = jitter_coords(out)
    return out


# ---------------------------------------------------------------------------
# Phase 9e — synthetic keypoint augmentation
#
# Pure functions. Composition into the training pipeline happens at 9e.6
# (the dataset / model touchpoint) which is deferred pending a model-side
# check-in per the Phase 9 working agreement. Available today for unit
# tests and explicit opt-in via experiments.
# ---------------------------------------------------------------------------


def skeleton_rescale_hands(
    x: np.ndarray,
    scale_range: tuple[float, float] = (0.85, 1.15),
) -> np.ndarray:
    """Phase 9e.1 — synthesize hand-size variation.

    Each hand block (left, right) is independently rescaled around its
    wrist landmark (index 0 of the hand). This simulates the body-
    proportion spread you see across signers without requiring a full
    skeletal-rig model — wrists stay where MediaPipe put them, finger
    distances stretch or compress isotropically. (Arm-length and
    shoulder-width rescaling would require structural connectivity the
    raw landmark tensor doesn't expose, so we don't attempt them.)
    """
    out = x.copy()
    T = out.shape[0]
    scale = float(np.random.uniform(*scale_range))
    for hand_offset in (LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET):
        anchor = out[:, hand_offset : hand_offset + COORDS_PER_LANDMARK]  # (T, 3) — wrist
        for i in range(1, NUM_HAND_LANDMARKS):  # skip wrist itself
            base = hand_offset + i * COORDS_PER_LANDMARK
            out[:, base : base + COORDS_PER_LANDMARK] = (
                anchor + (out[:, base : base + COORDS_PER_LANDMARK] - anchor) * scale
            )
    return out


def trajectory_perturbation(
    x: np.ndarray,
    sigma: float = 0.005,
    smooth_window: int = 4,
) -> np.ndarray:
    """Phase 9e.2 — add smooth (low-frequency) per-keypoint trajectory noise.

    Distinct from ``jitter_coords`` (white per-frame Gaussian): here the
    noise is correlated across time, so each landmark's path is pushed
    along a smooth curve rather than vibrated. Models slow drift like
    subtle camera movement or the kind of recording-condition variation
    real signers produce.
    """
    T, K = x.shape
    raw = np.random.normal(0.0, sigma, size=(T, K)).astype(np.float32)
    if smooth_window <= 1:
        return x + raw
    kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
    smoothed = np.stack(
        [np.convolve(raw[:, k], kernel, mode="same") for k in range(K)], axis=1
    ).astype(np.float32)
    return x + smoothed


def frame_mix_timing(
    x: np.ndarray,
    tau_range: tuple[float, float] = (0.3, 0.7),
    seg_factor_range: tuple[float, float] = (0.7, 1.4),
) -> np.ndarray:
    """Phase 9e.3 — synthesize timing variation by piecewise non-uniform resample.

    Picks a split point ``tau`` in [0.3, 0.7] of clip length, then
    independently stretches/compresses the [0, tau] and [tau, 1]
    segments before concatenating and resampling back to TEMPORAL_LENGTH.
    Mimics signers who execute the onset slowly and the body of the
    sign quickly (or vice versa) — variation that uniform
    ``temporal_stretch`` can't produce.
    """
    T = x.shape[0]
    tau = float(np.random.uniform(*tau_range))
    f1 = float(np.random.uniform(*seg_factor_range))
    f2 = float(np.random.uniform(*seg_factor_range))
    t1 = max(1, int(round(tau * T * f1)))
    t2 = max(1, int(round((1.0 - tau) * T * f2)))
    idx1 = np.linspace(0.0, tau * (T - 1), num=t1)
    idx2 = np.linspace(tau * (T - 1), float(T - 1), num=t2)
    src_idx = np.concatenate([idx1, idx2])
    resampled = np.stack([_lerp_row(x, float(i)) for i in src_idx], axis=0)
    if resampled.shape[0] >= TEMPORAL_LENGTH:
        start = int(np.random.randint(0, resampled.shape[0] - TEMPORAL_LENGTH + 1))
        return resampled[start : start + TEMPORAL_LENGTH]
    pad = np.tile(resampled[-1:], (TEMPORAL_LENGTH - resampled.shape[0], 1))
    return np.concatenate([resampled, pad], axis=0)


def mediapipe_noise_apply(
    x: np.ndarray,
    miss_rate: float = 0.15,
    p_hand_drop: float = 0.5,
) -> np.ndarray:
    """Phase 9e.4 — apply realistic MediaPipe-style noise to a clean clip.

    For each frame, with probability ``miss_rate`` drop one hand (zero
    its 21 landmarks). Mimics the asymmetric per-hand miss pattern we
    saw in the v1 manifest: hands aren't dropped together; one hand
    detects while the other misses, especially during inter-sign rest
    poses and motion blur peaks.

    The v2 manifest's `mediapipe_misses` distribution can later be
    used to fit ``miss_rate`` and the temporal correlation of misses.
    Until that data exists this implementation uses an IID frame model
    with a fixed ``miss_rate`` — structurally honest, parameters TBD.
    """
    out = x.copy()
    T = out.shape[0]
    for t in range(T):
        if np.random.rand() >= miss_rate:
            continue
        if np.random.rand() < p_hand_drop:
            out[t, LEFT_HAND_OFFSET : LEFT_HAND_OFFSET + HAND_FLOATS] = 0.0
        else:
            out[t, RIGHT_HAND_OFFSET : RIGHT_HAND_OFFSET + HAND_FLOATS] = 0.0
    return out


def rotate_3d(x: np.ndarray, max_deg: float = 15.0) -> np.ndarray:
    """Phase 9e.5 — small 3D rotation (yaw/pitch/roll) around the frame
    center. Uses the z-coordinate channel that ``rotate_xy`` ignores,
    so the model sees viewpoint variation the 2D rotation can't
    produce. Z-coords are in MediaPipe's relative-depth units (roughly
    image-width-scaled); we rotate around the origin (0.5, 0.5, 0.0)
    consistent with the rest of the augmentation stack.
    """
    yaw, pitch, roll = np.deg2rad(
        np.random.uniform(-max_deg, max_deg, size=3).astype(np.float32)
    )
    cz, sz = math.cos(yaw), math.sin(yaw)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cx, sx = math.cos(roll), math.sin(roll)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    R = (Rz @ Ry @ Rx).astype(np.float32)
    out = x.copy()
    n_landmarks = TOTAL_COORDS // COORDS_PER_LANDMARK
    center = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    for lm in range(n_landmarks):
        base = lm * COORDS_PER_LANDMARK
        pts = out[:, base : base + 3] - center  # (T, 3)
        out[:, base : base + 3] = pts @ R.T + center
    return out
