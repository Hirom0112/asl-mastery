"""Convert Sem-Lex pre-extracted features → our keypoint layout.

Sem-Lex ships `(T, 553, 3)` float16 arrays per clip:
  - 33 MediaPipe pose landmarks (indices 0..32)
  - 21 left-hand landmarks (33..53)
  - 21 right-hand landmarks (54..74)
  - 478 face landmarks (75..552)
  - Missing landmarks encoded as NaN
  - Coordinate space: custom body-centered normalization (NOT MediaPipe [0,1])

Our layout:
  - `(16, 150)` float32: 21 left + 21 right + 8 pose subset, flat (x, y, z)
  - Missing landmarks encoded as 0.0

Conversion steps per clip:
  1. Load .npy, convert to float32, replace NaN with 0.
  2. Re-normalize coordinates to match our [0, 1] convention: shift+scale so the
     y range fits in [0, 1] and x is centered. This is a heuristic; the
     classifier should be invariant to small affine differences after
     augmentation (Phase 9d.2 stronger rotation+jitter).
  3. Select the 50 landmarks matching our subset.
  4. Subsample T-variable → 16 frames via motion-weighted sampling (same as
     `training.data.clean._sample_frames` but on keypoints rather than pixels).
  5. Build the per-landmark mask sibling tensor (1.0 where a coord was
     non-NaN, 0.0 where it was missing) — matches the 9a.4 mask convention.
  6. Save `<clip_id>.npy` (16, 150) + `<clip_id>.mask.npy` (16, 50).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.keypoints import (  # noqa: E402
    COORDS_PER_LANDMARK,
    LEFT_HAND_OFFSET,
    NUM_HAND_LANDMARKS,
    POSE_INDEX_SUBSET,
    POSE_OFFSET,
    RIGHT_HAND_OFFSET,
    TEMPORAL_LENGTH,
    TOTAL_COORDS,
    TOTAL_LANDMARKS,
)

log = logging.getLogger("convert_sem_lex")

# Sem-Lex 553-landmark layout (assumed MediaPipe Holistic canonical order):
SEM_LEX_POSE_START = 0
SEM_LEX_POSE_END = 33
SEM_LEX_LEFT_HAND_START = 33
SEM_LEX_LEFT_HAND_END = 54
SEM_LEX_RIGHT_HAND_START = 54
SEM_LEX_RIGHT_HAND_END = 75
SEM_LEX_FACE_START = 75
SEM_LEX_FACE_END = 553


def _motion_weighted_sample_keypoints(arr: np.ndarray, mask: np.ndarray, T_out: int = TEMPORAL_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    """Subsample T_in keypoint frames → T_out via inverse-CDF over per-frame motion."""
    T_in = arr.shape[0]
    if T_in <= T_out:
        # Pad with last valid frame
        pad_arr = np.tile(arr[-1:], (T_out - T_in, 1, 1))
        pad_mask = np.tile(mask[-1:], (T_out - T_in, 1))
        return (
            np.concatenate([arr, pad_arr], axis=0),
            np.concatenate([mask, pad_mask], axis=0),
        )

    # Motion = mean abs diff of consecutive frames over present landmarks.
    diffs = np.zeros(T_in, dtype=np.float32)
    for t in range(1, T_in):
        # Only consider landmarks present in both frames.
        joint_present = mask[t] * mask[t - 1]  # (N,)
        if joint_present.sum() < 1:
            continue
        d = np.abs(arr[t] - arr[t - 1])  # (N, 3)
        # Average over present landmarks and all 3 coords
        diffs[t] = float((d.sum(axis=-1) * joint_present).sum() / max(joint_present.sum(), 1))
    peak = diffs.max()
    if peak <= 1e-6:
        idxs = np.linspace(0, T_in - 1, num=T_out, dtype=int)
    else:
        weights = diffs + 0.15 * peak
        cdf = np.cumsum(weights)
        cdf /= cdf[-1]
        quantiles = (np.arange(T_out) + 0.5) / T_out
        idxs = np.searchsorted(cdf, quantiles).clip(0, T_in - 1)
        # Monotonic with nudge for duplicates
        out_idxs: list[int] = []
        for i, idx in enumerate(idxs):
            if out_idxs and idx <= out_idxs[-1]:
                idx = min(out_idxs[-1] + 1, T_in - 1)
            out_idxs.append(int(idx))
        idxs = np.array(out_idxs)
    return arr[idxs], mask[idxs]


def convert_one(arr_553: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert one (T, 553, 3) float16 array to (16, 150) float32 + (16, 50) mask."""
    arr = arr_553.astype(np.float32)
    present_553 = ~np.isnan(arr[..., 0])  # (T, 553) — BEFORE NaN→0
    arr = np.nan_to_num(arr, nan=0.0)

    T_in = arr.shape[0]
    out_50 = np.zeros((T_in, TOTAL_LANDMARKS, COORDS_PER_LANDMARK), dtype=np.float32)
    mask_50 = np.zeros((T_in, TOTAL_LANDMARKS), dtype=np.float32)

    # Left hand (Sem-Lex 33..54 → our slots 0..21)
    out_50[:, 0:NUM_HAND_LANDMARKS, :] = arr[:, SEM_LEX_LEFT_HAND_START:SEM_LEX_LEFT_HAND_END, :]
    mask_50[:, 0:NUM_HAND_LANDMARKS] = present_553[:, SEM_LEX_LEFT_HAND_START:SEM_LEX_LEFT_HAND_END].astype(np.float32)

    # Right hand (Sem-Lex 54..75 → our slots 21..42)
    out_50[:, NUM_HAND_LANDMARKS : 2 * NUM_HAND_LANDMARKS, :] = arr[:, SEM_LEX_RIGHT_HAND_START:SEM_LEX_RIGHT_HAND_END, :]
    mask_50[:, NUM_HAND_LANDMARKS : 2 * NUM_HAND_LANDMARKS] = present_553[:, SEM_LEX_RIGHT_HAND_START:SEM_LEX_RIGHT_HAND_END].astype(np.float32)

    # Pose subset (Sem-Lex 0..33 → our slots 42..50, picking POSE_INDEX_SUBSET)
    for i, mp_idx in enumerate(POSE_INDEX_SUBSET):
        out_50[:, 2 * NUM_HAND_LANDMARKS + i, :] = arr[:, SEM_LEX_POSE_START + mp_idx, :]
        mask_50[:, 2 * NUM_HAND_LANDMARKS + i] = present_553[:, SEM_LEX_POSE_START + mp_idx].astype(np.float32)

    # Renormalize after dropping face — bbox over only the 50 kept landmarks so
    # the body+hands fill most of the [0,1] frame (matching our v2 training data).
    valid_xy = mask_50.astype(bool)  # (T, 50)
    if valid_xy.any():
        xs = out_50[..., 0][valid_xy]
        ys = out_50[..., 1][valid_xy]
        x_min, x_max = float(np.min(xs)), float(np.max(xs))
        y_min, y_max = float(np.min(ys)), float(np.max(ys))
        x_span = max(x_max - x_min, 1e-6)
        y_span = max(y_max - y_min, 1e-6)
        out_50[..., 0] = 0.05 + 0.9 * (out_50[..., 0] - x_min) / x_span
        out_50[..., 1] = 0.05 + 0.9 * (out_50[..., 1] - y_min) / y_span

    # Motion-weighted temporal subsample.
    out_50, mask_50 = _motion_weighted_sample_keypoints(out_50, mask_50, T_out=TEMPORAL_LENGTH)

    # Flatten (16, 50, 3) → (16, 150)
    out_flat = out_50.reshape(TEMPORAL_LENGTH, TOTAL_COORDS)
    return out_flat, mask_50


def process_chunk(
    chunk_path: Path,
    wanted_video_ids: set[str],
    dst_keypoints_dir: Path,
) -> tuple[int, int]:
    """Walk a Sem-Lex .tar.gz chunk; convert + save each wanted clip. Returns (converted, missing)."""
    dst_keypoints_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_skip = 0
    log.info("processing %s", chunk_path.name)
    with tarfile.open(chunk_path, "r:gz") as tf:
        for info in tf:
            if not info.isfile() or not info.name.endswith(".npy"):
                continue
            stem = Path(info.name).stem
            if stem not in wanted_video_ids:
                continue
            try:
                import io
                f = tf.extractfile(info)
                if f is None:
                    n_skip += 1
                    continue
                arr_553 = np.load(io.BytesIO(f.read()), allow_pickle=False)
                if arr_553.ndim != 3 or arr_553.shape[1] != 553:
                    log.warning("unexpected shape for %s: %s", stem, arr_553.shape)
                    n_skip += 1
                    continue
                kp, mask = convert_one(arr_553)
                np.save(dst_keypoints_dir / f"{stem}.npy", kp.astype(np.float32))
                np.save(dst_keypoints_dir / f"{stem}.mask.npy", mask.astype(np.float32))
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                log.warning("conversion failed for %s: %s", stem, e)
                n_skip += 1
            if (n_ok + n_skip) % 200 == 0:
                log.info("  progress: ok=%d skip=%d", n_ok, n_skip)
    return n_ok, n_skip


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, nargs="+", required=True)
    parser.add_argument("--wanted-ids", type=Path, required=True,
                        help="JSON list of {video_id, sign_id, signer_id, split} dicts.")
    parser.add_argument("--output-keypoints", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    with args.wanted_ids.open() as f:
        wanted = json.load(f)
    by_id: dict[str, dict[str, Any]] = {w["video_id"]: w for w in wanted}
    log.info("targeting %d unique video_ids across our 75 signs", len(by_id))

    n_ok_total = 0
    n_skip_total = 0
    for ch in args.chunks:
        if not ch.exists():
            log.warning("missing chunk: %s", ch)
            continue
        n_ok, n_skip = process_chunk(ch, set(by_id.keys()), args.output_keypoints)
        log.info("  → %s: converted=%d skipped=%d", ch.name, n_ok, n_skip)
        n_ok_total += n_ok
        n_skip_total += n_skip

    # Build manifest
    records: list[dict[str, Any]] = []
    for vid, info in by_id.items():
        kp_path = args.output_keypoints / f"{vid}.npy"
        mask_path = args.output_keypoints / f"{vid}.mask.npy"
        if not kp_path.exists():
            records.append({
                "clip_id": vid, "sign_id": info["sign_id"],
                "source": "sem_lex", "source_video_id": vid,
                "source_signer_id": str(info.get("signer_id")) if info.get("signer_id") else None,
                "source_split": info.get("split"),
                "split": "pending",
                "keypoint_path": None, "mask_path": None,
                "download_status": "missing",
            })
            continue
        records.append({
            "clip_id": vid, "sign_id": info["sign_id"],
            "source": "sem_lex", "source_video_id": vid,
            "source_signer_id": str(info.get("signer_id")) if info.get("signer_id") else None,
            "source_split": info.get("split"),
            "split": "pending",
            "keypoint_path": str(kp_path),
            "mask_path": str(mask_path),
            "download_status": "ok",
            "frame_start": None, "frame_end": None,
            "mediapipe_misses": 0,
            "sign_window_trimmed": False,
            "hand_crop_bbox": None,
            "converted_from_sem_lex": True,
        })

    manifest = {
        "source": "sem_lex",
        "license": "CC BY-NC-SA 4.0",
        "citation": "Kezar et al. 2023, ASSETS '23 (arXiv:2310.00196)",
        "n_records": len(records),
        "n_converted": n_ok_total,
        "clips": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    log.info("wrote %s (%d converted, %d missing)", args.manifest, n_ok_total,
             sum(1 for r in records if r["download_status"] == "missing"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
