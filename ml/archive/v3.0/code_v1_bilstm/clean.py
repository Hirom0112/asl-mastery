"""Cleaning + keypoint-extraction pipeline.

Phase 3f. Single command: reads raw clips from `dataset/raw/`,
runs the stages from docs/ARCHITECTURE.md §2.5 and docs/DATASET.md §3,
writes versioned outputs to `dataset/clean/v<N>/` and emits the
manifest the training loop consumes.

Stages (per docs/DATASET.md §3, re-numbered under ADR 0008):
  1. Ingest raw clips (already on disk from Phase 3d ingestion).
  2. Per-sign clip-count filter (ADR 0008) — already in slice1_vocabulary.json.
  3. Sign-window trim (from WLASL frame_start/frame_end).
  4. Frame-rate normalization to 30 fps.
  5. Length normalization (sample 16 frames evenly).
  6. Resize to 256×256 (square crop, then bilinear).
  7. Dedup via pHash (within a source-signer).
  8. MediaPipe Holistic extraction → (T=16, K=150) keypoint tensor.
  9. Signer-disjoint split assignment.
 10. Manifest write.

Usage:

    python -m training.data.clean \\
        --raw-manifest dataset/raw/wlasl_manifest.json \\
        --filter dataset/slice1_vocabulary.json \\
        --output dataset/clean/v1/ \\
        --version v1 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import subprocess
import sys
from collections import defaultdict
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
)

log = logging.getLogger("clean")

# MediaPipe import is deferred to avoid blocking imports for testing.
# Set in _lazy_mediapipe().
_MEDIAPIPE = None
_HOLISTIC = None  # Process-global Holistic instance (created lazily per worker)
MEDIAPIPE_VERSION_TARGET = "0.10.18"  # must match training/requirements.txt


def _lazy_mediapipe():
    global _MEDIAPIPE
    if _MEDIAPIPE is not None:
        return _MEDIAPIPE
    import mediapipe as mp  # type: ignore

    _MEDIAPIPE = mp
    return mp


def _get_holistic():
    """Return a process-global MediaPipe Holistic instance.

    Created lazily on first use (each ProcessPoolExecutor worker initializes
    its own instance). Re-using across clips saves ~0.5–1 s of startup per
    clip; over a 4k-clip dataset that's half the wall-clock. Cleanup happens
    at process exit; we don't ``__exit__`` the context because the worker
    process is short-lived.
    """
    global _HOLISTIC
    if _HOLISTIC is None:
        mp = _lazy_mediapipe()
        _HOLISTIC = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
        )
    return _HOLISTIC


# -- Stage 4–6: video re-encode to 30 fps, 256×256, then frame extraction.


def _detect_hand_bbox(
    src: Path,
    frame_start: int | None,
    frame_end: int | None,
    num_probes: int = 6,
) -> tuple[int, int, int, int] | None:
    """Probe `num_probes` frames spread across the sign window and return
    a square (x, y, w, h) crop bbox in source-pixel coordinates that
    encompasses all detected hand landmarks (with shoulder fallback).

    Returns None if MediaPipe finds nothing usable — caller falls back
    to center-square crop. Phase 9a.3.
    """
    import cv2  # type: ignore

    mp = _lazy_mediapipe()

    cap = cv2.VideoCapture(str(src))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0 or src_w <= 0 or src_h <= 0:
        cap.release()
        return None

    has_start = isinstance(frame_start, int) and frame_start > 1
    has_end = isinstance(frame_end, int) and frame_end > 0
    start = (frame_start - 1) if has_start else 0
    end = (frame_end - 1) if has_end else (total - 1)
    if end <= start:
        end = total - 1
    probe_idxs = np.linspace(start, end, num=num_probes, dtype=int)

    xs: list[float] = []
    ys: list[float] = []
    shoulder_xs: list[float] = []
    shoulder_ys: list[float] = []
    with mp.solutions.holistic.Holistic(
        static_image_mode=True,
        model_complexity=0,
        enable_segmentation=False,
        refine_face_landmarks=False,
    ) as holistic:
        for idx in probe_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            for hand_lm in (res.left_hand_landmarks, res.right_hand_landmarks):
                if not hand_lm:
                    continue
                for lm in hand_lm.landmark:
                    xs.append(float(lm.x) * src_w)
                    ys.append(float(lm.y) * src_h)
            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                if len(lms) > 12:
                    for i in (11, 12):
                        shoulder_xs.append(float(lms[i].x) * src_w)
                        shoulder_ys.append(float(lms[i].y) * src_h)
    cap.release()

    if not xs:
        # Hand fallback to shoulders: center crop around shoulders, square
        # with side = 2.5 × shoulder distance (covers signing space above
        # and below the shoulder line).
        if len(shoulder_xs) < 2:
            return None
        sx_min, sx_max = min(shoulder_xs), max(shoulder_xs)
        sy_min, sy_max = min(shoulder_ys), max(shoulder_ys)
        shoulder_w = max(sx_max - sx_min, 1.0)
        cx = (sx_min + sx_max) / 2
        cy = (sy_min + sy_max) / 2
        side = shoulder_w * 2.5
    else:
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2
        # 50% padding around the hand-activity span, and a floor of half
        # the smaller source dim so we don't over-crop a single static hand.
        side = max((maxx - minx), (maxy - miny)) * 1.5

    # Resolution preservation: never crop tighter than 70% of the smaller
    # source dimension, and never below the 256 px target output (avoids
    # upscaling which destroys fingertip detail and tanks the MediaPipe
    # miss rate — see Phase 9a.3 verification notes).
    side = max(side, min(src_w, src_h) * 0.7)
    side = max(side, 256.0)
    side = min(side, float(min(src_w, src_h)))
    cx = max(side / 2, min(src_w - side / 2, cx))
    cy = max(side / 2, min(src_h - side / 2, cy))

    # If the hand-activity center is close to the frame center (within 10%
    # of the smaller dimension), the center-square crop already captures
    # the signing space — skip hand-aware crop. Avoids regressions on
    # well-framed clips where the only cost is upscale noise.
    src_cx = src_w / 2.0
    src_cy = src_h / 2.0
    if (
        abs(cx - src_cx) < 0.10 * min(src_w, src_h)
        and abs(cy - src_cy) < 0.10 * min(src_w, src_h)
    ):
        return None
    x = int(round(cx - side / 2))
    y = int(round(cy - side / 2))
    s = int(round(side))
    # Ensure inside bounds after rounding.
    if x + s > src_w:
        x = src_w - s
    if y + s > src_h:
        y = src_h - s
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    return x, y, s, s


def _ffmpeg_normalize(
    src: Path,
    dst: Path,
    fps: int = 30,
    size: int = 256,
    frame_start: int | None = None,
    frame_end: int | None = None,
    crop_bbox: tuple[int, int, int, int] | None = None,
) -> None:
    """Re-encode src → dst at `fps` fps, square-cropped to `size`.

    If `frame_start`/`frame_end` are provided (WLASL sign-window annotation,
    1-indexed in the source video's original frame numbering, with -1 meaning
    "to end"), the trim is applied *before* scale/crop/fps so MediaPipe only
    sees the sign portion of the clip. See Phase 9a.1 in TODO.md.
    """
    vf_parts: list[str] = []

    # Sign-window trim (WLASL frame_start/frame_end). Both fields are
    # 1-indexed in the source's native frame numbering. frame_end == -1
    # means "to end of video"; frame_start == 1 means "from beginning".
    # Only emit a select filter if at least one bound is non-trivial.
    has_start = isinstance(frame_start, int) and frame_start > 1
    has_end = isinstance(frame_end, int) and frame_end > 0
    if has_start or has_end:
        start_idx = (frame_start - 1) if has_start else 0
        if has_end and frame_end - 1 > start_idx:
            vf_parts.append(f"select='between(n,{start_idx},{frame_end - 1})'")
        elif has_start:
            vf_parts.append(f"select='gte(n,{start_idx})'")
        else:
            vf_parts.append(f"select='lte(n,{frame_end - 1})'")
        # Reset PTS so the trimmed stream is contiguous for the downstream fps filter.
        vf_parts.append("setpts=N/FRAME_RATE/TB")

    if crop_bbox is not None:
        cx, cy, cw, ch = crop_bbox
        # Crop in source pixel coords first, then scale the (already square)
        # bbox to `size`×`size`. No aspect-preservation needed.
        vf_parts.append(f"crop={cw}:{ch}:{cx}:{cy}")
        vf_parts.append(f"scale={size}:{size}")
    else:
        vf_parts.append(f"scale={size}:{size}:force_original_aspect_ratio=increase")
        vf_parts.append(f"crop={size}:{size}")
    vf_parts.append(f"fps={fps}")
    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        vf,
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def _sample_frames(
    video_path: Path,
    num_frames: int = TEMPORAL_LENGTH,
    motion_weighted: bool = True,
    motion_floor: float = 0.15,
) -> list[np.ndarray]:
    """Sample `num_frames` BGR frames from the clip.

    With ``motion_weighted=True`` (default, Phase 9a.2), the 16 chosen
    frames are drawn from the per-frame motion-magnitude CDF — frame-to-
    frame mean absolute grayscale difference, plus a ``motion_floor``
    fraction of the peak as a uniform offset so flat regions still get
    *some* representation (a single high-motion spike must not collapse
    all 16 samples onto adjacent frames).

    For clips of <= ``num_frames`` frames, falls back to whatever frames
    exist (caller pads).
    """
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    # Read every frame once. For our normalized clips (≤ a few hundred
    # frames at 30 fps × 256 px) this is cheap and lets us compute motion
    # without a second pass.
    all_frames: list[np.ndarray] = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        all_frames.append(f)
    cap.release()

    if not all_frames:
        return []

    if len(all_frames) <= num_frames or not motion_weighted:
        idxs = np.linspace(0, len(all_frames) - 1, num=num_frames, dtype=int)
        return [all_frames[int(i)].copy() for i in idxs]

    # Inter-frame motion magnitude: mean abs diff of consecutive grayscale frames.
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in all_frames]
    diffs = np.zeros(len(grays), dtype=np.float32)
    for i in range(1, len(grays)):
        diffs[i] = float(np.mean(np.abs(grays[i].astype(np.int16) - grays[i - 1].astype(np.int16))))

    peak = diffs.max()
    if peak <= 1e-6:
        # Degenerate (static video) — fall back to even sampling.
        idxs = np.linspace(0, len(all_frames) - 1, num=num_frames, dtype=int)
        return [all_frames[int(i)].copy() for i in idxs]

    weights = diffs + motion_floor * peak
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    # Sample at the midpoints of `num_frames` equal-probability bins so we
    # don't anchor the first/last sample to the clip endpoints.
    quantiles = (np.arange(num_frames) + 0.5) / num_frames
    idxs = np.searchsorted(cdf, quantiles).clip(0, len(all_frames) - 1)
    # Ensure monotonic ordering (searchsorted already gives sorted output
    # for sorted quantiles, but de-dup adjacent indices by nudging if they
    # collide — preserves temporal ordering even on heavy peaks).
    out_idxs: list[int] = []
    for i, idx in enumerate(idxs):
        if out_idxs and idx <= out_idxs[-1]:
            idx = min(out_idxs[-1] + 1, len(all_frames) - 1)
        out_idxs.append(int(idx))
    return [all_frames[i].copy() for i in out_idxs]


# -- Stage 7: pHash for intra-source-signer dedup.


def _phash(frame: np.ndarray) -> str:
    import cv2  # type: ignore

    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (8, 8))
    avg = small.mean()
    bits = (small > avg).flatten().astype(np.uint8)
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return f"{h:016x}"


# -- Stage 8: MediaPipe Holistic keypoint extraction.


def _extract_keypoints_for_clip(
    frames: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run MediaPipe Holistic per frame.

    Returns ``(keypoints, mask, miss_count)``:

    - ``keypoints``: shape ``(T, TOTAL_COORDS)`` float32 — same layout as
      before this change.
    - ``mask``: shape ``(T, TOTAL_LANDMARKS)`` float32 in ``[0, 1]``. One
      scalar per *landmark* (not per coord). Hand landmarks: ``1.0`` if
      that hand was detected in the frame, ``0.0`` otherwise. Pose
      landmarks: ``visibility × presence``. Phase 9a.4 — consumed by
      Phase 9d.1 (frame-level masking in the model).
    - ``miss_count``: number of frames where neither hand was detected
      (kept for backward compatibility with the existing miss-rate logs
      and the >30% drop filter in 9a.5).
    """
    NUM_LANDMARKS = TOTAL_COORDS // COORDS_PER_LANDMARK
    LEFT_HAND_LM_START = LEFT_HAND_OFFSET // COORDS_PER_LANDMARK  # 0
    RIGHT_HAND_LM_START = RIGHT_HAND_OFFSET // COORDS_PER_LANDMARK  # 21
    POSE_LM_START = POSE_OFFSET // COORDS_PER_LANDMARK  # 42

    out = np.zeros((TEMPORAL_LENGTH, TOTAL_COORDS), dtype=np.float32)
    mask = np.zeros((TEMPORAL_LENGTH, NUM_LANDMARKS), dtype=np.float32)
    miss = 0

    holistic = _get_holistic()  # process-global instance; reused across clips
    if True:  # preserves indentation against the previous `with` block
        for t, bgr in enumerate(frames[:TEMPORAL_LENGTH]):
            import cv2  # type: ignore

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            row = out[t]
            mrow = mask[t]

            def _write_hand(landmarks, offset: int, lm_start: int) -> bool:
                if not landmarks:
                    return False
                lms = landmarks.landmark
                if len(lms) < NUM_HAND_LANDMARKS:
                    return False
                for i in range(NUM_HAND_LANDMARKS):
                    base = offset + i * COORDS_PER_LANDMARK
                    row[base] = lms[i].x
                    row[base + 1] = lms[i].y
                    row[base + 2] = lms[i].z
                    mrow[lm_start + i] = 1.0
                return True

            has_left = _write_hand(res.left_hand_landmarks, LEFT_HAND_OFFSET, LEFT_HAND_LM_START)
            has_right = _write_hand(res.right_hand_landmarks, RIGHT_HAND_OFFSET, RIGHT_HAND_LM_START)
            if not (has_left or has_right):
                miss += 1

            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                for i, mp_idx in enumerate(POSE_INDEX_SUBSET):
                    base = POSE_OFFSET + i * COORDS_PER_LANDMARK
                    if mp_idx < len(lms):
                        lm = lms[mp_idx]
                        row[base] = lm.x
                        row[base + 1] = lm.y
                        row[base + 2] = lm.z
                        # MediaPipe Holistic populates `visibility` (~1.0
                        # when confident) but leaves `presence` at 0 for
                        # pose. Use visibility alone as the soft-mask
                        # weight, clamped into [0, 1].
                        vis = float(getattr(lm, "visibility", 1.0))
                        mrow[POSE_LM_START + i] = max(0.0, min(1.0, vis))

    return out, mask, miss


# -- Stage 9: signer-disjoint split assignment.


def _hash_str(s: str, n_buckets: int) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % n_buckets


def _assign_splits_stratified(
    clip_records: list[dict[str, Any]],
    seed: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> None:
    """Phase 9a.6 — per-sign stratified 70/15/15 split, signer-clustered.

    For each sign:
      1. Deterministically order clips so all clips from the same source
         signer are adjacent. This preserves the *spirit* of signer-
         disjoint splitting (no signer split across train+test) without
         the population-imbalance failure mode the old global-hash
         bucketing had.
      2. Assign the first ``train_frac`` of the ordered list to train,
         the next ``val_frac`` to val, the remainder to test.
      3. Guarantee every sign with ≥3 clips contributes at least one
         val and one test sample (steal from train if needed).

    Mutates each record in place, adding ``split`` ∈ {train, val, test}.
    """
    by_sign: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in clip_records:
        by_sign[c["sign_id"]].append(c)

    for sign_id, clips in by_sign.items():
        def _key(c: dict[str, Any]) -> tuple[int, int]:
            signer = c.get("source_signer_id") or f"_{c['source']}_{c['clip_id']}"
            return (
                _hash_str(f"{seed}:{sign_id}:signer:{signer}", 1 << 30),
                _hash_str(f"{seed}:{sign_id}:clip:{c['clip_id']}", 1 << 30),
            )

        clips.sort(key=_key)
        n = len(clips)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        # Ensure non-empty val + test when we have the budget.
        if n >= 3:
            if n_val < 1:
                n_val = 1
                n_train = min(n_train, n - n_val - 1)
            if n - n_train - n_val < 1:
                n_train = max(0, n - n_val - 1)
        for i, c in enumerate(clips):
            if i < n_train:
                c["split"] = "train"
            elif i < n_train + n_val:
                c["split"] = "val"
            else:
                c["split"] = "test"


# -- Per-clip worker (top-level so it pickles cleanly for ProcessPoolExecutor)


def _process_one_clip(
    rec: dict[str, Any],
    norm_dir: str,
    kp_dir: str,
    hand_aware_crop: bool,
    skip_normalize: bool,
    max_miss_rate: float,
) -> dict[str, Any]:
    """Run the full per-clip cleaning pipeline on one raw record.

    Top-level (not nested in ``clean()``) so it pickles for
    ``ProcessPoolExecutor``. Returns a dict with one of:
      - ``{"kind": "ok", "record": <clip_record_dict>}``
      - ``{"kind": "drop", "info": <drop_info_dict>}``  (Phase 9a.5)
      - ``{"kind": "skip", "reason": <str>}``           (preflight failure)
    """
    src_path = Path(rec["local_path"])
    if not src_path.exists():
        return {"kind": "skip", "reason": "missing-local"}

    clip_id = src_path.stem
    dst_video = Path(norm_dir) / f"{clip_id}.mp4"

    # Idempotency: if the keypoint + mask files already exist (e.g. an
    # earlier run extracted this clip and crashed elsewhere), reuse them
    # and reconstruct the record without re-running MediaPipe.
    kp_path = Path(kp_dir) / f"{clip_id}.npy"
    mask_path = Path(kp_dir) / f"{clip_id}.mask.npy"
    if kp_path.exists() and mask_path.exists():
        try:
            kp_mask = np.load(mask_path)
            # Per-frame validity: a frame is "missed" if both hands missed
            # → mask[t, :42] == 0. Hand masks are 0/1 per hand per frame
            # by construction (9a.4), so we can reconstruct miss count.
            left_hand_present = kp_mask[:, 0] > 0
            right_hand_present = kp_mask[:, 21] > 0
            frame_present = left_hand_present | right_hand_present
            miss = int((~frame_present).sum())
            miss_rate = miss / TEMPORAL_LENGTH
        except Exception:  # noqa: BLE001 — fall through to fresh extraction
            kp_path.unlink(missing_ok=True)
            mask_path.unlink(missing_ok=True)
        else:
            if miss_rate > max_miss_rate:
                return {
                    "kind": "drop",
                    "info": {
                        "clip_id": clip_id,
                        "sign_id": rec["sign_id"],
                        "source": rec["source"],
                        "source_video_id": rec.get("wlasl_video_id"),
                        "mediapipe_misses": miss,
                        "miss_rate": miss_rate,
                    },
                }
            frame_start = rec.get("frame_start")
            frame_end = rec.get("frame_end")
            return {
                "kind": "ok",
                "record": {
                    "clip_id": clip_id,
                    "sign_id": rec["sign_id"],
                    "source": rec["source"],
                    "source_video_id": rec.get("wlasl_video_id"),
                    "source_signer_id": rec.get("source_signer_id"),
                    "source_split": rec.get("source_split"),
                    "split": "pending",
                    "keypoint_path": str(kp_path),
                    "mask_path": str(mask_path),
                    "normalized_video_path": str(dst_video) if dst_video.exists() else None,
                    "mediapipe_misses": miss,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "sign_window_trimmed": bool(
                        (isinstance(frame_start, int) and frame_start > 1)
                        or (isinstance(frame_end, int) and frame_end > 0)
                    ),
                    "hand_crop_bbox": None,
                    "resumed": True,
                },
            }

    frame_start = rec.get("frame_start")
    frame_end = rec.get("frame_end")
    crop_bbox: tuple[int, int, int, int] | None = None

    if not skip_normalize:
        if hand_aware_crop:
            try:
                crop_bbox = _detect_hand_bbox(src_path, frame_start, frame_end)
            except Exception:  # noqa: BLE001
                crop_bbox = None
        try:
            _ffmpeg_normalize(
                src_path,
                dst_video,
                frame_start=frame_start,
                frame_end=frame_end,
                crop_bbox=crop_bbox,
            )
        except subprocess.CalledProcessError:
            return {"kind": "skip", "reason": "ffmpeg-failed"}

    frames = _sample_frames(dst_video if dst_video.exists() else src_path)
    if len(frames) < TEMPORAL_LENGTH and frames:
        while len(frames) < TEMPORAL_LENGTH:
            frames.append(frames[-1])
    if not frames:
        return {"kind": "skip", "reason": "no-frames"}

    keypoints, kp_mask, miss = _extract_keypoints_for_clip(frames)
    miss_rate = miss / TEMPORAL_LENGTH
    if miss_rate > max_miss_rate:
        return {
            "kind": "drop",
            "info": {
                "clip_id": clip_id,
                "sign_id": rec["sign_id"],
                "source": rec["source"],
                "source_video_id": rec.get("wlasl_video_id"),
                "mediapipe_misses": miss,
                "miss_rate": miss_rate,
            },
        }

    kp_path = Path(kp_dir) / f"{clip_id}.npy"
    mask_path = Path(kp_dir) / f"{clip_id}.mask.npy"
    np.save(kp_path, keypoints)
    np.save(mask_path, kp_mask)

    return {
        "kind": "ok",
        "record": {
            "clip_id": clip_id,
            "sign_id": rec["sign_id"],
            "source": rec["source"],
            "source_video_id": rec.get("wlasl_video_id"),
            "source_signer_id": rec.get("source_signer_id"),
            "source_split": rec.get("source_split"),
            "split": "pending",  # set by _assign_splits_stratified() below
            "keypoint_path": str(kp_path),
            "mask_path": str(mask_path),
            "normalized_video_path": str(dst_video) if dst_video.exists() else None,
            "mediapipe_misses": miss,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "sign_window_trimmed": bool(
                (isinstance(frame_start, int) and frame_start > 1)
                or (isinstance(frame_end, int) and frame_end > 0)
            ),
            "hand_crop_bbox": list(crop_bbox) if crop_bbox else None,
        },
    }


# -- Main pipeline


def _merge_raw_manifests(paths: list[Path]) -> dict:
    """Concatenate multiple raw ingestion manifests (WLASL + Lifeprint
    + ytsearch) into a single record stream for the cleaning pipeline."""
    all_records: list[dict[str, Any]] = []
    for p in paths:
        with p.open() as f:
            m = json.load(f)
        all_records.extend(m.get("records", []))
    return {"records": all_records}


def clean(
    raw_manifests: list[Path],
    filter_path: Path,
    output_dir: Path,
    version: str,
    seed: int,
    skip_normalize: bool,
    hand_aware_crop: bool = False,
    max_miss_rate: float = 0.30,
    workers: int = 1,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    raw = _merge_raw_manifests(raw_manifests)
    with filter_path.open() as f:
        vocab_filter = json.load(f)

    kept_signs = {item["sign_id"] for item in vocab_filter["kept_signs"]}
    log.info(
        "kept %d signs from filter (applied_floor=%d, reduced=%s)",
        len(kept_signs),
        vocab_filter["applied_floor"],
        vocab_filter["floor_was_reduced"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    norm_dir = output_dir / "normalized_videos"
    kp_dir = output_dir / "keypoints"
    norm_dir.mkdir(exist_ok=True)
    kp_dir.mkdir(exist_ok=True)

    # Pre-filter: only records that pass the basic gates go to workers.
    eligible: list[dict[str, Any]] = []
    for rec in raw["records"]:
        if rec.get("download_status") != "ok" or not rec.get("local_path"):
            continue
        if rec["sign_id"] not in kept_signs:
            continue
        eligible.append(rec)
    log.info("dispatching %d eligible clips to %d worker(s)", len(eligible), workers)

    clip_records: list[dict[str, Any]] = []
    dropped_high_miss: list[dict[str, Any]] = []  # Phase 9a.5
    skipped_count = 0

    def _absorb(result: dict[str, Any]) -> None:
        nonlocal skipped_count
        kind = result["kind"]
        if kind == "ok":
            clip_records.append(result["record"])
        elif kind == "drop":
            dropped_high_miss.append(result["info"])
        else:
            skipped_count += 1

    norm_dir_s = str(norm_dir)
    kp_dir_s = str(kp_dir)

    if workers <= 1:
        for i, rec in enumerate(eligible):
            res = _process_one_clip(
                rec, norm_dir_s, kp_dir_s, hand_aware_crop, skip_normalize, max_miss_rate
            )
            _absorb(res)
            if (i + 1) % 50 == 0:
                log.info("progress: %d/%d (ok=%d drop=%d skip=%d)",
                         i + 1, len(eligible), len(clip_records), len(dropped_high_miss), skipped_count)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        log.info("ProcessPoolExecutor: %d workers", workers)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(
                    _process_one_clip,
                    rec, norm_dir_s, kp_dir_s, hand_aware_crop, skip_normalize, max_miss_rate,
                )
                for rec in eligible
            ]
            done = 0
            for fut in as_completed(futures):
                try:
                    _absorb(fut.result())
                except Exception as e:  # noqa: BLE001 — worker exception shouldn't kill the run
                    log.warning("worker raised: %s", e)
                    skipped_count += 1
                done += 1
                if done % 100 == 0:
                    log.info(
                        "progress: %d/%d (ok=%d drop=%d skip=%d)",
                        done, len(eligible), len(clip_records), len(dropped_high_miss), skipped_count,
                    )

    log.info(
        "extraction done: kept=%d dropped_high_miss=%d skipped=%d",
        len(clip_records), len(dropped_high_miss), skipped_count,
    )

    # Phase 9a.6 — stratified-per-sign split assignment with signer clustering.
    _assign_splits_stratified(clip_records, seed)

    # Write manifest.
    mp = _lazy_mediapipe() if not skip_normalize else None
    mp_version = getattr(mp, "__version__", "skipped") if mp else "skipped"

    manifest = {
        "version": version,
        "seed": seed,
        "mediapipe_version": mp_version,
        "mediapipe_version_target": MEDIAPIPE_VERSION_TARGET,
        "vocabulary_filter": {
            "applied_floor": vocab_filter["applied_floor"],
            "requested_floor": vocab_filter["requested_floor"],
            "floor_was_reduced": vocab_filter["floor_was_reduced"],
            "kept_count": vocab_filter["kept_count"],
            "dropped_signs": [d["sign_id"] for d in vocab_filter["dropped_signs"]],
        },
        "clips": clip_records,
        "stage_count": {
            "raw_records": len(raw["records"]),
            "after_filter": sum(
                1 for r in raw["records"] if r["sign_id"] in kept_signs and r["download_status"] == "ok"
            ),
            "after_dedup_and_mediapipe": len(clip_records),
            "dropped_high_miss_rate": len(dropped_high_miss),
            "high_miss_threshold": max_miss_rate,
        },
        "dropped_high_miss": dropped_high_miss,
    }
    manifest_path = output_dir / f"dataset_{version}_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    log.info("wrote %s with %d clips", manifest_path, len(clip_records))

    # Also write a separate splits file for easy diffing.
    splits_dir = Path(__file__).resolve().parents[1] / "splits"
    splits_dir.mkdir(exist_ok=True)
    splits = {
        "train": [c["clip_id"] for c in clip_records if c["split"] == "train"],
        "val": [c["clip_id"] for c in clip_records if c["split"] == "val"],
        "test": [c["clip_id"] for c in clip_records if c["split"] == "test"],
    }
    splits_path = splits_dir / f"{version}.json"
    with splits_path.open("w") as f:
        json.dump(splits, f, indent=2)
    log.info("wrote %s (train=%d val=%d test=%d)", splits_path, len(splits["train"]), len(splits["val"]), len(splits["test"]))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-manifest",
        type=Path,
        nargs="+",
        required=True,
        help="One or more raw ingestion manifests (e.g. WLASL + Lifeprint + ytsearch).",
    )
    parser.add_argument("--filter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-normalize",
        action="store_true",
        help="Skip ffmpeg re-encode (assume source already at 30 fps / 256). Useful for re-runs.",
    )
    parser.add_argument(
        "--hand-aware-crop",
        action="store_true",
        help="Phase 9a.3 — probe MediaPipe to find the hand-activity region and crop around it instead of center-square. Off by default; on the random 20-clip WLASL sample this regressed miss rate by ~2pp (well-framed clips don't benefit). Enable for targeted traveling-sign experiments.",
    )
    parser.add_argument(
        "--max-miss-rate",
        type=float,
        default=0.30,
        help="Phase 9a.5 — drop clips with MediaPipe miss rate above this threshold. Default 0.30. On the v1 distribution this drops most clips, so loosen for first re-extraction (e.g. 0.50) and tighten once 9a.2 + ASL Citizen lift the floor.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel processes for the per-clip pipeline (ffmpeg + sample + MediaPipe + save). 1 = serial (current behavior). On a Modal L4 8 vCPUs use 8.",
    )
    args = parser.parse_args()
    clean(
        args.raw_manifest,
        args.filter,
        args.output,
        args.version,
        args.seed,
        args.skip_normalize,
        hand_aware_crop=args.hand_aware_crop,
        max_miss_rate=args.max_miss_rate,
        workers=args.workers,
    )
    # args.raw_manifest is now list[Path] due to nargs="+"
    return 0


if __name__ == "__main__":
    sys.exit(main())
