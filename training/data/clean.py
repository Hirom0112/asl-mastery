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
MEDIAPIPE_VERSION_TARGET = "0.10.18"  # must match training/requirements.txt


def _lazy_mediapipe():
    global _MEDIAPIPE
    if _MEDIAPIPE is not None:
        return _MEDIAPIPE
    import mediapipe as mp  # type: ignore

    _MEDIAPIPE = mp
    return mp


# -- Stage 4–6: video re-encode to 30 fps, 256×256, then frame extraction.


def _ffmpeg_normalize(src: Path, dst: Path, fps: int = 30, size: int = 256) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        f"scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size},fps={fps}",
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def _sample_frames(video_path: Path, num_frames: int = TEMPORAL_LENGTH) -> list[np.ndarray]:
    """Sample `num_frames` BGR frames evenly across the clip via OpenCV."""
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, total - 1, num=num_frames, dtype=int)
    frames: list[np.ndarray] = []
    last_idx = -1
    cur = None
    for target in idxs:
        if target != last_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
            ok, cur = cap.read()
            last_idx = target
            if not ok:
                cur = None
        if cur is not None:
            frames.append(cur.copy())
    cap.release()
    return frames


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


def _extract_keypoints_for_clip(frames: list[np.ndarray]) -> tuple[np.ndarray, int]:
    """Run MediaPipe Holistic per frame, return (T, K) array + miss count."""
    mp = _lazy_mediapipe()
    out = np.zeros((TEMPORAL_LENGTH, TOTAL_COORDS), dtype=np.float32)
    miss = 0

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
    ) as holistic:
        for t, bgr in enumerate(frames[:TEMPORAL_LENGTH]):
            import cv2  # type: ignore

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            row = out[t]

            def _write_hand(landmarks, offset: int) -> bool:
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
                return True

            has_left = _write_hand(res.left_hand_landmarks, LEFT_HAND_OFFSET)
            has_right = _write_hand(res.right_hand_landmarks, RIGHT_HAND_OFFSET)
            if not (has_left or has_right):
                miss += 1

            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                for i, mp_idx in enumerate(POSE_INDEX_SUBSET):
                    base = POSE_OFFSET + i * COORDS_PER_LANDMARK
                    if mp_idx < len(lms):
                        row[base] = lms[mp_idx].x
                        row[base + 1] = lms[mp_idx].y
                        row[base + 2] = lms[mp_idx].z

    return out, miss


# -- Stage 9: signer-disjoint split assignment.


def _hash_str(s: str, n_buckets: int) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % n_buckets


def _assign_split(source_signer_key: str, seed: int) -> str:
    """Stable signer-disjoint split. 70/15/15 by signer."""
    # Mix the seed into the hash so the split can be regenerated
    # with a different seed if needed.
    key = f"{seed}:{source_signer_key}"
    bucket = _hash_str(key, 100)
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


# -- Main pipeline


def clean(
    raw_manifest: Path,
    filter_path: Path,
    output_dir: Path,
    version: str,
    seed: int,
    skip_normalize: bool,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    with raw_manifest.open() as f:
        raw = json.load(f)
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

    clip_records: list[dict[str, Any]] = []
    seen_phash: dict[str, set[str]] = defaultdict(set)  # signer_key → set of phashes

    for rec in raw["records"]:
        if rec["download_status"] != "ok" or not rec["local_path"]:
            continue
        if rec["sign_id"] not in kept_signs:
            continue
        src_path = Path(rec["local_path"])
        if not src_path.exists():
            log.warning("missing local file: %s", src_path)
            continue

        clip_id = f"{rec['source']}_{rec['wlasl_video_id']}"
        dst_video = norm_dir / f"{clip_id}.mp4"

        if not skip_normalize:
            try:
                _ffmpeg_normalize(src_path, dst_video)
            except subprocess.CalledProcessError as e:
                log.warning("ffmpeg failed for %s: %s", clip_id, e)
                continue

        frames = _sample_frames(dst_video if dst_video.exists() else src_path)
        if len(frames) < TEMPORAL_LENGTH:
            log.warning("only %d frames for %s, padding with last", len(frames), clip_id)
            while len(frames) < TEMPORAL_LENGTH and frames:
                frames.append(frames[-1])
        if not frames:
            log.warning("no frames extracted for %s", clip_id)
            continue

        # Dedup within source signer.
        signer_key = rec["source_signer_id"] or f"{rec['source']}_{rec['wlasl_video_id']}"
        ph = _phash(frames[len(frames) // 2])
        if ph in seen_phash[signer_key]:
            log.info("dedup skip: %s (phash collision)", clip_id)
            continue
        seen_phash[signer_key].add(ph)

        keypoints, miss = _extract_keypoints_for_clip(frames)
        if miss / TEMPORAL_LENGTH > 0.5:
            log.info("mediapipe missed too many frames for %s (%d/%d)", clip_id, miss, TEMPORAL_LENGTH)
            # Still record so the validation report can measure detection success.

        kp_path = kp_dir / f"{clip_id}.npy"
        np.save(kp_path, keypoints)

        clip_records.append(
            {
                "clip_id": clip_id,
                "sign_id": rec["sign_id"],
                "source": rec["source"],
                "source_video_id": rec["wlasl_video_id"],
                "source_signer_id": rec["source_signer_id"],
                "source_split": rec["source_split"],
                "split": _assign_split(signer_key, seed),
                "keypoint_path": str(kp_path),
                "normalized_video_path": str(dst_video) if dst_video.exists() else None,
                "mediapipe_misses": miss,
            }
        )

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
        },
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
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--filter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-normalize",
        action="store_true",
        help="Skip ffmpeg re-encode (assume source already at 30 fps / 256). Useful for re-runs.",
    )
    args = parser.parse_args()
    clean(args.raw_manifest, args.filter, args.output, args.version, args.seed, args.skip_normalize)
    return 0


if __name__ == "__main__":
    sys.exit(main())
