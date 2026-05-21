"""Cleaning pipeline (post-ADR 0010, no MediaPipe).

Phase 3f. Single command: reads raw clips from `dataset/raw/`,
runs the non-extraction stages from `docs/ARCHITECTURE.md` §2.5
and `docs/DATASET.md` §3, writes versioned per-clip MP4 outputs
to `dataset/clean/v<N>/normalized_videos/`, and emits the manifest
the training-time MP4 dataset loader consumes (written in T4 as
`training/classifier/dataset_video.py`).

Stages (under ADR 0010 + ADR 0005):
  1. Ingest raw clips (already on disk from Phase 3d ingestion).
  2. Per-sign clip-count filter (ADR 0008) — already encoded in
     the vocabulary filter JSON.
  3. Sign-window trim (from WLASL frame_start/frame_end, ASL Citizen
     split CSVs, Sem-Lex metadata; clips without an annotation use
     the full duration).
  4. Frame-rate normalization to 30 fps.
  5. Resize to 256×256 (center-square crop → bilinear). Training-time
     augmentation crops down to the model's H×W with offset jitter.
  6. Per-clip MP4 write to `normalized_videos/<clip_id>.mp4`.
  7. pHash dedup within a source-signer cohort.
  8. Stratified signer-disjoint split assignment.
  9. Manifest write.

The MediaPipe Holistic keypoint-extraction stage that ran under the
now-superseded ADR 0006 is removed. There is no `.npy` output. The
training-time loader reads the MP4s directly.

Usage:

    python -m training.data.clean \\
        --raw-manifest dataset/raw/wlasl_manifest.json \\
                       dataset/raw/asl_citizen_manifest.json \\
                       dataset/raw/sem_lex_manifest.json \\
        --filter dataset/slice1b_vocabulary.json \\
        --output dataset/clean/v3/ \\
        --version v3 \\
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

# T = 16 frames over the 2-second capture window. Kept inline (was
# previously `training.keypoints.TEMPORAL_LENGTH`, deleted under
# ADR 0010 with the rest of the keypoint plumbing).
TEMPORAL_LENGTH = 16
NORMALIZED_SIZE = 256
TARGET_FPS = 30

log = logging.getLogger("clean")


# -- Stage 4–6: ffmpeg normalize (trim → 30 fps → 256×256 → MP4) ----


def _ffmpeg_normalize(
    src: Path,
    dst: Path,
    *,
    frame_start: int | None,
    frame_end: int | None,
) -> None:
    """Normalize one source clip to a 30-fps, 256×256, libx264 MP4.

    Idempotent on the destination — caller checks `dst.exists()` and
    skips if so.

    Trim semantics: if `frame_start`/`frame_end` are provided (1-indexed
    per WLASL convention), we ask ffmpeg to trim to that range *first*,
    then resample / resize. WLASL frame numbers are at the source clip's
    native fps; we don't try to be clever about converting — ffmpeg's
    `select='between(n,...)'` is robust enough.
    """
    vf_filters: list[str] = []
    if frame_start is not None and frame_start > 1:
        # Frame numbers in WLASL are 1-indexed; ffmpeg's `between(n,a,b)`
        # is 0-indexed inclusive. fs - 1 = start frame index.
        fs = max(0, int(frame_start) - 1)
        if frame_end is not None and frame_end > 0:
            fe = int(frame_end) - 1
            vf_filters.append(f"select='between(n,{fs},{fe})',setpts=PTS-STARTPTS")
        else:
            vf_filters.append(f"select='gte(n,{fs})',setpts=PTS-STARTPTS")
    vf_filters.append(f"fps={TARGET_FPS}")
    # Center-square crop, then bilinear resize to NORMALIZED_SIZE².
    vf_filters.append(f"crop='min(iw,ih)':'min(iw,ih)'")
    vf_filters.append(f"scale={NORMALIZED_SIZE}:{NORMALIZED_SIZE}:flags=bilinear")
    vf = ",".join(vf_filters)

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
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


# -- Stage 7: pHash for intra-source-signer dedup --------------------


def _phash(path: Path) -> str:
    """Cheap perceptual hash of the first decoded frame of an MP4.

    A real per-clip hash would aggregate across frames; for dedup
    inside a single source signer, the first-frame hash is enough to
    catch exact-duplicate uploads / same-clip-twice ingest mistakes.
    """
    # Read one 32×32 grayscale frame via ffmpeg's image2pipe.
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vframes",
        "1",
        "-vf",
        "scale=32:32,format=gray",
        "-f",
        "rawvideo",
        "-",
    ]
    out = subprocess.run(cmd, check=True, capture_output=True).stdout
    if len(out) != 32 * 32:
        # Fall back to a sha of the raw bytes if the frame extract
        # came back malformed (e.g. zero-length output on a broken MP4).
        return hashlib.sha256(out).hexdigest()[:16]
    arr = np.frombuffer(out, dtype=np.uint8).reshape(32, 32)
    avg = arr.mean()
    bits = (arr > avg).astype(np.uint8).flatten()
    # Pack the 1024-bit signature into a 256-char hex string.
    packed = np.packbits(bits)
    return packed.tobytes().hex()


# -- Stage 8: signer-disjoint stratified split assignment ------------


def _assign_splits_stratified(
    records: list[dict[str, Any]],
    seed: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> None:
    """Assign each record a `split` ∈ {train, val, test}.

    Signer-disjoint per-sign: clips from the same source-signer for a
    given sign cluster on the same side of the split. Stratified per
    sign so each sign appears in train, val, AND test (even at small
    n the test row gets at least one clip if available).
    """
    rng = random.Random(seed)
    by_sign: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_sign[r["sign_id"]].append(r)

    for sign_id, clips in by_sign.items():
        # Group by signer; clips without a signer id are each their own
        # "signer" (conservative — won't accidentally leak).
        by_signer: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for c in clips:
            sid = c.get("source_signer_id") or f"_anon_{c['clip_id']}"
            by_signer[sid].append(c)

        signer_ids = list(by_signer.keys())
        rng.shuffle(signer_ids)
        # Walk signers in shuffled order, accumulating clips per split
        # until each side hits its share of the per-sign total.
        n = len(clips)
        n_train_target = max(1, int(n * train_frac))
        n_val_target = max(1, int(n * val_frac))
        n_train = n_val = 0
        for sid in signer_ids:
            sclips = by_signer[sid]
            if n_train < n_train_target:
                bucket = "train"
                n_train += len(sclips)
            elif n_val < n_val_target:
                bucket = "val"
                n_val += len(sclips)
            else:
                bucket = "test"
            for c in sclips:
                c["split"] = bucket


# -- Per-clip worker (top-level so it pickles for ProcessPoolExecutor)


def _process_one_clip(rec: dict[str, Any], norm_dir: str) -> dict[str, Any]:
    """Normalize one raw record to a per-clip MP4 in `norm_dir`.

    Returns one of:
      - {"kind": "ok", "record": <clip_record_dict>}
      - {"kind": "skip", "reason": <str>}
    """
    src_path = Path(rec["local_path"])
    if not src_path.exists():
        return {"kind": "skip", "reason": "missing-local"}

    clip_id = src_path.stem
    dst_video = Path(norm_dir) / f"{clip_id}.mp4"
    frame_start = rec.get("frame_start")
    frame_end = rec.get("frame_end")

    if not dst_video.exists():
        try:
            _ffmpeg_normalize(src_path, dst_video, frame_start=frame_start, frame_end=frame_end)
        except subprocess.CalledProcessError:
            return {"kind": "skip", "reason": "ffmpeg-failed"}

    try:
        phash = _phash(dst_video)
    except subprocess.CalledProcessError:
        return {"kind": "skip", "reason": "phash-failed"}

    return {
        "kind": "ok",
        "record": {
            "clip_id": clip_id,
            "sign_id": rec["sign_id"],
            "source": rec["source"],
            "source_video_id": rec.get("wlasl_video_id") or rec.get("source_video_id"),
            "source_signer_id": rec.get("source_signer_id"),
            "source_split": rec.get("source_split"),
            "split": "pending",
            "normalized_video_path": str(dst_video),
            "phash": phash,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "sign_window_trimmed": bool(
                (isinstance(frame_start, int) and frame_start > 1)
                or (isinstance(frame_end, int) and frame_end > 0)
            ),
        },
    }


# -- Main pipeline ---------------------------------------------------


def _merge_raw_manifests(paths: list[Path]) -> dict:
    all_records: list[dict[str, Any]] = []
    for p in paths:
        with p.open() as f:
            m = json.load(f)
        all_records.extend(m.get("records", []))
    return {"records": all_records}


def _dedup_within_signer(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop pHash duplicates within the same source-signer cohort.

    Returns (kept_records, dropped_count).
    """
    seen: set[tuple[str, str]] = set()
    kept: list[dict[str, Any]] = []
    dropped = 0
    for r in records:
        key = (r.get("source_signer_id") or f"_anon_{r['clip_id']}", r["phash"])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(r)
    return kept, dropped


def clean(
    raw_manifests: list[Path],
    filter_path: Path,
    output_dir: Path,
    *,
    version: str,
    seed: int = 42,
    workers: int = 1,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    raw = _merge_raw_manifests(raw_manifests)
    with filter_path.open() as f:
        vocab_filter = json.load(f)
    # slice1b_vocabulary.json (and friends from filter_vocabulary.py)
    # carries `kept_signs: [{sign_id, gloss, ...}]`. Older filters may
    # use `included_signs: [sign_id, ...]` or a flat {sign_id: count}
    # dict — accept all three for robustness.
    allowed_signs: set[str] = set()
    if isinstance(vocab_filter, dict):
        if "kept_signs" in vocab_filter:
            allowed_signs = {entry["sign_id"] for entry in vocab_filter["kept_signs"]}
        elif "included_signs" in vocab_filter:
            allowed_signs = set(vocab_filter["included_signs"])
        else:
            allowed_signs = set(vocab_filter.keys())
    log.info("vocabulary filter: %d signs allowed", len(allowed_signs))

    filtered = [r for r in raw["records"] if r["sign_id"] in allowed_signs]
    log.info("post-filter records: %d (of %d raw)", len(filtered), len(raw["records"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    norm_dir = output_dir / "normalized_videos"
    norm_dir.mkdir(exist_ok=True)

    # Normalize each clip in parallel.
    ok_records: list[dict[str, Any]] = []
    skipped = defaultdict(int)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_process_one_clip, r, str(norm_dir)) for r in filtered]
            for fut in as_completed(futures):
                res = fut.result()
                if res["kind"] == "ok":
                    ok_records.append(res["record"])
                else:
                    skipped[res["reason"]] += 1
    else:
        for r in filtered:
            res = _process_one_clip(r, str(norm_dir))
            if res["kind"] == "ok":
                ok_records.append(res["record"])
            else:
                skipped[res["reason"]] += 1
    log.info("normalized %d clips (skipped: %s)", len(ok_records), dict(skipped))

    # Dedup within source-signer cohorts.
    ok_records, dropped_dups = _dedup_within_signer(ok_records)
    log.info("dropped %d pHash duplicates within source-signer cohorts", dropped_dups)

    # Assign signer-disjoint stratified splits.
    _assign_splits_stratified(ok_records, seed=seed)

    # Manifest write.
    classes = sorted({r["sign_id"] for r in ok_records})
    manifest = {
        "version": version,
        "temporal_length": TEMPORAL_LENGTH,
        "normalized_size": NORMALIZED_SIZE,
        "target_fps": TARGET_FPS,
        "num_classes": len(classes),
        "classes": classes,
        "split_counts": {
            split: sum(1 for r in ok_records if r["split"] == split) for split in ("train", "val", "test")
        },
        "skipped": dict(skipped),
        "dropped_dups": dropped_dups,
        "records": ok_records,
    }
    manifest_path = output_dir / f"dataset_{version}_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    log.info("wrote manifest → %s (%d records, %d classes)", manifest_path, len(ok_records), len(classes))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", type=Path, nargs="+", required=True, dest="raw_manifests")
    parser.add_argument("--filter", type=Path, required=True, dest="filter_path")
    parser.add_argument("--output", type=Path, required=True, dest="output_dir")
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    clean(
        args.raw_manifests,
        args.filter_path,
        args.output_dir,
        version=args.version,
        seed=args.seed,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
