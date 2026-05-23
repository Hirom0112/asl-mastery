"""Phase 4.6 Stage 1 — dump 224×224 hand-crop JPGs from v7 trajectories.

For each trajectory JSON, opens the source video once, decodes at the
trajectory's fps, crops each per-frame hand bbox (with 15% pad), resizes
to 224×224, writes JPG to data/hand_crops/<clip_id>/{frame:04d}_h{idx}.jpg.

Also writes data/hand_crops/index.json — a flat list of crop records:
    {crop_path, clip_id, sign_id, source, frame_idx, hand_idx}

The index is used downstream for split-integrity (all crops from one
clip_id stay in the same encoder train/val split) and to keep contrastive
augmentations free of self-pairings within an item.

Usage:
    training/.venv/bin/python -m scripts.dump_hand_crops \\
        --trajectories-root data/trajectories_v7_pull/trajectories_v7 \\
        --out-root data/hand_crops \\
        --workers 12

Expected: ~225K crops in 30-60 min on a 10+ core Mac.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


log = logging.getLogger("dump_crops")

# Modal volume path → local path mapping for v7 trajectories
PATH_MAP = [
    ("/data/datasets/sem_lex_clips/clips/", "dataset/raw/sem_lex/clips/"),
    ("/data/datasets/asl_clips/v2-yt/normalized_videos/",
     "dataset/clean/v2-yt/normalized_videos/"),
]


def _resolve_local_path(modal_path: str, repo_root: Path) -> Optional[Path]:
    for prefix, local_prefix in PATH_MAP:
        if modal_path.startswith(prefix):
            tail = modal_path[len(prefix):]
            return repo_root / local_prefix / tail
    return None


def _pad_bbox(x0: float, y0: float, x1: float, y1: float,
              w: int, h: int, pad_frac: float = 0.15) -> tuple[int, int, int, int]:
    bw = x1 - x0
    bh = y1 - y0
    pad = pad_frac * max(bw, bh)
    nx0 = max(0, int(round(x0 - pad)))
    ny0 = max(0, int(round(y0 - pad)))
    nx1 = min(w, int(round(x1 + pad)))
    ny1 = min(h, int(round(y1 + pad)))
    return nx0, ny0, nx1, ny1


def _process_trajectory(args: tuple) -> dict:
    """Worker: process ONE trajectory JSON. Open source video once, dump all
    hand crops for it. Returns summary dict for the index."""
    traj_path_str, out_root_str, repo_root_str = args
    traj_path = Path(traj_path_str)
    out_root = Path(out_root_str)
    repo_root = Path(repo_root_str)

    try:
        traj = json.loads(traj_path.read_text())
    except Exception as e:
        return {"clip_id": traj_path.stem, "status": "json-error", "err": repr(e), "n_crops": 0, "records": []}

    clip_path_str = traj.get("clip_path", "")
    local_path = _resolve_local_path(clip_path_str, repo_root)
    if local_path is None or not local_path.exists():
        return {"clip_id": traj_path.stem, "status": "source-missing",
                "src": clip_path_str, "n_crops": 0, "records": []}

    sign_id = traj["sign_id"]
    fps = float(traj.get("fps", 15.0))
    clip_id = traj_path.stem
    out_dir = out_root / sign_id / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open source video once
    cap = cv2.VideoCapture(str(local_path))
    if not cap.isOpened():
        return {"clip_id": clip_id, "status": "video-open-failed", "n_crops": 0, "records": []}
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Build the frame indices we need by mapping trajectory frame_idx (which
    # was sampled at `fps`) back to source-video frames.
    needed_frames: dict[int, list[tuple[int, list[float]]]] = {}
    # ^ src_frame_idx → list of (hand_idx, bbox)
    for f in traj.get("frames", []):
        if not f.get("hands"):
            continue
        traj_frame_idx = int(f["frame_idx"])
        src_frame_idx = int(round(traj_frame_idx * src_fps / fps))
        per_hand = []
        for hi, hand in enumerate(f["hands"]):
            bbox = hand.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            per_hand.append((hi, bbox))
        if per_hand:
            needed_frames[src_frame_idx] = per_hand

    if not needed_frames:
        cap.release()
        return {"clip_id": clip_id, "status": "no-hands", "n_crops": 0, "records": []}

    records: list[dict] = []
    target = sorted(needed_frames.keys())
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sequential decode + skip: cheaper than per-frame seek for short clips
    cur = 0
    ti = 0
    while ti < len(target):
        want = target[ti]
        # Skip-read to `want`
        while cur < want:
            ok = cap.grab()
            if not ok:
                break
            cur += 1
        if cur != want:
            break
        ret, frame = cap.retrieve()
        cur += 1
        if not ret or frame is None:
            ti += 1
            continue
        h, w = frame.shape[:2]
        # The trajectory's bbox coords are in the trajectory's frame_size
        # space. For v7 these match the source video (extract_trajectories_v2
        # doesn't resize). Cross-check via frame_size if available.
        traj_w, traj_h = (traj.get("frame_size") or [w, h])
        sx = w / max(traj_w, 1)
        sy = h / max(traj_h, 1)
        for hi, bbox in needed_frames[want]:
            x0, y0, x1, y1 = bbox
            x0 *= sx; x1 *= sx; y0 *= sy; y1 *= sy
            nx0, ny0, nx1, ny1 = _pad_bbox(x0, y0, x1, y1, w, h)
            if nx1 - nx0 < 8 or ny1 - ny0 < 8:
                continue
            crop = frame[ny0:ny1, nx0:nx1]
            crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
            out_path = out_dir / f"{want:04d}_h{hi}.jpg"
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            records.append({
                "crop_path": str(out_path.relative_to(out_root.parent.parent)) if out_root.is_absolute() else str(out_path),
                "clip_id": clip_id,
                "sign_id": sign_id,
                "src_frame_idx": want,
                "hand_idx": hi,
            })
        ti += 1
    cap.release()
    return {"clip_id": clip_id, "status": "ok", "n_crops": len(records), "records": records}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories-root", type=Path,
                    default=Path("data/trajectories_v7_pull/trajectories_v7"))
    ap.add_argument("--out-root", type=Path, default=Path("data/hand_crops"))
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="Process only first N trajectories (debug)")
    args = ap.parse_args()

    if not args.trajectories_root.exists():
        raise SystemExit(f"trajectories root not found: {args.trajectories_root}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    traj_files = sorted(args.trajectories_root.rglob("*.json"))
    if args.limit:
        traj_files = traj_files[:args.limit]
    log.info("found %d trajectories", len(traj_files))

    work = [(str(p), str(args.out_root), str(args.repo_root)) for p in traj_files]
    t0 = time.time()
    all_records: list[dict] = []
    status_counts: dict[str, int] = {}
    n_crops_total = 0
    with mp.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_process_trajectory, work), 1):
            status_counts[res["status"]] = status_counts.get(res["status"], 0) + 1
            n_crops_total += res["n_crops"]
            all_records.extend(res["records"])
            if i % 200 == 0 or i == len(work):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 0.1)
                log.info("  %d/%d  status=%s  crops=%d  (%.1f traj/s, %.0fs elapsed)",
                         i, len(work), status_counts, n_crops_total, rate, elapsed)

    index = {
        "n_trajectories": len(traj_files),
        "n_crops": n_crops_total,
        "status_counts": status_counts,
        "records": all_records,
    }
    idx_path = args.out_root / "index.json"
    idx_path.write_text(json.dumps(index, indent=2))
    log.info("done: %d crops written, index at %s (%.0fs)",
             n_crops_total, idx_path, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
