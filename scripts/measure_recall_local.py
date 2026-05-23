"""Measure hand-DETECTOR RECALL on real ASL clips — locally, no Modal, no GPU.

For a random sample of clips per source, run the hand detector over frames and
report the hands-per-frame distribution: the same "% frames with 0 hands" that
runs/p1_landmark_diag/coverage.json reports, but on WHICHEVER checkpoint you
point at. Use it to compare the OLD vs NEW detector for free.

Uses the exact same detect_hands path as the live demo, so the numbers match
what you see on screen.

Usage:
  training/.venv/bin/python -m scripts.measure_recall_local \
      --ckpt data/ckpts_new/hand_det_v2_best_ema.pt \
      --clips-per-source 10 --frame-stride 5
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import torch

from training.detectors.hand_detector import HandDetector
from scripts.live_demo import _load_pt, _frame_to_tensor, detect_hands

SOURCES = {
    "sem_lex": "dataset/raw/sem_lex/clips",
    "asl_citizen": "dataset/raw/asl_citizen/clips",
    "wlasl": "dataset/raw/wlasl",
    "lifeprint": "dataset/raw/lifeprint",
}
VIDEO_EXT = (".mp4", ".mov", ".webm", ".avi")


def _list_clips(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXT]


@torch.no_grad()
def measure_source(detector, device, clips, frame_stride, max_frames=40) -> dict | None:
    n0 = nf = total_hands = 0
    aw_n0 = aw_nf = 0  # "clean data" = active signing window (idle trimmed)
    for clip in clips:
        cap = cv2.VideoCapture(str(clip))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if n_frames <= 0:
            cap.release()
            continue
        # SEEK to up-to-max_frames evenly-spaced indices and decode only those
        # (long tutorial videos otherwise force decoding every frame = slow).
        count = min(max_frames, max(1, n_frames // frame_stride))
        if count > 1:
            idxs = [int(j * (n_frames - 1) / (count - 1)) for j in range(count)]
        else:
            idxs = [0]
        presence = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            boxes, _ = detect_hands(detector, _frame_to_tensor(frame, device))
            k = len(boxes)
            nf += 1
            total_hands += k
            n0 += (k == 0)
            presence.append(k)
        # Active signing window: frames between first and last hand-bearing
        # frame (drops idle lead-in/out — the same cleaning drop_handless_frames
        # applies to the trajectories).
        with_hand = [i for i, k in enumerate(presence) if k > 0]
        if with_hand:
            for k in presence[with_hand[0]:with_hand[-1] + 1]:
                aw_nf += 1
                aw_n0 += (k == 0)
        cap.release()
    if nf == 0:
        return None
    return {
        "clips": len(clips),
        "frames_sampled": nf,
        "avg_hands_per_frame": round(total_hands / nf, 3),
        "pct_frames_0_hands": round(100 * n0 / nf, 1),
        "active_window_frames": aw_nf,
        "pct_active_window_0_hands": round(100 * aw_n0 / aw_nf, 1) if aw_nf else None,
    }


def _pick_device(arg: str | None) -> str:
    if arg:
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="data/ckpts_new/hand_det_v2_best_ema.pt,"
                    "data/ckpts/hand_det_v2_best_ema.pt",
                    help="comma-separated checkpoints to compare (new,old)")
    ap.add_argument("--clips-per-source", type=int, default=20)
    ap.add_argument("--frame-stride", type=int, default=4)
    ap.add_argument("--max-frames-per-clip", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/recall_local.json"))
    args = ap.parse_args()
    device = _pick_device(args.device)
    ckpts = [Path(c.strip()) for c in args.ckpts.split(",") if c.strip()]

    print(f"device: {device}  clips/source: {args.clips_per_source}  "
          f"stride: {args.frame_stride}\n", flush=True)

    # Same clip sample for every checkpoint (fair comparison).
    sampled: dict[str, list[Path]] = {}
    for name, rel in SOURCES.items():
        root = Path(rel)
        clips = _list_clips(root) if root.exists() else []
        random.seed(args.seed)
        random.shuffle(clips)
        sampled[name] = clips[:args.clips_per_source]

    results: dict[str, dict] = {}
    for ckpt in ckpts:
        if not ckpt.exists():
            print(f"!! missing ckpt {ckpt} — skip", flush=True)
            continue
        print(f"=== {ckpt} ===", flush=True)
        detector = _load_pt(ckpt, HandDetector(), device)
        results[str(ckpt)] = {}
        for name, clips in sampled.items():
            if not clips:
                print(f"  [{name}] no clips — skip", flush=True)
                continue
            res = measure_source(detector, device, clips, args.frame_stride,
                                 args.max_frames_per_clip)
            results[str(ckpt)][name] = res
            aw = res['pct_active_window_0_hands']
            print(f"  [{name:12s}] raw 0-hands={res['pct_frames_0_hands']:5.1f}%  "
                  f"-> CLEAN (signing window) 0-hands="
                  f"{(str(aw)+'%') if aw is not None else 'n/a':>6s}  "
                  f"(n={res['frames_sampled']})", flush=True)
        print(flush=True)

    # RAW (whole clip) vs CLEAN (active signing window) — 0-hand % per source.
    print("\n0-HAND % :  RAW whole-clip  vs  CLEAN signing-window  (lower = better)")
    for c in results:
        print(f"\n  detector: {Path(c).parent.name}")
        print(f"    {'source':14s}{'raw':>10s}{'clean':>10s}")
        print(f"    {'-'*34}")
        for name in SOURCES:
            r = results[c].get(name)
            if not r:
                continue
            aw = r['pct_active_window_0_hands']
            print(f"    {name:14s}{str(r['pct_frames_0_hands'])+'%':>10s}"
                  f"{(str(aw)+'%') if aw is not None else '-':>10s}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import json
    args.out.write_text(json.dumps({"device": device, "stride": args.frame_stride,
                                    "clips_per_source": args.clips_per_source,
                                    "results": results}, indent=2))
    print(f"\nsaved → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
