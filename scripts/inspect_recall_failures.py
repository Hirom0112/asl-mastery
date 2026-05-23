"""Dig into WHY the hand detector scores frames as '0 hands'.

A frame is '0 hands' exactly when the detector's peak confidence < threshold
(0.15). This dumps those failed frames as JPGs with the peak confidence and the
position-in-clip stamped on each, so we can SEE whether they are:
  - idle frames (hands down/off-screen at clip start/end) — legit no-hand,
  - near-misses (a hand is there but peak landed just under threshold),
  - true failures (a hand is plainly visible and the detector saw nothing).

Also prints, per source: where in the clip failures cluster, and a histogram of
the failure peak-confidence (near-miss vs truly empty).

Usage:
  training/.venv/bin/python -m scripts.inspect_recall_failures \
      --ckpt data/ckpts_new/hand_det_v2_best_ema.pt --clips-per-source 6
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from training.detectors.hand_detector import HandDetector
from scripts.live_demo import _load_pt, _frame_to_tensor

SOURCES = {
    "sem_lex": "dataset/raw/sem_lex/clips",
    "asl_citizen": "dataset/raw/asl_citizen/clips",
}
VIDEO_EXT = (".mp4", ".mov", ".webm", ".avi")
THRESH = 0.15


@torch.no_grad()
def peak_prob(detector, frame_chw) -> float:
    x = F.interpolate(frame_chw, size=(320, 320), mode="bilinear", align_corners=False)
    prob = torch.sigmoid(detector(x)["heatmap"])
    return float(prob.max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("data/ckpts_new/hand_det_v2_best_ema.pt"))
    ap.add_argument("--clips-per-source", type=int, default=6)
    ap.add_argument("--save-per-clip", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/recall_debug"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    random.seed(args.seed)
    detector = _load_pt(args.ckpt, HandDetector(), device)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"detector: {args.ckpt}  device: {device}\n", flush=True)

    for name, rel in SOURCES.items():
        root = Path(rel)
        clips = [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXT] if root.exists() else []
        if not clips:
            print(f"[{name}] no clips — skip", flush=True)
            continue
        random.shuffle(clips)
        clips = clips[:args.clips_per_source]
        sdir = args.out / name
        sdir.mkdir(parents=True, exist_ok=True)

        # failure-position buckets (start/mid/end third of clip) + conf histogram
        pos = {"start": 0, "mid": 0, "end": 0}
        conf_bins = {"empty<0.05": 0, "low0.05-0.10": 0, "nearmiss0.10-0.15": 0}
        n_fail = n_tot = 0

        for clip in clips:
            cap = cv2.VideoCapture(str(clip))
            frames = []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(fr)
            cap.release()
            if not frames:
                continue
            T = len(frames)
            saved = 0
            for i, fr in enumerate(frames):
                p = peak_prob(detector, _frame_to_tensor(fr, device))
                n_tot += 1
                if p >= THRESH:
                    continue
                n_fail += 1
                frac = i / max(1, T - 1)
                pos["start" if frac < 0.33 else "end" if frac > 0.67 else "mid"] += 1
                conf_bins["empty<0.05" if p < 0.05 else
                          "low0.05-0.10" if p < 0.10 else "nearmiss0.10-0.15"] += 1
                # save a spread of failed frames per clip for visual review
                if saved < args.save_per_clip and i % max(1, T // args.save_per_clip) == 0:
                    lab = fr.copy()
                    cv2.putText(lab, f"peak={p:.3f}  f{i}/{T}  {int(frac*100)}%",
                                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imwrite(str(sdir / f"{clip.stem}_f{i:03d}_p{int(p*1000):03d}.jpg"), lab)
                    saved += 1

        print(f"[{name}] {n_fail}/{n_tot} frames = 0-hand ({100*n_fail/max(1,n_tot):.1f}%)", flush=True)
        print(f"   where in clip: {pos}", flush=True)
        print(f"   failure peak-conf: {conf_bins}", flush=True)
        print(f"   saved sample frames → {sdir}\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
