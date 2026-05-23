"""P1 — landmark-quality diagnostic, free portion (FREE / local).

Measures the binding-constraint signal directly from the extracted v8
trajectories, per source: how often the from-scratch detectors actually
produced hands. This validates / quantifies the "~60% missing-hands on
ASL Citizen" claim and tells us where detector quality (the real lever,
P3) is failing before we spend a cent on retraining.

What it CANNOT do here: per-keypoint px error vs ground truth — that needs
the detector checkpoints + raw frames + GT labels (that's P1b, a GPU/local
detector run). This script reports detection *coverage*, which is free and
already decisive for source triage.

Usage:
    training/.venv/bin/python -m scripts.p1_landmark_diag \
        --traj-root data/trajectories_v8_pull/trajectories_v8 \
        --manifest  data/labeled_frames/unified_clip_manifest_modal_v4.json \
        --out runs/p1_landmark_diag/coverage.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def build_clip_source(manifest_path: Path) -> dict[str, str]:
    raw = json.loads(manifest_path.read_text())
    recs = raw if isinstance(raw, list) else next(
        v for v in raw.values() if isinstance(v, list))
    return {r["clip_path"]: r.get("source") for r in recs if r.get("clip_path")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/p1_landmark_diag/coverage.json"))
    args = ap.parse_args()

    clip_src = build_clip_source(args.manifest)

    # per-source accumulators
    st = defaultdict(lambda: {
        "clips": 0, "frames": 0,
        "frames_0_hands": 0, "frames_1_hand": 0, "frames_2plus_hands": 0,
        "hands_total": 0,
        "clips_any_empty_frame": 0, "clips_all_frames_empty": 0,
        "clips_no_pose": 0,
    })

    for sign_dir in sorted(args.traj_root.iterdir()):
        if not sign_dir.is_dir():
            continue
        for j in sign_dir.glob("*.json"):
            try:
                t = json.loads(j.read_text())
            except Exception:
                continue
            src = clip_src.get(t.get("clip_path"), "UNKNOWN")
            s = st[src]
            frames = t.get("frames", [])
            s["clips"] += 1
            empty_frames = 0
            pose_frames = 0
            for f in frames:
                hands = f.get("hands") or []
                nh = len(hands)
                s["frames"] += 1
                s["hands_total"] += nh
                if nh == 0:
                    s["frames_0_hands"] += 1
                    empty_frames += 1
                elif nh == 1:
                    s["frames_1_hand"] += 1
                else:
                    s["frames_2plus_hands"] += 1
                if f.get("pose"):
                    pose_frames += 1
            if empty_frames > 0:
                s["clips_any_empty_frame"] += 1
            if frames and empty_frames == len(frames):
                s["clips_all_frames_empty"] += 1
            if pose_frames == 0:
                s["clips_no_pose"] += 1

    report = {}
    for src, s in sorted(st.items()):
        fr = max(s["frames"], 1)
        cl = max(s["clips"], 1)
        report[src] = {
            "clips": s["clips"],
            "frames": s["frames"],
            "avg_hands_per_frame": round(s["hands_total"] / fr, 3),
            "pct_frames_0_hands": round(100 * s["frames_0_hands"] / fr, 1),
            "pct_frames_1_hand": round(100 * s["frames_1_hand"] / fr, 1),
            "pct_frames_2plus_hands": round(100 * s["frames_2plus_hands"] / fr, 1),
            "pct_clips_with_any_empty_frame": round(100 * s["clips_any_empty_frame"] / cl, 1),
            "pct_clips_all_frames_empty": round(100 * s["clips_all_frames_empty"] / cl, 1),
            "pct_clips_no_pose": round(100 * s["clips_no_pose"] / cl, 1),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print("=== P1 detection-coverage by source (from-scratch detector health) ===\n")
    hdr = f"{'source':<14}{'clips':>7}{'hands/fr':>10}{'%0-hand fr':>12}{'%2+ fr':>9}{'%clips any-empty':>18}{'%clips no-pose':>16}"
    print(hdr)
    print("-" * len(hdr))
    for src, r in report.items():
        print(f"{src:<14}{r['clips']:>7}{r['avg_hands_per_frame']:>10}"
              f"{r['pct_frames_0_hands']:>12}{r['pct_frames_2plus_hands']:>9}"
              f"{r['pct_clips_with_any_empty_frame']:>18}{r['pct_clips_no_pose']:>16}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
