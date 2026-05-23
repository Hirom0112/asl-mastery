"""Run the pose-ablation experiment.

For each annotated clip:
  1. Load the original trajectory JSON.
  2. Override pose[0:4] (nose, neck, r_sh, l_sh) at annotated frames with
     hand-labeled values. Linearly interpolate between annotated frames
     for the unannotated ones. Leave pose[4:8] (elbows, wrists) as the
     v0 outputs — we don't have hand labels for them.
  3. Featurize and run the matcher both ways.
  4. Compare per-clip and aggregate metrics.

The test: if the override produces a big lift in top-1 accuracy or
threshold-pass rate, pose IS the matcher ceiling and a pose retrain is
the right next investment. If not, the matcher itself is the ceiling
and a learned head (Slice 4.4) is the right call.

This is a directional, not definitive, signal — N=25 clips.
"""
from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np

TIME_STEPS = 32
from training.detectors.sign_matcher import (
    load_templates,
    mahalanobis_score,
    predict,
    trajectory_from_frames,
)


KP_NAMES = ["nose", "neck", "r_shoulder", "l_shoulder"]


def interpolate_overrides(
    annotated: dict[int, dict],
    all_frame_idxs: list[int],
) -> dict[int, list[list[float]]]:
    """Build {frame_idx: [4x [x,y]]} for every frame, linearly interpolated
    between annotated frames. Frames outside the annotated range use the
    nearest annotated frame (clamp)."""
    if not annotated:
        return {}
    sorted_idxs = sorted(annotated.keys())
    out: dict[int, list[list[float]]] = {}
    for fi in all_frame_idxs:
        if fi in annotated:
            out[fi] = [annotated[fi][kp] for kp in KP_NAMES]
            continue
        # find bracketing annotated frames
        lo = max([i for i in sorted_idxs if i <= fi], default=None)
        hi = min([i for i in sorted_idxs if i >= fi], default=None)
        if lo is None:
            out[fi] = [annotated[hi][kp] for kp in KP_NAMES]
        elif hi is None:
            out[fi] = [annotated[lo][kp] for kp in KP_NAMES]
        elif lo == hi:
            out[fi] = [annotated[lo][kp] for kp in KP_NAMES]
        else:
            t = (fi - lo) / (hi - lo)
            out[fi] = [
                [
                    annotated[lo][kp][0] * (1 - t) + annotated[hi][kp][0] * t,
                    annotated[lo][kp][1] * (1 - t) + annotated[hi][kp][1] * t,
                ]
                for kp in KP_NAMES
            ]
    return out


def make_overridden_trajectory(traj: dict, annotated_str_keys: dict[str, dict]) -> dict:
    """Return a deep-copied trajectory with pose[0:4] replaced at every
    frame using interpolated annotations. Pose[4:8] left untouched."""
    annotated = {int(k): v for k, v in annotated_str_keys.items()}
    out = deepcopy(traj)
    frame_idxs = [f["frame_idx"] for f in out["frames"]]
    overrides = interpolate_overrides(annotated, frame_idxs)
    for f in out["frames"]:
        fi = f["frame_idx"]
        if fi not in overrides:
            continue
        pose = f.get("pose")
        if not pose or len(pose) < 8:
            # pad to 8 kpts; missing tail filled with [0,0]
            pose = list(pose or []) + [[0.0, 0.0]] * (8 - len(pose or []))
        for i in range(4):
            pose[i] = overrides[fi][i]
        f["pose"] = pose
    return out


def per_clip_metrics(
    traj: dict,
    templates: dict,
    thresholds: dict,
) -> dict:
    feats = trajectory_from_frames(traj["frames"], TIME_STEPS)
    best_sign, conf, scores = predict(feats, templates)
    true_sign = traj.get("sign_id")
    true_score = scores.get(true_sign)
    true_threshold = (thresholds.get(true_sign) or {}).get("threshold")
    passes_threshold = (
        true_score is not None
        and true_threshold is not None
        and true_score < true_threshold
    )
    return {
        "true_sign": true_sign,
        "top1_sign": best_sign,
        "top1_correct": best_sign == true_sign,
        "top1_conf": conf,
        "true_score": true_score,
        "true_threshold": true_threshold,
        "passes_threshold": passes_threshold,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", type=Path,
                    default=Path("data/ablation/annotations.json"))
    ap.add_argument("--val-dir", type=Path,
                    default=Path("data/trajectories_v6_val"))
    ap.add_argument("--templates-dir", type=Path,
                    default=Path("data/templates_v7"))
    ap.add_argument("--thresholds", type=Path,
                    default=Path("data/templates_v7/_thresholds.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/ablation/results.json"))
    args = ap.parse_args()

    annotations = json.loads(args.annotations.read_text())
    templates = load_templates(args.templates_dir)
    thresholds = json.loads(args.thresholds.read_text())
    print(f"loaded {len(templates)} templates, {len(thresholds)} thresholds, "
          f"{len(annotations)} annotated clips")

    results = []
    for clip_key, ann in annotations.items():
        if not ann:
            print(f"  {clip_key}: no annotated frames, skip")
            continue
        sign, fname = clip_key.split("/", 1)
        traj_path = args.val_dir / sign / fname
        if not traj_path.exists():
            print(f"  {clip_key}: trajectory missing, skip")
            continue
        traj_orig = json.loads(traj_path.read_text())
        traj_over = make_overridden_trajectory(traj_orig, ann)
        m_orig = per_clip_metrics(traj_orig, templates, thresholds)
        m_over = per_clip_metrics(traj_over, templates, thresholds)
        results.append({
            "clip_key": clip_key,
            "n_annotated_frames": len(ann),
            "original": m_orig,
            "overridden": m_over,
        })

    # Aggregate
    n = len(results)
    if n == 0:
        print("no usable annotations — run annotate_pose_ablation.py first")
        return 1

    orig_top1 = sum(r["original"]["top1_correct"] for r in results)
    over_top1 = sum(r["overridden"]["top1_correct"] for r in results)
    orig_pass = sum(r["original"]["passes_threshold"] for r in results)
    over_pass = sum(r["overridden"]["passes_threshold"] for r in results)

    # Per-clip table
    print()
    print(f"{'clip':<32} {'true':<12} {'orig→top1':<14} {'over→top1':<14}  "
          f"{'orig_pass':>10}  {'over_pass':>10}")
    print("-" * 100)
    for r in results:
        ok_o = "✓" if r["original"]["top1_correct"] else "·"
        ok_v = "✓" if r["overridden"]["top1_correct"] else "·"
        pp_o = "✓" if r["original"]["passes_threshold"] else "·"
        pp_v = "✓" if r["overridden"]["passes_threshold"] else "·"
        delta_marker = ""
        if r["overridden"]["top1_correct"] and not r["original"]["top1_correct"]:
            delta_marker = "  ← LIFT"
        elif r["original"]["top1_correct"] and not r["overridden"]["top1_correct"]:
            delta_marker = "  ← REGRESS"
        print(f"{r['clip_key']:<32} {r['original']['true_sign']:<12} "
              f"{ok_o} {r['original']['top1_sign']:<10} "
              f"{ok_v} {r['overridden']['top1_sign']:<10}  "
              f"{pp_o:>10}  {pp_v:>10}{delta_marker}")

    print()
    print("=" * 60)
    print("AGGREGATE")
    print("=" * 60)
    print(f"  N clips:                       {n}")
    print(f"  Top-1 accuracy ORIGINAL:       {orig_top1}/{n}  ({100*orig_top1/n:.1f}%)")
    print(f"  Top-1 accuracy OVERRIDDEN:     {over_top1}/{n}  ({100*over_top1/n:.1f}%)")
    print(f"  Threshold pass ORIGINAL:       {orig_pass}/{n}  ({100*orig_pass/n:.1f}%)")
    print(f"  Threshold pass OVERRIDDEN:     {over_pass}/{n}  ({100*over_pass/n:.1f}%)")
    print()
    lift_top1 = over_top1 - orig_top1
    lift_pass = over_pass - orig_pass
    print(f"  Lift (top-1):                  {lift_top1:+d}")
    print(f"  Lift (threshold pass):         {lift_pass:+d}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_clips": n,
        "original": {
            "top1_correct": orig_top1,
            "threshold_pass": orig_pass,
        },
        "overridden": {
            "top1_correct": over_top1,
            "threshold_pass": over_pass,
        },
        "lift_top1": lift_top1,
        "lift_pass": lift_pass,
        "per_clip": results,
    }, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
