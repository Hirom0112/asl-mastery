"""Validate the corrected pose inference (upper-body-bbox crop) against
the existing run_pose_ablation.py three-way comparison.

For each of the 25 annotated val clips:
  - load original v6 trajectory (broken full-frame pose)
  - re-run pose using corrected inference (upper-body bbox crop from the
    HAND BBOXES already stored in v6 — isolates the pose fix)
  - load hand-annotated pose (ground-truth override)
  - compute matcher top-1 + threshold pass for all three

Headline question: does corrected-pose lift match overridden-pose lift?
If yes, the inference fix recovers the bottleneck without needing retrain.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch

from training.detectors._pose_inference import (
    upper_body_bbox_from_hands,
)
from training.detectors.pose_detector import PoseRegressor
from training.detectors.sign_matcher import (
    load_templates,
    predict,
    trajectory_from_frames,
)
from scripts.run_pose_ablation import make_overridden_trajectory, per_clip_metrics

TIME_STEPS = 32
SEM_LEX_LOCAL = Path("dataset/raw/sem_lex/clips")


def load_pose(ckpt: Path, device: str) -> PoseRegressor:
    model = PoseRegressor()
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    sd = obj.get("model", obj.get("state_dict", obj))
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model.eval().to(device)


def find_local_clip(clip_path_in_traj: str) -> Path | None:
    p = Path(clip_path_in_traj)
    name = p.name
    cand = SEM_LEX_LOCAL / name
    return cand if cand.exists() else None


def decode_clip_frames(mp4: Path, target_fps: float = 15.0) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(mp4))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / target_fps)))
    frames = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % step == 0:
            frames.append(f)
        i += 1
    cap.release()
    return frames


@torch.no_grad()
def corrected_pose_per_frame(
    model: PoseRegressor,
    frames: list[np.ndarray],
    per_frame_hand_bboxes: list[list[tuple[float, float, float, float]]],
    device: str,
) -> list[list[list[float]]]:
    """Return pose keypoints per frame in image-pixel coords (or [] if no hands)."""
    out: list[list[list[float]]] = [[] for _ in frames]
    for fi, (bgr, hbs) in enumerate(zip(frames, per_frame_hand_bboxes)):
        if not hbs:
            continue
        H, W = bgr.shape[:2]
        bbox = upper_body_bbox_from_hands(hbs, W, H)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        crop = bgr[y0:y1, x0:x1]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        t = torch.nn.functional.interpolate(
            t, size=(256, 256), mode="bilinear", align_corners=False
        ).to(device)
        coords = model(t)["coords"][0].cpu().numpy()
        cw, ch = x1 - x0, y1 - y0
        out[fi] = [[float(x0 + kx * cw), float(y0 + ky * ch)] for kx, ky in coords]
    return out


def make_corrected_trajectory(traj: dict, corrected_poses: list[list[list[float]]]) -> dict:
    """Build a new trajectory with corrected pose values per frame."""
    out = deepcopy(traj)
    for i, f in enumerate(out["frames"]):
        if i < len(corrected_poses) and corrected_poses[i]:
            f["pose"] = corrected_poses[i]
        # else: leave original pose (or empty if we couldn't recompute)
    return out


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
    ap.add_argument("--pose-ckpt", type=Path,
                    default=Path("data/ckpts/pose_v0_best.pt"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/ablation/results_corrected.json"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    annotations = json.loads(args.annotations.read_text())
    templates = load_templates(args.templates_dir)
    thresholds = json.loads(args.thresholds.read_text())
    pose_model = load_pose(args.pose_ckpt, device)
    print(f"[device] {device}")
    print(f"[load]   {len(templates)} templates, {len(thresholds)} thresholds, "
          f"{len(annotations)} annotated clips")

    results = []
    skipped = []
    for clip_key, ann in annotations.items():
        if not ann:
            skipped.append((clip_key, "no annotated frames")); continue
        sign, fname = clip_key.split("/", 1)
        traj_path = args.val_dir / sign / fname
        if not traj_path.exists():
            skipped.append((clip_key, "trajectory missing")); continue
        traj_orig = json.loads(traj_path.read_text())
        mp4 = find_local_clip(traj_orig["clip_path"])
        if mp4 is None:
            skipped.append((clip_key, "local clip missing")); continue

        frames = decode_clip_frames(mp4)
        # Reuse the hand bboxes stored in the v6 trajectory
        per_frame_hand_bboxes = []
        for f in traj_orig["frames"]:
            hbs = [tuple(h["bbox"]) for h in (f.get("hands") or []) if h.get("bbox")]
            per_frame_hand_bboxes.append(hbs)
        # If frame counts diverge between decode and trajectory, truncate to min
        n = min(len(frames), len(per_frame_hand_bboxes))
        frames = frames[:n]
        per_frame_hand_bboxes = per_frame_hand_bboxes[:n]

        corrected = corrected_pose_per_frame(pose_model, frames, per_frame_hand_bboxes, device)
        # Pad to original trajectory length so trajectory_from_frames sees the same T
        while len(corrected) < len(traj_orig["frames"]):
            corrected.append([])

        traj_corr = make_corrected_trajectory(traj_orig, corrected)
        traj_over = make_overridden_trajectory(traj_orig, ann)

        m_orig = per_clip_metrics(traj_orig, templates, thresholds)
        m_corr = per_clip_metrics(traj_corr, templates, thresholds)
        m_over = per_clip_metrics(traj_over, templates, thresholds)
        results.append({
            "clip_key": clip_key,
            "original": m_orig,
            "corrected": m_corr,
            "overridden": m_over,
        })

    if not results:
        print("[err] no usable clips")
        for ck, why in skipped:
            print(f"  skipped {ck}: {why}")
        return 1

    n = len(results)
    orig_top1 = sum(r["original"]["top1_correct"] for r in results)
    corr_top1 = sum(r["corrected"]["top1_correct"] for r in results)
    over_top1 = sum(r["overridden"]["top1_correct"] for r in results)
    orig_pass = sum(r["original"]["passes_threshold"] for r in results)
    corr_pass = sum(r["corrected"]["passes_threshold"] for r in results)
    over_pass = sum(r["overridden"]["passes_threshold"] for r in results)

    print()
    print(f"{'clip':<32} {'true':<10} {'ORIG':<14} {'CORR':<14} {'OVER':<14}")
    print("-" * 90)
    for r in results:
        mark = lambda b: "✓" if b else "·"
        print(
            f"{r['clip_key']:<32} {r['original']['true_sign']:<10}  "
            f"{mark(r['original']['top1_correct'])} {r['original']['top1_sign']:<10} "
            f"{mark(r['corrected']['top1_correct'])} {r['corrected']['top1_sign']:<10} "
            f"{mark(r['overridden']['top1_correct'])} {r['overridden']['top1_sign']:<10}"
        )

    print()
    print("=" * 60)
    print(f"  N usable clips:        {n}  (skipped {len(skipped)})")
    print(f"  Top-1 ORIGINAL  (broken full-frame pose) : {orig_top1}/{n}  ({100*orig_top1/n:.1f}%)")
    print(f"  Top-1 CORRECTED (upper-body crop pose)   : {corr_top1}/{n}  ({100*corr_top1/n:.1f}%)")
    print(f"  Top-1 OVERRIDDEN (ground-truth pose)     : {over_top1}/{n}  ({100*over_top1/n:.1f}%)")
    print(f"  Thresh pass ORIGINAL :  {orig_pass}/{n}  ({100*orig_pass/n:.1f}%)")
    print(f"  Thresh pass CORRECTED:  {corr_pass}/{n}  ({100*corr_pass/n:.1f}%)")
    print(f"  Thresh pass OVERRIDDEN: {over_pass}/{n}  ({100*over_pass/n:.1f}%)")
    print(f"  Lift (top-1):    corrected={corr_top1-orig_top1:+d}  overridden={over_top1-orig_top1:+d}")
    print(f"  Lift (thresh):   corrected={corr_pass-orig_pass:+d}  overridden={over_pass-orig_pass:+d}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_clips": n,
        "skipped": skipped,
        "totals": {
            "original": {"top1": orig_top1, "pass": orig_pass},
            "corrected": {"top1": corr_top1, "pass": corr_pass},
            "overridden": {"top1": over_top1, "pass": over_pass},
        },
        "per_clip": results,
    }, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
