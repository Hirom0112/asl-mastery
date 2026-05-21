"""Phase 4 Slice 4.2 — fit per-sign trajectory templates.

For each sign in the frozen 75-sign vocabulary:
  - Gather all per-clip trajectories from data/trajectories/<sign>/*.json
  - Time-normalize each trajectory to T fixed frames (default 32)
  - Stack the (T, num_features) trajectories across clips
  - Compute mean trajectory + per-feature per-timestep diagonal variance
  - Save as data/templates/<sign>.npz

Features per frame (flattened):
  - 2 hands × 21 keypoints × 2 coords (relative to pose-anchor for translation invariance) = 84
  - 8 pose keypoints × 2 coords = 16
  - Total: 100 floats per frame
If a hand is missing in a frame, fill with NaN; statistics ignore NaN.

The resulting template is consumed by the sign matcher at inference:
Mahalanobis distance under the per-keypoint per-timestep variance gives
a per-sign similarity score. (Per-sign thresholds in Phase 4 Slice 4.3.)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


NUM_HAND_KP = 21
NUM_POSE_KP = 8
FEATURES_PER_FRAME = 2 * NUM_HAND_KP * 2 + NUM_POSE_KP * 2  # 100


def _pose_anchor(frame: dict) -> tuple[float, float] | None:
    """Return the (x, y) point we treat as origin for translation
    invariance — neck if visible, else mean of shoulders, else nose.
    """
    pose = frame.get("pose")
    if not pose or len(pose) < 4:
        return None
    nose, neck, r_sh, l_sh = pose[0], pose[1], pose[2], pose[3]
    if neck and not (neck[0] == 0 and neck[1] == 0):
        return tuple(neck)
    if r_sh and l_sh:
        return ((r_sh[0] + l_sh[0]) / 2, (r_sh[1] + l_sh[1]) / 2)
    if nose:
        return tuple(nose)
    return None


def _frame_to_features(frame: dict) -> np.ndarray:
    """Flatten one frame's hands+pose into the 100-d feature vector,
    with NaN where keypoints are missing. Anchors translation to the
    pose-derived origin.
    """
    feats = np.full(FEATURES_PER_FRAME, np.nan, dtype=np.float32)
    anchor = _pose_anchor(frame)
    if anchor is None:
        return feats
    ax, ay = anchor

    # Two hand slots (right=0, left=1) — but we don't know handedness
    # from the detector. Use bbox center x to disambiguate: hand with
    # smaller x → slot 0, larger x → slot 1.
    hands = frame.get("hands") or []
    if hands:
        hands_sorted = sorted(hands, key=lambda h: (h["bbox"][0] + h["bbox"][2]) / 2)
        for slot, hand in enumerate(hands_sorted[:2]):
            kps = hand["keypoints"]
            base = slot * NUM_HAND_KP * 2
            for i in range(NUM_HAND_KP):
                feats[base + i * 2 + 0] = kps[i][0] - ax
                feats[base + i * 2 + 1] = kps[i][1] - ay

    # Pose slot
    pose_base = 2 * NUM_HAND_KP * 2
    pose = frame.get("pose") or []
    for i in range(min(NUM_POSE_KP, len(pose))):
        feats[pose_base + i * 2 + 0] = pose[i][0] - ax
        feats[pose_base + i * 2 + 1] = pose[i][1] - ay
    return feats


def _resample_trajectory(traj: np.ndarray, T: int) -> np.ndarray:
    """Linear time-resample a (n, F) trajectory to (T, F). NaNs are
    propagated; downstream stats use nanmean / nanvar.
    """
    n, F = traj.shape
    if n == T:
        return traj
    if n < 2:
        return np.tile(traj, (T, 1)).reshape(T, F)
    src_t = np.linspace(0.0, 1.0, n)
    dst_t = np.linspace(0.0, 1.0, T)
    out = np.zeros((T, F), dtype=np.float32)
    for f in range(F):
        out[:, f] = np.interp(dst_t, src_t, traj[:, f])
    return out


def _trajectory_from_file(path: Path) -> np.ndarray | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    frames = data.get("frames") or []
    if not frames:
        return None
    rows = [_frame_to_features(f) for f in frames]
    return np.stack(rows, axis=0)


def fit_template(
    sign_id: str,
    trajectories_dir: Path,
    out_dir: Path,
    T: int = 32,
    min_clips: int = 5,
) -> dict:
    sign_dir = trajectories_dir / sign_id
    if not sign_dir.exists():
        return {"sign_id": sign_id, "status": "no_dir"}
    json_files = sorted(sign_dir.glob("*.json"))
    if len(json_files) < min_clips:
        return {"sign_id": sign_id, "status": "too_few_clips", "n": len(json_files)}

    stacked = []
    for p in json_files:
        traj = _trajectory_from_file(p)
        if traj is None:
            continue
        stacked.append(_resample_trajectory(traj, T))
    if len(stacked) < min_clips:
        return {"sign_id": sign_id, "status": "too_few_usable", "n": len(stacked)}

    arr = np.stack(stacked, axis=0)  # (n_clips, T, F)
    mean = np.nanmean(arr, axis=0)
    var = np.nanvar(arr, axis=0)
    # Floor variance to avoid divide-by-zero in Mahalanobis
    var = np.maximum(var, 1.0)  # 1 pixel² floor

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sign_id}.npz"
    np.savez_compressed(out_path, mean=mean, var=var, n_clips=len(stacked), T=T)
    return {"sign_id": sign_id, "status": "ok", "n": len(stacked), "out": str(out_path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories-dir", type=Path, default=Path("data/trajectories"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/templates"))
    ap.add_argument("--vocab-json", type=Path, default=Path("dataset/slice1b_vocabulary.json"))
    ap.add_argument("--time-steps", type=int, default=32)
    ap.add_argument("--min-clips", type=int, default=5)
    args = ap.parse_args()

    vocab = json.loads(args.vocab_json.read_text())
    sign_ids = [s["sign_id"] for s in vocab["kept_signs"]]
    results = []
    for sid in sign_ids:
        r = fit_template(sid, args.trajectories_dir, args.out_dir,
                         T=args.time_steps, min_clips=args.min_clips)
        results.append(r)
        print(f"  {sid}: {r}")

    summary_path = args.out_dir / "_fit_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nfit {ok}/{len(results)} signs; summary at {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
