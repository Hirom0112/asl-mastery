"""Convert per-sign template .npz into compact JSON the browser avatar can fetch.

100-d feature layout (per fit_templates.py): [slot0 21*2 kpts][slot1 21*2 kpts][pose 8*2 kpts].
Pose joint order: nose, neck, r_sh, l_sh, r_el, l_el, r_wr, l_wr.
Coordinates are pose-anchor-subtracted and shoulder-scale normalized, so they are
already in a unitless "body space" — perfect for direct mapping to the avatar rig.

By default we export the **medoid clip** (the single real clip with the smallest
total distance to all other clips of that sign) rather than the mean. The mean
of ~N clips collapses toward a low-amplitude central blob (both wrists drift to
center → the "crossed/mushy arms" artifact); the medoid is a real, crisp motion.
Pass --mean to fall back to the old mean-trajectory behavior. Per-frame variance
/ per-sign thresholds stay server-side for the matcher; the avatar only needs the
canonical motion to play back.

Usage:
    python scripts/export_templates_to_json.py \\
        --src data/templates_top80 --dst public/templates
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "templates_v3"
DST = ROOT / "public" / "templates"
ONE_HANDED_JSON = ROOT / "dataset" / "one_handed_signs.json"

NUM_HAND_KP = 21
NUM_POSE_KP = 8
SLOT_DIM = NUM_HAND_KP * 2  # 42
POSE_BASE = 2 * SLOT_DIM  # 84


def _select_medoid(clips: np.ndarray) -> np.ndarray:
    """Return the medoid clip (T, F) from a (n_clips, T, F) stack.

    Medoid = the clip minimizing the summed NaN-tolerant per-element MSE to all
    other clips. Mirrors fit_templates._clip_distance so the avatar plays back a
    representative *real* motion instead of the washed-out mean.
    """
    n = clips.shape[0]
    if n == 1:
        return clips[0]
    # Pairwise NaN-tolerant MSE. n is at most a few hundred → O(n^2) is fine.
    total = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = clips[i] - clips[j]
            valid = ~np.isnan(diff)
            if not valid.any():
                d = float("inf")
            else:
                d = float((np.where(valid, diff, 0.0) ** 2).sum() / valid.sum())
            total[i] += d
            total[j] += d
    return clips[int(np.argmin(total))]


def _reshape_hand(flat: np.ndarray) -> list[list[list[float] | None]]:
    pts: list[list[float] | None] = []
    for i in range(NUM_HAND_KP):
        x = float(flat[i * 2 + 0])
        y = float(flat[i * 2 + 1])
        if np.isnan(x) or np.isnan(y):
            pts.append(None)
        else:
            pts.append([round(x, 4), round(y, 4)])
    return pts  # type: ignore[return-value]


def _reshape_pose(flat: np.ndarray) -> list[list[float] | None]:
    pts: list[list[float] | None] = []
    for i in range(NUM_POSE_KP):
        x = float(flat[i * 2 + 0])
        y = float(flat[i * 2 + 1])
        if np.isnan(x) or np.isnan(y):
            pts.append(None)
        else:
            pts.append([round(x, 4), round(y, 4)])
    return pts


def export_one(npz_path: Path, out_path: Path, one_handed: bool = False,
               use_medoid: bool = True) -> None:
    d = np.load(npz_path)
    # Prefer the fitted dominant_hand_only flag (auto motion-ratio OR the
    # manual one_handed_signs.json forced set, already OR'd in fit_templates)
    # over the export-time json set, so new-vocab signs get the avatar arm-rest.
    if "dominant_hand_only" in d:
        one_handed = bool(d["dominant_hand_only"]) or one_handed
    if use_medoid and "clips" in d and d["clips"].shape[0] > 0:
        mean = _select_medoid(d["clips"])  # (T, F) — representative real clip
    else:
        mean = d["mean"]  # (T, F)
    T, F = mean.shape
    if F != 100:
        print(f"skip {npz_path.name}: feature dim {F} (expected 100)")
        return

    # Re-normalize so shoulder span ~= 1 unit, in case upstream scale fell
    # through to the floor. Use the median shoulder span across frames where
    # both shoulders are present.
    r_sh = mean[:, POSE_BASE + 2 * 2:POSE_BASE + 2 * 2 + 2]  # idx 2
    l_sh = mean[:, POSE_BASE + 3 * 2:POSE_BASE + 3 * 2 + 2]  # idx 3
    span = np.hypot(r_sh[:, 0] - l_sh[:, 0], r_sh[:, 1] - l_sh[:, 1])
    valid = span[np.isfinite(span) & (span > 1e-3)]
    norm = float(np.median(valid)) if valid.size else 1.0
    if norm < 1e-3:
        norm = 1.0
    mean = mean / norm

    frames = []
    for t in range(T):
        f = mean[t]
        frames.append(
            {
                "hand0": _reshape_hand(f[0:SLOT_DIM]),
                "hand1": _reshape_hand(f[SLOT_DIM:POSE_BASE]),
                "pose": _reshape_pose(f[POSE_BASE:POSE_BASE + NUM_POSE_KP * 2]),
            }
        )

    out = {
        "sign": npz_path.stem,
        "frames": frames,
        "n_clips": int(d["n_clips"]),
        "T": T,
        # Canonical one-handed signs (from dataset/one_handed_signs.json): the
        # avatar drives only the dominant arm and rests the other at the side,
        # since the mean trajectory's resting/non-dominant wrist is washed
        # toward center and otherwise reads as crossed arms.
        "one_handed": bool(one_handed),
        "schema": {
            "pose_order": ["nose", "neck", "r_sh", "l_sh", "r_el", "l_el", "r_wr", "l_wr"],
            "coord_space": "pose_anchor_subtracted_shoulder_scaled",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, separators=(",", ":")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=SRC,
                    help="dir of per-sign template .npz (default: data/templates_v3)")
    ap.add_argument("--dst", type=Path, default=DST,
                    help="output dir for <gloss>.json (default: public/templates)")
    ap.add_argument("--one-handed-json", type=Path, default=ONE_HANDED_JSON)
    ap.add_argument("--mean", action="store_true",
                    help="export the mean trajectory instead of the medoid clip")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these sign ids (default: all .npz in --src)")
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"missing {args.src}")
    use_medoid = not args.mean
    one_handed: set[str] = set()
    if args.one_handed_json.exists():
        one_handed = set(json.loads(args.one_handed_json.read_text()).get("one_handed", []))
    files = sorted(args.src.glob("*.npz"))
    files = [f for f in files if not f.stem.startswith("_")]
    if args.only:
        keep = set(args.only)
        files = [f for f in files if f.stem in keep]
    mode = "medoid" if use_medoid else "mean"
    print(f"exporting {len(files)} templates ({mode}) → {args.dst} "
          f"({len(one_handed)} one-handed)")
    n_oh = 0
    for f in files:
        out = args.dst / f"{f.stem}.json"
        is_oh = f.stem in one_handed
        n_oh += is_oh
        export_one(f, out, one_handed=is_oh, use_medoid=use_medoid)
    print(f"wrote {len(files)} files ({n_oh} flagged one-handed)")


if __name__ == "__main__":
    main()
