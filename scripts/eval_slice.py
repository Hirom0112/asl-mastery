"""Per-slice top-1 evaluation harness.

The deployed UX is "Sign 3 of 8" — a closed lesson slice. A sign with low
global top-1 accuracy may still have high in-slice accuracy because the
realistic confusion set is the 8 lesson signs, not the full 75-sign vocab.

Usage:
    python -m scripts.eval_slice \
        --templates-dir data/templates_v6 \
        --val-trajectories-dir runs/trajectories_v5_val \
        --slice-size 8 --n-slices 20 --seed 0

Reports mean in-slice top-1 across randomly sampled slices of the given
size, plus the global (full-vocab) top-1 for the same val clips for
contrast.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from training.detectors.sign_matcher import (
    load_templates,
    predict,
    predict_in_slice,
    trajectory_from_frames,
)


def _val_trajectories(val_dir: Path, templates: dict) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {}
    for sign_dir in sorted(val_dir.glob("*")):
        if not sign_dir.is_dir():
            continue
        sign = sign_dir.name
        if sign not in templates:
            continue
        T = templates[sign]["T"]
        trajs: list[np.ndarray] = []
        for jp in sorted(sign_dir.glob("*.json")):
            try:
                data = json.loads(jp.read_text())
                trajs.append(trajectory_from_frames(data["frames"], T))
            except Exception:
                continue
        if trajs:
            out[sign] = trajs
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates-dir", type=Path, default=Path("data/templates_v6"))
    ap.add_argument("--val-trajectories-dir", type=Path, default=Path("runs/trajectories_v5_val"))
    ap.add_argument("--slice-size", type=int, default=8)
    ap.add_argument("--n-slices", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/templates_v6/_slice_eval.json"))
    args = ap.parse_args()

    templates = load_templates(args.templates_dir)
    print(f"loaded {len(templates)} templates")
    val = _val_trajectories(args.val_trajectories_dir, templates)
    print(f"val clips: {sum(len(v) for v in val.values())} across {len(val)} signs")

    rng = random.Random(args.seed)
    signs_with_val = [s for s in templates if val.get(s)]

    # Global top-1 baseline
    global_correct = 0
    global_total = 0
    for sign, trajs in val.items():
        for t in trajs:
            best, _, _ = predict(t, templates)
            global_correct += int(best == sign)
            global_total += 1
    global_top1 = global_correct / max(global_total, 1)
    print(f"global top-1: {global_top1:.3f}  ({global_correct}/{global_total})")

    # Random slices
    per_slice: list[dict] = []
    for s_idx in range(args.n_slices):
        slice_signs = rng.sample(signs_with_val, k=min(args.slice_size, len(signs_with_val)))
        correct = 0
        total = 0
        for sign in slice_signs:
            for t in val.get(sign, []):
                best, conf, _ = predict_in_slice(t, templates, slice_signs)
                correct += int(best == sign)
                total += 1
        acc = correct / max(total, 1)
        per_slice.append({"slice_signs": slice_signs, "top1": acc,
                          "correct": correct, "total": total})

    mean_slice = float(np.mean([s["top1"] for s in per_slice]))
    print(f"in-slice top-1 (n={args.slice_size}, {args.n_slices} samples): mean={mean_slice:.3f}")
    print(f"  best: {max(s['top1'] for s in per_slice):.3f}  "
          f"worst: {min(s['top1'] for s in per_slice):.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "slice_size": args.slice_size,
        "n_slices": args.n_slices,
        "global_top1": global_top1,
        "mean_slice_top1": mean_slice,
        "per_slice": per_slice,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
