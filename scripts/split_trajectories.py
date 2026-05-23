"""Split per-sign trajectories into train (80%) + val (20%) directories
for fit_templates → sign_matcher calibration.

Uses a fixed seed for reproducibility. Symlinks instead of copies to save
disk + I/O.

Usage:
    python -m scripts.split_trajectories \
        --in-dir  runs/trajectories_v2/trajectories_v2 \
        --train-dir runs/trajectories_v2_train \
        --val-dir   runs/trajectories_v2_val \
        --val-frac 0.20 --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--train-dir", type=Path, required=True)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    train_n = val_n = 0
    sign_stats = {}

    for sign_dir in sorted(args.in_dir.glob("*")):
        if not sign_dir.is_dir():
            continue
        sign = sign_dir.name
        jsons = sorted(sign_dir.glob("*.json"))
        if not jsons:
            continue
        rnd.shuffle(jsons)
        n_val = max(1, int(len(jsons) * args.val_frac)) if len(jsons) >= 2 else 0
        val_files = jsons[:n_val]
        train_files = jsons[n_val:]

        (args.train_dir / sign).mkdir(parents=True, exist_ok=True)
        (args.val_dir / sign).mkdir(parents=True, exist_ok=True)
        for jp in train_files:
            link = args.train_dir / sign / jp.name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(jp.resolve())
        for jp in val_files:
            link = args.val_dir / sign / jp.name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(jp.resolve())

        train_n += len(train_files)
        val_n += len(val_files)
        sign_stats[sign] = {"train": len(train_files), "val": len(val_files)}

    print(f"signs: {len(sign_stats)}")
    print(f"train: {train_n}   val: {val_n}")
    print()
    print("per-sign (train/val):")
    for s, c in sorted(sign_stats.items(), key=lambda kv: -kv[1]["train"]):
        print(f"  {s}: {c['train']}/{c['val']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
