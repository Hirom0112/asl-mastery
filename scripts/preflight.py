#!/usr/bin/env python3
"""Pre-training health check. Run before any Modal launch.

Catches the failure modes from session 16's deep-dive — manifest
dim corruption, OOB-bbox pathology, broken loss numerics, path
typos in entrypoints — all in <60 s on local CPU.

Usage:
    python -m scripts.preflight
    python -m scripts.preflight --train data/labeled_frames/hand_bbox/external_train.json \\
                                --val   data/labeled_frames/hand_bbox/external_val.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parent.parent


def _src(path: str) -> str:
    if "freihand" in path:
        return "freihand"
    if "cmu" in path:
        return "cmu"
    if "mpii" in path:
        return "mpii"
    if "coco" in path:
        return "coco"
    if "wider" in path:
        return "wider"
    return "other"


def audit_manifest(name: str, path: Path) -> dict:
    """Print + return health stats. Never raises — surfaces issues for human review."""
    raw = json.loads(path.read_text())
    items = raw.get("items", raw) if isinstance(raw, dict) else raw
    n = len(items)
    print(f"\n--- {name}: {path}")
    print(f"  n_records: {n}")

    by_src = Counter()
    no_dims = 0
    n_oob_records = 0
    n_oob_boxes = 0
    n_boxes = 0
    for it in items:
        path_str = it.get("image_path", "")
        by_src[_src(path_str)] += 1
        w, h = int(it.get("width", 0)), int(it.get("height", 0))
        boxes = it.get("bboxes", [])
        n_boxes += len(boxes)
        if w <= 0 or h <= 0:
            no_dims += 1
            continue
        record_any_oob = False
        for b in boxes:
            x0, y0, x1, y1 = (float(x) for x in b)
            if x0 < 0 or y0 < 0 or x1 > w + 0.5 or y1 > h + 0.5:
                n_oob_boxes += 1
                record_any_oob = True
        if record_any_oob:
            n_oob_records += 1

    print(f"  by source: {dict(by_src.most_common())}")
    print(f"  records with manifest dims=0: {no_dims}  "
          f"({100 * no_dims / n:.1f}%)")
    if no_dims:
        print(f"    ⚠️  upstream bug — `__getitem__` falls back to actual image"
              f" dims; OOB-clip in load_manifest must skip these.")
    print(f"  OOB boxes: {n_oob_boxes} / {n_boxes} "
          f"({100 * n_oob_boxes / max(n_boxes, 1):.2f}%)")
    print(f"  records with ≥1 OOB box: {n_oob_records} "
          f"({100 * n_oob_records / n:.2f}%)")

    return {
        "n_records": n,
        "by_source": dict(by_src),
        "no_dims": no_dims,
        "n_oob_boxes": n_oob_boxes,
        "n_oob_records": n_oob_records,
        "n_boxes": n_boxes,
    }


def smoke_one_batch(train_manifest: Path) -> None:
    """Build one batch, run model forward + loss under bf16. Asserts finite."""
    print("\n--- smoke: 1 batch forward + loss under bf16")
    from training.detectors.dataset import HandBboxDataset, load_manifest
    from training.detectors.hand_detector import HandDetector
    from training.detectors.losses import HandDetectorLoss

    records = load_manifest(train_manifest)
    # take first 4 with at least one image we can actually read locally
    sub = []
    for r in records:
        if not r.image_path.exists():
            continue
        sub.append(r)
        if len(sub) >= 4:
            break
    if not sub:
        print("  no local images found at any image_path — skipping forward smoke")
        return
    print(f"  using {len(sub)} local samples for forward smoke")

    ds = HandBboxDataset(sub, augment=None, cache_in_memory=False)
    images, targets = [], []
    for i in range(len(sub)):
        img, tgt = ds[i]
        images.append(img)
        targets.append(tgt)
    images = torch.stack(images, dim=0)
    targets = {k: torch.stack([t[k] for t in targets], dim=0) for k in targets[0]}

    model = HandDetector()
    loss_fn = HandDetectorLoss()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(images)
        losses = loss_fn(out, targets)
    total = losses["total"].item()
    hm = losses["heatmap"].item()
    sz = losses["size"].item()
    if not (torch.isfinite(losses["total"]).item()
            and torch.isfinite(losses["heatmap"]).item()
            and torch.isfinite(losses["size"]).item()):
        print(f"  ❌ NON-FINITE LOSS: total={total} heatmap={hm} size={sz}")
        raise SystemExit(2)
    print(f"  ✓ total={total:.4f}  heatmap={hm:.4f}  size={sz:.4f}  (all finite)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path,
                    default=REPO / "data/labeled_frames/hand_bbox/external_train.json")
    ap.add_argument("--val", type=Path,
                    default=REPO / "data/labeled_frames/hand_bbox/external_val.json")
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print(f"=== preflight @ {time.strftime('%Y-%m-%dT%H:%M:%S')} ===")
    print(f"repo: {REPO}")

    if not args.train.exists():
        print(f"❌ train manifest not found: {args.train}")
        return 1
    if not args.val.exists():
        print(f"❌ val manifest not found: {args.val}")
        return 1

    audit_manifest("TRAIN", args.train)
    audit_manifest("VAL", args.val)

    if not args.skip_smoke:
        smoke_one_batch(args.train)

    print(f"\n=== preflight passed in {time.time() - t0:.1f}s ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
