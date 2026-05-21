#!/usr/bin/env python3
"""Subset a v3 cleaning-pipeline manifest down to N signs for smoke training.

Picks the N signs with the most clips (so smoke training has the best
chance of overfitting and producing a non-trivial accuracy in 5 epochs).
Preserves split assignments and class indexing.

Usage:

    python scripts/make_smoke_manifest.py \\
        --input  dataset/clean/v3/dataset_v3_manifest.json \\
        --output dataset/clean/v3/dataset_v3_smoke_manifest.json \\
        --signs  2
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signs", type=int, default=2)
    args = parser.parse_args()

    with args.input.open() as f:
        manifest = json.load(f)

    records = manifest.get("records", [])
    counts = Counter(r["sign_id"] for r in records)
    top_signs = [s for s, _ in counts.most_common(args.signs)]
    kept = [r for r in records if r["sign_id"] in top_signs]

    out = {
        **manifest,
        "version": manifest.get("version", "v3") + "_smoke",
        "num_classes": len(top_signs),
        "classes": sorted(top_signs),
        "split_counts": {
            split: sum(1 for r in kept if r["split"] == split)
            for split in ("train", "val", "test")
        },
        "records": kept,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"wrote {args.output} with {len(kept)} clips across {len(top_signs)} signs: {sorted(top_signs)}")
    print(f"split counts: {out['split_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
