"""P3 Step 1 — add hard negatives to the hand-detector training manifest.

Diagnosis (scripts/p1_landmark_diag.py + live-demo screenshots): the hand
detector hallucinates a phantom hand on the face/beard when one hand is
present. Root cause confirmed here: the current hand_bbox manifest is 100%
single-hand close-ups (FreiHAND + CMU), with ZERO no-hand images. The model
has never been taught "this region is NOT a hand."

Fix: mix in WIDER FACE images (faces, no hands) as NEGATIVES — same schema,
empty `bboxes`. The CenterNet focal loss treats an empty-box image as all
negatives, so the heatmap learns to stay low on faces/skin. Negatives are
capped to a fraction of the positives so we don't bias the detector toward
predicting nothing.

Runs on whatever paths you point it at (local manifests or Modal-volume
paths). Images are never read — this only rewrites the manifest JSON.

Usage:
    python -m scripts.build_hand_manifest_with_negatives \
        --hand-manifest  data/labeled_frames/hand_bbox/external_train.json \
        --face-manifest  data/labeled_frames/face_bbox/external_train.json \
        --out            data/labeled_frames/hand_bbox/train_with_negatives.json \
        --neg-frac 0.20 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _load_items(p: Path) -> tuple[dict, list]:
    raw = json.loads(p.read_text())
    return raw, raw.get("items", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand-manifest", type=Path, required=True)
    ap.add_argument("--face-manifest", type=Path, required=True,
                    help="A face_bbox manifest (WIDER FACE) — its images are "
                         "reused as no-hand negatives.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--neg-frac", type=float, default=0.20,
                    help="Negatives added = neg_frac × #positives. 0.20 = 20%.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    hand_raw, hand_items = _load_items(args.hand_manifest)
    assert hand_raw.get("task") == "hand_bbox", hand_raw.get("task")
    _, face_items = _load_items(args.face_manifest)

    n_pos = len(hand_items)
    n_neg = min(len(face_items), int(round(n_pos * args.neg_frac)))
    rnd = random.Random(args.seed)
    neg_src = rnd.sample(face_items, n_neg)

    negatives = []
    for it in neg_src:
        negatives.append({
            "image_path": it["image_path"],
            "width": it.get("width"),
            "height": it.get("height"),
            "bboxes": [],          # the whole point: a hand-free image
        })

    merged_items = hand_items + negatives
    rnd.shuffle(merged_items)
    out = {
        "version": hand_raw.get("version", 1),
        "task": "hand_bbox",
        "items": merged_items,
        "_negatives_added": n_neg,
        "_negatives_source": str(args.face_manifest),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"positives={n_pos}  negatives={n_neg} (neg_frac={args.neg_frac})  "
          f"→ {len(merged_items)} items → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
