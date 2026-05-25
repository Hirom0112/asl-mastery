"""COCO-WholeBody → project-internal hand_keypoints manifest.

COCO-WholeBody (Jin et al., ECCV 2020) extends COCO 2017 person annotations
with 133 keypoints per person, including **21 left-hand + 21 right-hand**
keypoints in the SAME topology we use (wrist, then 4 per finger). These are
**human-annotated, in-the-wild** hands across diverse real backgrounds, scales
and poses — the exact distribution FreiHAND (green-screen) lacks. Provenance:
CC-BY 4.0, human-annotated → ADR-0015 clean (see docs/data/external_datasets_audit.md §E).

Per-annotation fields used:
- lefthand_kpts / righthand_kpts : flat [x,y,v]*21 in image pixel coords
- lefthand_valid / righthand_valid : bool (only emit valid hands)
- lefthand_box / righthand_box : [x,y,w,h] (preferred bbox; fall back to kpts)

We emit one hand_keypoints item per IMAGE that has >=1 valid hand, with a
`hands` list (a COCO image can contain several people / up to many hands).
"""

from __future__ import annotations

import json
from pathlib import Path

from training.detectors.external_loaders._util import (
    EXTERNAL_ROOT,
    bbox_from_keypoints,
    repo_rel,
)

DATASET_DIR = EXTERNAL_ROOT / "coco_wholebody"
MIN_VISIBLE_KPTS = 6  # skip near-empty hand annotations


def _kpts_to_list(flat: list[float]) -> list[tuple[float, float, float]]:
    return [(float(flat[i * 3]), float(flat[i * 3 + 1]), float(flat[i * 3 + 2]))
            for i in range(21)]


def _box_xywh_to_xyxy(box) -> list[float] | None:
    if not box or len(box) != 4:
        return None
    x, y, w, h = box
    if w <= 1 or h <= 1:
        return None
    return [float(x), float(y), float(x + w), float(y + h)]


def load_hand_keypoints(split: str = "train") -> list[dict]:
    """Emit hand_keypoints items for the COCO-WholeBody {train,val} split."""
    if not DATASET_DIR.exists():
        print(f"[coco_wb_hands] dataset dir missing: {DATASET_DIR} — skipping")
        return []
    ann_path = DATASET_DIR / "annotations" / f"coco_wholebody_{split}_v1.0.json"
    img_root = DATASET_DIR / f"{split}2017"
    if not ann_path.exists():
        print(f"[coco_wb_hands] missing annotations {ann_path} — skipping")
        return []

    data = json.loads(ann_path.read_text())
    img_index = {im["id"]: im for im in data["images"]}

    by_image: dict[int, dict] = {}
    n_hands = 0
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        im = img_index.get(img_id)
        if im is None:
            continue
        for side in ("lefthand", "righthand"):
            if not ann.get(f"{side}_valid"):
                continue
            flat = ann.get(f"{side}_kpts")
            if not flat or len(flat) != 63:
                continue
            kps = _kpts_to_list(flat)
            if sum(1 for _, _, v in kps if v > 0) < MIN_VISIBLE_KPTS:
                continue
            bbox = _box_xywh_to_xyxy(ann.get(f"{side}_box")) or bbox_from_keypoints(kps)
            if bbox is None:
                continue
            entry = by_image.setdefault(img_id, {
                "image_path": repo_rel(img_root / im["file_name"]),
                "width": int(im.get("width", 0)),
                "height": int(im.get("height", 0)),
                "hands": [],
            })
            entry["hands"].append({"bbox": bbox, "keypoints": kps, "source": "coco_wholebody"})
            n_hands += 1

    items = [v for v in by_image.values() if v["hands"]]
    print(f"[coco_wb_hands:{split}] emitted {len(items)} images / {n_hands} hands")
    return items


if __name__ == "__main__":
    items = load_hand_keypoints("val")
    print("sample:", items[0] if items else None)
