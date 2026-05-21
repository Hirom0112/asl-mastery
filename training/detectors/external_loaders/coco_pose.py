"""COCO 2017 keypoints → project-internal pose manifest.

COCO ships 17 keypoints per person; we map a subset to our 8-keypoint
upper-body topology:

  ours        ← coco
  0 nose      ← 0 nose
  1 neck      ← average of (5 l_shoulder, 6 r_shoulder)
  2 r_shoulder← 6 right_shoulder
  3 l_shoulder← 5 left_shoulder
  4 r_elbow   ← 8 right_elbow
  5 l_elbow   ← 7 left_elbow
  6 r_wrist   ← 10 right_wrist
  7 l_wrist   ← 9 left_wrist

COCO visibility convention: v=0 not labeled, v=1 labeled but invisible,
v=2 labeled + visible. We preserve.
"""

from __future__ import annotations

import json
from pathlib import Path

from training.detectors.external_loaders._util import EXTERNAL_ROOT, repo_rel

DATASET_DIR = EXTERNAL_ROOT / "coco_wholebody"
COCO_PERSON_CATEGORY_ID = 1


def _map_to_8(coco_kps: list[float]) -> list[tuple[float, float, float]]:
    """COCO 17*3 flat list -> our 8 keypoints."""
    # coco_kps is [x0,y0,v0, x1,y1,v1, ..., x16,y16,v16]
    def get(i: int) -> tuple[float, float, float]:
        return (coco_kps[i * 3], coco_kps[i * 3 + 1], coco_kps[i * 3 + 2])

    nose = get(0)
    l_sh = get(5)
    r_sh = get(6)
    # Neck: midpoint of shoulders if either visible
    if l_sh[2] > 0 and r_sh[2] > 0:
        neck = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2, 2.0)
    elif l_sh[2] > 0:
        neck = (l_sh[0], l_sh[1], 1.0)
    elif r_sh[2] > 0:
        neck = (r_sh[0], r_sh[1], 1.0)
    else:
        neck = (0.0, 0.0, 0.0)
    return [
        nose,
        neck,
        r_sh,
        l_sh,
        get(8),   # r_elbow
        get(7),   # l_elbow
        get(10),  # r_wrist
        get(9),   # l_wrist
    ]


def _load_split(split: str) -> list[dict]:
    """split in {'train', 'val'}."""
    ann_path = DATASET_DIR / "annotations" / f"person_keypoints_{split}2017.json"
    img_dir = DATASET_DIR / f"{split}2017"
    if not ann_path.exists():
        # The annotations ZIP unpacks to annotations/...; fall back to root.
        ann_path = DATASET_DIR / f"person_keypoints_{split}2017.json"
    if not ann_path.exists() or not img_dir.exists():
        print(f"[coco_pose:{split}] missing {ann_path} or {img_dir} — skipping")
        return []

    data = json.loads(ann_path.read_text())
    img_index = {img["id"]: img for img in data["images"]}

    by_image: dict[int, list[tuple[float, float, float]]] = {}
    bboxes_by_image: dict[int, list[list[float]]] = {}
    for ann in data["annotations"]:
        if ann.get("category_id") != COCO_PERSON_CATEGORY_ID:
            continue
        kps = ann.get("keypoints")
        if not kps:
            continue
        mapped = _map_to_8(kps)
        # Skip if no upper-body keypoint is visible
        if not any(v > 0 for _, _, v in mapped[2:]):
            continue
        by_image.setdefault(ann["image_id"], []).append(mapped)
        x, y, w, h = ann["bbox"]
        bboxes_by_image.setdefault(ann["image_id"], []).append([x, y, x + w, y + h])

    items: list[dict] = []
    for img_id, persons in by_image.items():
        img = img_index[img_id]
        img_path = img_dir / img["file_name"]
        if not img_path.exists():
            continue
        items.append(
            {
                "image_path": repo_rel(img_path),
                "width": img["width"],
                "height": img["height"],
                "persons": [
                    {"bbox": bbox, "keypoints": kp, "source": "coco_pose"}
                    for bbox, kp in zip(bboxes_by_image[img_id], persons)
                ],
            }
        )
    print(f"[coco_pose:{split}] emitted {len(items)} pose items")
    return items


def load_pose_keypoints() -> list[dict]:
    return _load_split("train") + _load_split("val")
