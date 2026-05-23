"""WFLW (Wider Facial Landmarks in-the-wild) → project-internal face_keypoints manifest.

WFLW: 10,000 faces (7,500 train / 2,500 test), 98 FULLY MANUALLY-ANNOTATED
landmarks (Wu et al., CVPR 2018, "Look at Boundary"). ADR-0015 compliant —
the 98 points were hand-clicked by human annotators on WIDER FACE images,
NOT produced by any pretrained model. The 98 points include the full
face-boundary contour (jaw/chin/cheeks/temples — points 0–32), eyebrows
(33–50), nose (51–59), eyes (60–95), and mouth (76–95 region) — exactly the
ASL location anchors (chin/forehead/cheek/mouth).

Annotation line format (list_98pt_rect_attr_{train,test}.txt), space-separated:
  x0 y0 x1 y1 ... x97 y97  (196 floats: 98 landmark coords)
  x_min y_min x_max y_max  (4 floats: face rect)
  pose expression illumination make-up occlusion blur  (6 ints: attributes)
  image_name               (1 str: path under WFLW_images/)
  = 207 tokens total.

Emits the keypoint manifest schema consumed by landmarks_dataset.py with
instance_key="faces" and num_keypoints=98.
"""
from __future__ import annotations

from pathlib import Path

N_LM = 98


def _parse_line(line: str, images_root: Path) -> dict | None:
    t = line.strip().split()
    if len(t) < 207:
        return None
    coords = [float(v) for v in t[:196]]
    kps = [[coords[2 * i], coords[2 * i + 1], 2] for i in range(N_LM)]  # v=2 (visible)
    x0, y0, x1, y1 = (float(v) for v in t[196:200])
    name = t[206]
    img_path = images_root / name
    return {
        "image_path": str(img_path),
        "width": 0,            # 0 → landmarks_dataset reads real dims at load
        "height": 0,
        "faces": [{"bbox": [x0, y0, x1, y1], "keypoints": kps, "source": "wflw"}],
    }


def load_split(wflw_root: Path, split: str) -> list[dict]:
    """split: 'train' or 'test'. wflw_root holds WFLW_images/ and
    WFLW_annotations/."""
    ann = (wflw_root / "WFLW_annotations"
           / "list_98pt_rect_attr_train_test"
           / f"list_98pt_rect_attr_{split}.txt")
    images_root = wflw_root / "WFLW_images"
    if not ann.exists():
        print(f"[wflw] annotation file missing: {ann}")
        return []
    items = []
    for line in ann.read_text().splitlines():
        it = _parse_line(line, images_root)
        if it and it["image_path"] and Path(it["image_path"]).exists():
            items.append(it)
    print(f"[wflw:{split}] emitted {len(items)} face_keypoints items")
    return items


def load_face_keypoints(wflw_root: Path) -> tuple[list[dict], list[dict]]:
    """Returns (train_items, val_items). WFLW's official test split is used
    as val (signer/identity-disjoint by construction of the benchmark)."""
    return load_split(wflw_root, "train"), load_split(wflw_root, "test")
