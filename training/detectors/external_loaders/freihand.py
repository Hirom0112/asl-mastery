"""FreiHAND → project-internal manifest.

FreiHAND ships:
- training/rgb/<8-digit-id>.jpg (RGB images; 130,240 augmented samples)
- training_xyz.json  — list[N] of [21][3] 3D joint coords (millimeters, MANO)
- training_K.json    — list[N] of [3][3] camera intrinsics
- evaluation/rgb/... + evaluation_K.json (no GT keypoints in eval set)

We use training_xyz + training_K to project 3D joints → 2D pixel coordinates.
FreiHAND's 21-joint MANO order matches our canonical topology (wrist, thumb,
index, middle, ring, pinky — root then 4 along each finger).

Per the paper, the first 32,560 training samples are unique; samples
32,561-130,240 are 3 background-swap augmentations of the first 32,560.
We emit only the unique samples by default to avoid label duplication.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.detectors.external_loaders._util import (
    EXTERNAL_ROOT,
    bbox_from_keypoints,
    repo_rel,
)

DATASET_DIR = EXTERNAL_ROOT / "freihand"


def _project(xyz_mm: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project (21, 3) 3D points to (21, 2) pixel coords via K."""
    proj = (K @ xyz_mm.T).T
    return proj[:, :2] / proj[:, 2:3]


def load_hand_keypoints(
    unique_only: bool = True,
    image_subdir: str = "training/rgb",
) -> list[dict]:
    """Emit hand_keypoints manifest items for the FreiHAND training split.

    Returns a list where every item is a single-hand record (FreiHAND has
    one hand per image), in the project-internal hand_keypoints schema.
    """
    if not DATASET_DIR.exists():
        print(f"[freihand] dataset dir missing: {DATASET_DIR} — skipping")
        return []

    xyz_path = DATASET_DIR / "training_xyz.json"
    K_path = DATASET_DIR / "training_K.json"
    img_root = DATASET_DIR / image_subdir
    if not (xyz_path.exists() and K_path.exists() and img_root.exists()):
        print(f"[freihand] expected files missing under {DATASET_DIR} — skipping")
        return []

    xyz_all = np.array(json.loads(xyz_path.read_text()), dtype=np.float32)
    K_all = np.array(json.loads(K_path.read_text()), dtype=np.float32)

    n_unique = 32560
    n_total = xyz_all.shape[0]
    n = n_unique if (unique_only and n_total >= n_unique) else n_total

    items: list[dict] = []
    for i in range(n):
        img_path = img_root / f"{i:08d}.jpg"
        if not img_path.exists():
            continue
        pts2d = _project(xyz_all[i], K_all[i])
        kps = [(float(x), float(y), 2.0) for x, y in pts2d]  # all visible
        bbox = bbox_from_keypoints(kps)
        if bbox is None:
            continue
        items.append(
            {
                "image_path": repo_rel(img_path),
                "width": 224,  # FreiHAND images are 224x224
                "height": 224,
                "hands": [
                    {
                        "bbox": bbox,
                        "keypoints": kps,
                        "source": "freihand",
                    }
                ],
            }
        )
    print(f"[freihand] emitted {len(items)} hand_keypoint items (unique_only={unique_only})")
    return items


def load_hand_bbox() -> list[dict]:
    """Convert FreiHAND to hand_bbox manifest (one bbox per image, derived
    from the keypoint extents).
    """
    kp_items = load_hand_keypoints()
    out = []
    for it in kp_items:
        out.append(
            {
                "image_path": it["image_path"],
                "width": it["width"],
                "height": it["height"],
                "bboxes": [h["bbox"] for h in it["hands"]],
            }
        )
    return out
