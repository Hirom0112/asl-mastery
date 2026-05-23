"""CMU Panoptic HandDB → project-internal manifest.

CMU HandDB ships three subsets, each with per-image JSON sidecars:

- hand_labels/        — manual subset, MPII+NZSL
- hand_labels_synth/  — synthetic renders
- hand143_panopticdb/ — multiview-bootstrapped (from CMU 31-cam dome)

Each JSON has the shape:
  {
    "version": 1.0,
    "hand_pts": [[x, y, visibility], ...]   # length 21, canonical topology
  }

Image filename matches the JSON's stem with a .jpg extension.
"""

from __future__ import annotations

import json
from pathlib import Path

from training.detectors.external_loaders._util import (
    EXTERNAL_ROOT,
    bbox_from_keypoints,
    repo_rel,
)

DATASET_DIR = EXTERNAL_ROOT / "cmu_panoptic_handdb"

SUBSETS = {
    "manual": "hand_labels",
    "synth": "hand_labels_synth",
    "multiview": "hand143_panopticdb",
}


def _try_imagesize(p: Path) -> tuple[int, int] | None:
    """Return (width, height) or None without requiring PIL.

    Disabled by default for speed — opening 17K JPGs to read headers
    is slow on a Modal volume. The dataset loader can compute size
    at __getitem__ time when it actually reads the image. Set the env
    var ASL_LOADER_READ_IMAGE_SIZE=1 to re-enable for local dev.
    """
    import os
    if not os.environ.get("ASL_LOADER_READ_IMAGE_SIZE"):
        return None
    try:
        from PIL import Image  # noqa: WPS433 (optional dep)
    except Exception:
        return None
    try:
        with Image.open(p) as im:
            return im.size
    except Exception:
        return None


def _load_subset(subset_dir: Path, source_tag: str) -> list[dict]:
    if not subset_dir.exists():
        print(f"[cmu_handdb] subset missing: {subset_dir} — skipping")
        return []

    # Multiview subset (hand143_panopticdb) ships a single combined JSON
    # `hands_v143_14817.json` with a `root` array of per-frame records,
    # rather than per-image JSON sidecars. Detect and dispatch.
    big_json_candidates = list(subset_dir.glob("hands_v*.json"))
    if big_json_candidates:
        return _load_panopticdb_combined(big_json_candidates[0], subset_dir, source_tag)

    items: list[dict] = []
    # Sidecar layout: per-image JSON with same stem as .jpg / .png
    for json_path in subset_dir.rglob("*.json"):
        try:
            data = json.loads(json_path.read_text())
        except Exception as e:
            print(f"[cmu_handdb] bad JSON {json_path}: {e}")
            continue
        pts = data.get("hand_pts")
        if not pts or len(pts) != 21:
            continue

        img_path = json_path.with_suffix(".jpg")
        if not img_path.exists():
            img_path = json_path.with_suffix(".png")
            if not img_path.exists():
                continue
        size = _try_imagesize(img_path) or (0, 0)
        kps = [(float(p[0]), float(p[1]), float(p[2])) for p in pts]
        bbox = bbox_from_keypoints(kps)
        if bbox is None:
            continue
        items.append(
            {
                "image_path": repo_rel(img_path),
                "width": size[0],
                "height": size[1],
                "hands": [
                    {
                        "bbox": bbox,
                        "keypoints": kps,
                        "source": f"cmu_handdb_{source_tag}",
                    }
                ],
            }
        )
    print(f"[cmu_handdb:{source_tag}] emitted {len(items)} hand_keypoint items")
    return items


def _load_panopticdb_combined(json_path: Path, subset_dir: Path, source_tag: str) -> list[dict]:
    """hand143_panopticdb single-JSON format:
      {"root": [{"img_paths": "imgs/00000000.jpg",
                 "img_width": 1920, "img_height": 1080,
                 "joint_self": [[x, y, v], ...]   # length 21
                 }, ...]}
    """
    data = json.loads(json_path.read_text())
    items: list[dict] = []
    for rec in data.get("root", []):
        kps_raw = rec.get("joint_self")
        if not kps_raw or len(kps_raw) != 21:
            continue
        img_rel = rec.get("img_paths")
        if not img_rel:
            continue
        img_path = (subset_dir / img_rel).resolve()
        if not img_path.exists():
            continue
        kps = [(float(p[0]), float(p[1]), float(p[2])) for p in kps_raw]
        bbox = bbox_from_keypoints(kps)
        if bbox is None:
            continue
        items.append(
            {
                "image_path": repo_rel(img_path),
                "width": int(rec.get("img_width", 0)),
                "height": int(rec.get("img_height", 0)),
                "hands": [
                    {
                        "bbox": bbox,
                        "keypoints": kps,
                        "source": f"cmu_handdb_{source_tag}",
                    }
                ],
            }
        )
    print(f"[cmu_handdb:{source_tag}] emitted {len(items)} hand_keypoint items (combined-JSON format)")
    return items


def load_hand_keypoints(include: tuple[str, ...] = ("manual", "synth", "multiview")) -> list[dict]:
    items: list[dict] = []
    for tag in include:
        sub = SUBSETS.get(tag)
        if sub is None:
            continue
        items.extend(_load_subset(DATASET_DIR / sub, tag))
    return items


def load_hand_bbox(include: tuple[str, ...] = ("manual", "synth", "multiview")) -> list[dict]:
    kp_items = load_hand_keypoints(include=include)
    return [
        {
            "image_path": it["image_path"],
            "width": it["width"],
            "height": it["height"],
            "bboxes": [h["bbox"] for h in it["hands"]],
        }
        for it in kp_items
    ]
