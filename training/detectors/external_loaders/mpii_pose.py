"""MPII Human Pose → project-internal pose manifest.

MPII ships annotations in a MATLAB v7.3 file:
  mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat

Each annorect has `annopoints.point` array with `id` ∈ [0..15]:
  0 r_ankle, 1 r_knee, 2 r_hip, 3 l_hip, 4 l_knee, 5 l_ankle,
  6 pelvis, 7 thorax, 8 upper_neck, 9 head_top,
  10 r_wrist, 11 r_elbow, 12 r_shoulder,
  13 l_shoulder, 14 l_elbow, 15 l_wrist

Mapping to our 8-keypoint topology:
  0 nose      ← (not annotated directly; use head_top as proxy, v halved)
  1 neck      ← 8 upper_neck (or fallback to 7 thorax)
  2 r_sh      ← 12, 3 l_sh ← 13
  4 r_elbow   ← 11, 5 l_elbow ← 14
  6 r_wrist   ← 10, 7 l_wrist ← 15

This loader requires scipy (already in training/requirements.txt) for
the .mat reader.
"""

from __future__ import annotations

from pathlib import Path

from training.detectors.external_loaders._util import EXTERNAL_ROOT, repo_rel

DATASET_DIR = EXTERNAL_ROOT / "mpii_pose"


def _try_imagesize(p: Path) -> tuple[int, int] | None:
    import os
    if not os.environ.get("ASL_LOADER_READ_IMAGE_SIZE"):
        return None
    try:
        from PIL import Image
        with Image.open(p) as im:
            return im.size
    except Exception:
        return None


def _kps_from_annorect(annorect) -> list[tuple[float, float, float]] | None:
    pts = {}
    try:
        annopoints = annorect.annopoints
        if not hasattr(annopoints, "point"):
            return None
        points = annopoints.point
        # scipy may return a single struct or array
        try:
            iter(points)
            items = list(points)
        except TypeError:
            items = [points]
        for p in items:
            try:
                pid = int(p.id)
                pts[pid] = (float(p.x), float(p.y))
            except Exception:
                continue
    except Exception:
        return None
    if not pts:
        return None

    def get(idx: int, v: float = 2.0) -> tuple[float, float, float]:
        if idx in pts:
            return (pts[idx][0], pts[idx][1], v)
        return (0.0, 0.0, 0.0)

    return [
        get(9, 1.0),   # nose ← head_top, lower confidence
        get(8) if 8 in pts else get(7),  # neck ← upper_neck or thorax
        get(12), get(13),  # shoulders
        get(11), get(14),  # elbows
        get(10), get(15),  # wrists
    ]


def load_pose_keypoints() -> list[dict]:
    mat_path = DATASET_DIR / "mpii_human_pose_v1_u12_2" / "mpii_human_pose_v1_u12_1.mat"
    img_root = DATASET_DIR / "images"
    if not mat_path.exists() or not img_root.exists():
        print(f"[mpii_pose] missing {mat_path} or {img_root} — skipping")
        return []

    try:
        from scipy.io import loadmat
    except ImportError:
        print("[mpii_pose] scipy not installed — install scipy to load MPII")
        return []

    try:
        mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"[mpii_pose] loadmat failed: {e}")
        return []

    release = mat["RELEASE"]
    annolist = release.annolist
    items: list[dict] = []
    for ann in annolist:
        try:
            img_name = ann.image.name
        except Exception:
            continue
        img_path = img_root / img_name
        if not img_path.exists():
            continue
        size = _try_imagesize(img_path) or (0, 0)

        # ann.annorect may be a single struct, an array, or missing
        try:
            rects = ann.annorect
            try:
                iter(rects)
                rect_list = list(rects)
            except TypeError:
                rect_list = [rects]
        except Exception:
            continue

        persons = []
        for r in rect_list:
            kps = _kps_from_annorect(r)
            if kps is None:
                continue
            # Skip if no upper-body keypoint is visible
            if not any(v > 0 for _, _, v in kps[2:]):
                continue
            xs = [p[0] for p in kps if p[2] > 0]
            ys = [p[1] for p in kps if p[2] > 0]
            if len(xs) < 2:
                continue
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            persons.append({"bbox": bbox, "keypoints": kps, "source": "mpii_pose"})

        if persons:
            items.append(
                {
                    "image_path": repo_rel(img_path),
                    "width": size[0],
                    "height": size[1],
                    "persons": persons,
                }
            )

    print(f"[mpii_pose] emitted {len(items)} pose items")
    return items
