"""WIDER FACE → project-internal face_bbox manifest.

The annotations ZIP unpacks to wider_face_split/ with three plaintext files:
  wider_face_train_bbx_gt.txt
  wider_face_val_bbx_gt.txt
  wider_face_test_bbx_gt.txt  (no GT)

Each file repeats the pattern:
  <image_relative_path>
  <num_faces>
  x y w h blur expression illumination invalid occlusion pose   (× num_faces)

Images live under data/external/wider_face/WIDER_<split>/images/... after
manual download (the script README explains).
"""

from __future__ import annotations

from pathlib import Path

from training.detectors.external_loaders._util import EXTERNAL_ROOT, repo_rel

DATASET_DIR = EXTERNAL_ROOT / "wider_face"


def _try_imagesize(p: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image
        with Image.open(p) as im:
            return im.size
    except Exception:
        return None


def _parse_split(gt_path: Path, image_root: Path) -> list[dict]:
    if not gt_path.exists():
        print(f"[wider_face] missing {gt_path} — skipping")
        return []
    lines = gt_path.read_text().splitlines()
    i = 0
    items: list[dict] = []
    while i < len(lines):
        rel = lines[i].strip()
        if not rel:
            i += 1
            continue
        i += 1
        try:
            n = int(lines[i].strip())
        except (ValueError, IndexError):
            i += 1
            continue
        i += 1
        bboxes: list[list[float]] = []
        # When n == 0, GT files include a single padding line "0 0 0 0 0 0 0 0 0 0"
        if n == 0:
            i += 1
            continue
        for _ in range(n):
            if i >= len(lines):
                break
            parts = lines[i].split()
            i += 1
            if len(parts) < 4:
                continue
            try:
                x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue
            # Optional invalid flag is index 7 in the 10-int rows
            invalid = parts[7] == "1" if len(parts) >= 8 else False
            if invalid:
                continue
            bboxes.append([x, y, x + w, y + h])

        if not bboxes:
            continue

        img_path = image_root / rel
        if not img_path.exists():
            # image splits are optional — skip silently when only annotations present
            continue
        size = _try_imagesize(img_path) or (0, 0)
        items.append(
            {
                "image_path": repo_rel(img_path),
                "width": size[0],
                "height": size[1],
                "bboxes": bboxes,
            }
        )
    print(f"[wider_face:{gt_path.name}] emitted {len(items)} face_bbox items")
    return items


def load_face_bbox() -> list[dict]:
    splits = [
        (DATASET_DIR / "wider_face_split" / "wider_face_train_bbx_gt.txt",
         DATASET_DIR / "WIDER_train" / "images"),
        (DATASET_DIR / "wider_face_split" / "wider_face_val_bbx_gt.txt",
         DATASET_DIR / "WIDER_val" / "images"),
    ]
    items: list[dict] = []
    for gt, root in splits:
        items.extend(_parse_split(gt, root))
    return items
