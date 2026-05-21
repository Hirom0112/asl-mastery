"""Convert downloaded external CV datasets into project-internal manifests.

Usage:
    python -m training.detectors.normalize_external --task hand_bbox
    python -m training.detectors.normalize_external --task hand_keypoints
    python -m training.detectors.normalize_external --task pose_keypoints
    python -m training.detectors.normalize_external --task face_bbox
    python -m training.detectors.normalize_external --task all

Output (per task):
    data/labeled_frames/<task>/external_<dataset>.json   (per-dataset manifest)
    data/labeled_frames/<task>/external_combined.json    (union of all sources)

Each manifest is the v1 schema documented in
training/detectors/dataset.py — no other code needs to change to consume
the external datasets once these files exist.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from training.detectors.external_loaders._util import REPO_ROOT


TASKS = ("hand_bbox", "hand_keypoints", "pose_keypoints", "face_bbox", "all")

OUT_ROOT = REPO_ROOT / "data" / "labeled_frames"


def _write_manifest(out_path: Path, task: str, items: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"version": 1, "task": task, "items": items}, indent=2)
    )
    print(f"wrote {out_path}  ({len(items)} items)")


def _split_train_val(items: list[dict], val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    rnd = random.Random(seed)
    indices = list(range(len(items)))
    rnd.shuffle(indices)
    n_val = max(1, int(len(items) * val_frac)) if items else 0
    val = [items[i] for i in indices[:n_val]]
    train = [items[i] for i in indices[n_val:]]
    return train, val


def normalize_hand_bbox() -> dict[str, list[dict]]:
    from training.detectors.external_loaders import freihand, cmu_handdb, multiview_hand_pose
    per_source = {
        "freihand": freihand.load_hand_bbox(),
        "cmu_handdb": cmu_handdb.load_hand_bbox(),
        "multiview_hand_pose": [
            {"image_path": it["image_path"], "width": it["width"], "height": it["height"],
             "bboxes": [h["bbox"] for h in it["hands"]]}
            for it in multiview_hand_pose.load_hand_keypoints()
        ],
    }
    return per_source


def normalize_hand_keypoints() -> dict[str, list[dict]]:
    from training.detectors.external_loaders import freihand, cmu_handdb, multiview_hand_pose
    return {
        "freihand": freihand.load_hand_keypoints(),
        "cmu_handdb": cmu_handdb.load_hand_keypoints(),
        "multiview_hand_pose": multiview_hand_pose.load_hand_keypoints(),
    }


def normalize_pose_keypoints() -> dict[str, list[dict]]:
    from training.detectors.external_loaders import coco_pose, mpii_pose
    return {
        "coco_pose": coco_pose.load_pose_keypoints(),
        "mpii_pose": mpii_pose.load_pose_keypoints(),
    }


def normalize_face_bbox() -> dict[str, list[dict]]:
    from training.detectors.external_loaders import wider_face
    return {"wider_face": wider_face.load_face_bbox()}


TASK_TO_FN = {
    "hand_bbox": normalize_hand_bbox,
    "hand_keypoints": normalize_hand_keypoints,
    "pose_keypoints": normalize_pose_keypoints,
    "face_bbox": normalize_face_bbox,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, default="all")
    ap.add_argument("--val-frac", type=float, default=0.05,
                    help="Fraction of combined items to hold out as validation split (per task)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tasks = [t for t in TASKS if t != "all"] if args.task == "all" else [args.task]

    for task in tasks:
        print(f"\n=== task: {task} ===")
        per_source = TASK_TO_FN[task]()
        out_dir = OUT_ROOT / task
        combined: list[dict] = []
        for source, items in per_source.items():
            if not items:
                continue
            _write_manifest(out_dir / f"external_{source}.json", task, items)
            combined.extend(items)
        if combined:
            train, val = _split_train_val(combined, args.val_frac, args.seed)
            _write_manifest(out_dir / "external_combined.json", task, combined)
            _write_manifest(out_dir / "external_train.json", task, train)
            _write_manifest(out_dir / "external_val.json", task, val)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
