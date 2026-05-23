"""HaGRID normalizer — legacy Sbercloud per-image-JSON path.

⚠️  SESSION-16 NOTE: This loader is SECONDARY under the current ingest plan.
The primary path is `training/modal_app.py::download_hagrid_from_hf`,
which streams the `cj-mills/hagrid-sample-500k-384p` HuggingFace mirror.
That mirror's schema does NOT include `hand_landmarks` or `meta`, so
the ADR-0012 compliance posture is structural (the disqualified fields
do not exist in our data source) rather than defensive (strip them at
ingest time).

This file is kept active because:
  1. The `_check_no_landmark_drift()` tripwire is reusable defense-in-
     depth — the HF entrypoint also calls it on each emitted record so
     a future schema change would still trip it.
  2. If we ever fall back to the upstream Sbercloud per-image-JSON
     format, this is the loader that handles it.

Upstream HaGRID's per-image JSONs ship four kinds of label data in the same file:

    {
      "bboxes":         human-drawn by Toloka crowdworkers  ← APPROVED, USE THIS
      "labels":         human-chosen gesture class          (unused; we do detection)
      "united_bbox":    derived, unused
      "united_label":   derived, unused
      "hand_landmarks": MEDIAPIPE-GENERATED                 ← ADR-0012 VIOLATION
      "meta":           FairFace + MiVOLO neural networks   ← ADR-0012 VIOLATION
      "user_id":        opaque identifier; unused
    }

If you're tempted to read `hand_landmarks` "just to crop better" or to
bootstrap a Phase 2 landmark model — DON'T. Re-read ADR-0012 §"Datasets
labeled by pretrained CV models are disqualified" and bring the question
to a new ADR review.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


# The two HaGRID fields that, if ever read into our training pipeline,
# constitute an ADR-0012 violation. Listed here so a grep across the
# repo immediately surfaces every place they could be touched.
_FORBIDDEN_HAGRID_FIELDS = ("hand_landmarks", "meta")

# These are dropped because we don't need them for detection training,
# not because they're prohibited.
_UNUSED_HAGRID_FIELDS = ("labels", "united_bbox", "united_label",
                         "user_id", "leading_hand", "leading_conf")


def _check_no_landmark_drift(rec: dict) -> None:
    """Defensive guard. Asserts the emitted FrameRecord-shaped dict has
    NO field name that matches a forbidden token. Tokens are checked at
    underscore-separated word granularity to avoid false positives like
    "image_path" matching "age". If this ever raises, it means new code
    accidentally widened the schema in a way that could re-introduce
    MediaPipe / FairFace / MiVOLO contamination."""
    forbidden_tokens = {
        "landmark", "landmarks", "keypoint", "keypoints", "kpt", "kpts",
        "age", "gender", "race", "ethnicity",
        "fairface", "mivolo", "mediapipe", "openpose",
        "skeleton", "joint", "joints", "pose", "poses",
        "meta",
    }
    # Allow-list the schema we expect (anything else triggers review).
    expected_fields = {"image_path", "width", "height", "bboxes"}
    for key in rec.keys():
        if key in expected_fields:
            continue
        # An unexpected key that hits a forbidden token name → drift.
        tokens = set(key.lower().replace("-", "_").split("_"))
        bad = tokens & forbidden_tokens
        if bad:
            raise AssertionError(
                f"ADR-0012 drift: HaGRID FrameRecord carries forbidden "
                f"field '{key}' (matched tokens: {sorted(bad)}). "
                "Review training/detectors/external_loaders/hagrid.py"
            )
        # An unexpected key without a forbidden token → soft warning by
        # raising too, so additions to the schema are intentional.
        raise AssertionError(
            f"ADR-0012 drift: HaGRID FrameRecord carries unexpected "
            f"field '{key}' not in {expected_fields}. If this is intentional, "
            "update expected_fields in hagrid.py:_check_no_landmark_drift."
        )


def iter_hagrid_records(
    annotations_root: Path,
    images_root: Path,
) -> Iterator[dict]:
    """Yield FrameRecord-shaped dicts from HaGRID's annotation JSONs.

    Each input JSON is a flat map: {image_id: {bboxes, labels, ...}}.
    For each (image_id, entry) we yield exactly one record with:
        - image_path:  absolute path on the Modal volume
        - width, height: HaGRID images are FullHD (1920×1080)
        - bboxes:      converted from normalized COCO [x, y, w, h] to
                       absolute-pixel xyxy [x0, y0, x1, y1]

    No other field crosses this boundary. Drift guard runs on every emit.
    """
    for jpath in sorted(annotations_root.glob("*.json")):
        try:
            data = json.loads(jpath.read_text())
        except Exception:
            continue
        gesture_class = jpath.stem  # e.g. "call", "stop", ...
        class_img_dir = images_root / gesture_class
        for image_id, entry in data.items():
            # Strict: only touch the bboxes field. Everything else is
            # ignored — including the forbidden hand_landmarks / meta.
            raw_bboxes = entry.get("bboxes")
            if not raw_bboxes:
                continue
            # HaGRID FullHD = 1920×1080 (per WACV 2024 paper §3).
            W, H = 1920, 1080
            abs_bboxes = []
            for b in raw_bboxes:
                # COCO normalized: [x, y, w, h] in [0, 1]
                if len(b) < 4:
                    continue
                nx, ny, nw, nh = b[0], b[1], b[2], b[3]
                if nw <= 0 or nh <= 0:
                    continue
                x0 = nx * W
                y0 = ny * H
                x1 = (nx + nw) * W
                y1 = (ny + nh) * H
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                abs_bboxes.append([x0, y0, x1, y1])
            if not abs_bboxes:
                continue
            img_path = class_img_dir / f"{image_id}.jpg"
            rec = {
                "image_path": str(img_path),
                "width": W,
                "height": H,
                "bboxes": abs_bboxes,
            }
            _check_no_landmark_drift(rec)  # tripwire
            yield rec


def build_hagrid_manifest(
    annotations_root: Path,
    images_root: Path,
    out_path: Path,
    limit: int | None = None,
) -> dict:
    """Drive iter_hagrid_records and write a FrameRecord-format manifest.

    Output matches `data/labeled_frames/hand_bbox/external_train.json`
    schema exactly, so downstream code (dataset.py:load_manifest, etc.)
    can mix HaGRID with CMU + FreiHAND without any branching.
    """
    items: list[dict] = []
    for rec in iter_hagrid_records(annotations_root, images_root):
        items.append(rec)
        if limit and len(items) >= limit:
            break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "task": "hand_bbox",
        "source": "hagrid_v2",
        "notes": (
            "Generated from HaGRID v2 with BBOX-ONLY ingestion. "
            "hand_landmarks and meta fields IGNORED per ADR-0012. "
            "See training/detectors/external_loaders/hagrid.py."
        ),
        "items": items,
    }
    out_path.write_text(json.dumps(payload))
    return {"n_records": len(items), "out_path": str(out_path)}
