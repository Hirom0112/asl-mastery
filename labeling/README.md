# `/labeling` — internal keypoint and bounding-box labeling tool

Per the project structure in `docs/VOCABULARY_TRAINER_ROADMAP.md`.

## Purpose

A local, offline labeling surface for the from-scratch CV pipeline:

- Hand bounding boxes (Phase 1, Slice 1.1 — ~3,000+ frames)
- Hand keypoints, 21 per visible hand (Phase 2, Slice 2.1 — 5,000–10,000 frames, bootstrapped)
- Pose keypoints, 8 upper-body (Phase 3, Slice 3.1 — ~2,000 frames)
- Face bounding boxes (Phase 3, Slice 3.2 — ~1,500 frames)

## Tool choice

To be decided in Slice 1.1 between **CVAT** (self-hosted Docker, full-featured) and **LabelStudio** (lighter weight, also self-hostable). The labeling tool must:

1. Run entirely on localhost (no third-party label hosting).
2. Export to a format the `/ml` dataset classes can consume directly (COCO bbox JSON for detection; flat per-image keypoint JSON for landmarks).
3. Support **model-assisted pre-labeling** using project-trained models for the bootstrapping loop (Phase 2 critical-path mitigation).

Pretrained pre-labeling models from third parties are prohibited
under `docs/decisions/0012-strict-from-scratch-cv-constraint.md`.
Only models trained by this project are wired into pre-labeling.

## Layout (to be created during Slice 1.1)

```
labeling/
├── README.md            # this file
├── tool/                # the labeling tool config / wrapper scripts
├── rubrics/             # written labeling rubrics per task (hand_bbox.md, hand_keypoints.md, …)
├── exports/             # raw exports from the tool (gitignored — see /data/labeled_frames/ for the canonical home)
└── runs/                # per-labeling-session metadata (gitignored)
```
