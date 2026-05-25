# Model card — hand_landmarks v2_combined

> Per ADR 0012 Rule 5, every model card declares pretrained components,
> alongside architecture, parameter count, training data provenance,
> training process, validation metrics, and known failure modes.
> Supersedes [hand_landmarks_v0](hand_landmarks_v0.md) as the production 2D
> landmark regressor (same arch, retrained on in-the-wild data).

## Overview

- **Model:** `hand_landmarks_v2_combined`
- **Task:** regress 21 hand keypoints (MediaPipe-compatible topology, trained
  from scratch — no MediaPipe weights or labels) from a cropped single-hand RGB
  patch. **2D only** (`predict_z=False`; z is shelved — see TRAINING.md).
- **Status:** trained — current production checkpoint (replaces v0).
- **Checkpoint:** `/runs/hand_landmarks_v2_combined/best.pt` on Modal volume
  `asl-mastery-data`; durable copy `/models/hand_landmarks_v2_combined/`; local
  `data/ckpts_new/hand_landmarks_v2_combined_best.pt`.
- **Architecture file:** `training/detectors/hand_landmarks.py`
- **Training entrypoint:** `training/modal_app.py::train_hand_landmarks_packed` (A100)
- **Per-source eval entrypoint:** `training/modal_app.py::eval_landmarks_per_source` (L4)
- **Governing ADRs:** [0011](../decisions/0011-landmarks-and-templates-pivot.md), [0012](../decisions/0012-strict-from-scratch-cv-constraint.md), [0015](../decisions/0015-external-cv-datasets-provenance.md)

## Pretrained components

**None.** Every weight trained by this project. Same initialization scheme as
v0 (Kaiming-normal conv, zero bias, ones BatchNorm). No foreign `load_state_dict`.

## Architecture

Identical to v0: direct keypoint-coordinate regressor (not heatmap-based),
conv backbone → global pool → MLP head → 21 × 2 normalized coords. Input
224 × 224 RGB crop in [0, 1]. **Parameter count:** ~4.3M (same arch as v0;
checkpoint is byte-for-byte the same size).

## Training data — the v2 change

The reason for the retrain: v0 only ever saw FreiHAND (green-screen) + CMU, so
it was **out-of-distribution on real webcam hands** (bunched dots, palm-forward,
close-ups, two-hand). v2 adds **COCO-WholeBody** (human-annotated, in-the-wild,
ADR-0015-clean, CC-BY).

- **Combined packed set:** `/labeled_frames/hand_keypoints/packed_combined.*`
  (17.8 GB uint8 memmap, 224² crops):
  **FreiHAND 32,560 + CMU-manual 2,758 + COCO-WholeBody 82,664 = 117,982 hands.**
- CMU synth/multiview **skipped** (per-file `exists()` over ~47k files is
  pathologically slow on the volume).
- Provenance audit: `docs/data/external_datasets_audit.md`.

## Training process

- **Loss:** mean L2 over keypoint coords (crop-normalized), vis-masked.
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4. Cosine anneal, **60 epochs**.
- **Batch size:** 512. **GPU:** Modal A100.
- **Augmentation:** GPU-side (`landmarks_gpu_augment.gpu_augment_batch`), scale
  aug **widened 0.65–1.40** (was 0.85–1.10) to cover close-ups.
- **Pipeline:** packed uint8 memmap localized to /tmp; uint8 host→device
  transfer (`.float()/255` on GPU); `pin_memory` + `persistent_workers` +
  `prefetch_factor=4`. **~52 s/epoch (~52 min total)** — see the speed note in
  TRAINING.md / `memory/feedback_packed_train_pipeline_opt.md` (159→59 s/epoch).
- **Split:** `val_frac=0.03`, `seed=42`; `best.pt` saved at min val-loss
  (epoch 48, val 0.0543).

## Validation — per-source px (the real comparison)

Held-out val split (seed=42, val_frac=0.03), metric = vis-masked
mean-per-keypoint pixel error @ 224 (same formula as v0). The combined val is
**harder** than v0's FreiHAND-only val, so only the per-source split is
apples-to-apples:

| Source bucket | v2_combined | v0 |
| --- | --- | --- |
| **FreiHAND+CMU (clean)** | **10.84 px** | 12.82 px |
| **COCO-WholeBody (in-the-wild)** | **21.61 px** | — (never trained/measured) |
| Overall (combined val) | 18.33 px | — |

**Takeaways:**
- On the clean distribution, v2 is **~2 px better than v0** (10.84 vs 12.82) —
  despite splitting capacity across 3.6× more, harder data. Adding COCO did not
  cost clean-hand accuracy; it improved it.
- v2 now has a **real in-the-wild number (21.61 px)** — a distribution v0 was
  completely blind to. This is the honest webcam-hand baseline and the next
  lever to push down.
- Overall 18.33 is dragged up by the hard COCO bucket, not a regression.

## Known failure modes

- **In-the-wild (COCO) at 21.61 px** is the worst bucket — the next target.
- **Two-hand double-box / boxes-on-faces is a DETECTOR issue, not landmarks.**
  The fix is a from-scratch hand-detector retrain on combined hand-boxes (next
  lever), not this regressor.
- Heavily occluded fingers may regress toward the dataset mean.
- Skin-tone subgroup error not yet measured.

## Changelog

- **v2_combined (2026-05-24):** retrained from scratch on FreiHAND + CMU +
  COCO-WholeBody (117,982 hands), 60 epochs A100. Clean-bucket 10.84 px (beats
  v0's 12.82); first in-the-wild number 21.61 px (COCO). 2D only; z shelved.
