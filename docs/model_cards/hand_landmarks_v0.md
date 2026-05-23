# Model card — hand_landmarks v0

> Per ADR 0012 Rule 5, every model card declares pretrained components,
> alongside architecture, parameter count, training data provenance,
> training process, validation metrics, and known failure modes.

## Overview

- **Model:** `hand_landmarks_v0`
- **Task:** regress 21 hand keypoints (MediaPipe-compatible topology, but
  trained from scratch — no MediaPipe weights or labels involved) from a
  cropped single-hand RGB patch.
- **Status:** trained — first production checkpoint
- **Checkpoint:** `/runs/hand_landmarks_v0_20260521_105603Z/best.pt` on Modal volume `asl-mastery-data`
- **Architecture file:** `training/detectors/hand_landmarks.py`
- **Training entrypoint:** `training/detectors/train_landmarks.py` (local), `training/modal_app.py::train_hand_landmarks` (Modal)
- **Governing ADRs:** [0011](../decisions/0011-landmarks-and-templates-pivot.md), [0012](../decisions/0012-strict-from-scratch-cv-constraint.md), [0015](../decisions/0015-external-cv-datasets-provenance.md)

## Pretrained components

**None.** Every weight in this model was trained by this project.
Initialization: Kaiming-normal for conv / transposed-conv weights,
zeros for biases, ones for BatchNorm weights. No `torchvision.models`
imports. No `load_state_dict` call reading foreign weights.

## Architecture

- Direct keypoint-coordinate regressor (not heatmap-based).
- Convolutional backbone with progressive downsampling, terminating
  in global pooling and an MLP head that outputs 21 × 2 normalized
  keypoint coordinates.
- Input: 224 × 224 RGB hand crop (square, padded if needed), values in [0, 1].
- Output: 42-dim vector (21 keypoints × {x, y}) in [0, 1] crop-relative coordinates.

**Parameter count:** 4,292,789 (~4.3M), measured from `best.pt`.

## Training data

- **Source pool:** external hand-keypoint datasets vetted per ADR 0015
  (human-labeled, sensor-derived, or multi-view-fit only). FreiHAND
  (multi-view fit) and CMU HandDB (multi-view fit). MediaPipe-labeled
  datasets excluded.
- **Manifests:** `/labeled_frames/hand_keypoints/external_{train,val}.json`
  on the Modal volume.
- **Provenance audit:** `docs/data/external_datasets_audit.md`.
- **Multiview Hand Pose:** dropped — both upstream sources serve
  corrupt archives. See audit memo § "Multiview drop".

## Training process

- **Loss:** mean L2 over keypoint coordinates, in crop-normalized space.
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4.
- **Schedule:** cosine annealing across 30 epochs.
- **Gradient clipping:** max-norm 5.0.
- **Augmentation:** random crop + horizontal flip + photometric +
  rotation (keypoints transformed accordingly). See
  `training/detectors/landmarks_augment.py`.
- **Epochs:** 30.
- **Batch size:** 32.
- **GPU:** Modal L4.
- **Wall time:** ~3 hr.

## Validation

- **Held-out set:** `/labeled_frames/hand_keypoints/external_val.json`.
- **Primary metric:** mean per-keypoint pixel error at 224 × 224.
- **Final val loss:** 0.0360 (normalized L2), best epoch 28.
- **Final mean keypoint pixel error:** 12.82 px @ 224 × 224 (≈ 5.7% of
  crop edge).
- **Trajectory:** 32.07 px (epoch 1) → 12.82 px (epoch 30), monotonic.
- **Roadmap target:** < 8 px @ 224 × 224. **Result: above target.**
  Possible remediations: 60-epoch run on A100, or self-training round
  against ASL corpus pseudo-labels. Decision deferred until
  end-to-end Phase 4 sign-accuracy is measured.

## Known failure modes

- Fingertips on motion-blurred frames likely the worst sub-population
  (typical for this regressor family — to be verified after Phase 4).
- Heavily occluded fingers (e.g., thumb behind palm) may regress
  toward the dataset mean position.
- Skin-tone subgroup error not yet measured.

## Bias notes

- Dataset skews toward certain Fitzpatrick types per ADR 0015 audit.
- Per ADR 0012, fairness mitigation is via additional labeled data and
  augmentation, not by importing a pretrained model.

## Browser inference

- Export: not yet wired (Phase 5 — `training/detectors/export_onnx.py`).
- Target latency: ≥ 30 FPS at 224 × 224; cold start < 200 ms.

## Changelog

- **v0 (2026-05-21):** first production checkpoint. Final val 0.036 /
  12.82 px. Above the < 8 px target; decision deferred pending Phase 4
  end-to-end accuracy.
