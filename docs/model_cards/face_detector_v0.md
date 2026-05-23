# Model card — face_detector v0

> Per ADR 0012 Rule 5, every model card declares pretrained components,
> alongside architecture, parameter count, training data provenance,
> training process, validation metrics, and known failure modes.

## Overview

- **Model:** `face_detector_v0`
- **Task:** axis-aligned face bounding-box detection from a single RGB frame
- **Status:** trained — first production checkpoint
- **Checkpoint:** `/runs/face_det_v0_20260521_105723Z/best.pt` on Modal volume `asl-mastery-data`
- **Architecture file:** `training/detectors/face_detector.py` (currently an alias of `HandDetector` from `hand_detector.py` — same backbone, separate weights)
- **Training entrypoint:** `training/detectors/train.py` (local), `training/modal_app.py::train_face` (Modal)
- **Governing ADRs:** [0011](../decisions/0011-landmarks-and-templates-pivot.md), [0012](../decisions/0012-strict-from-scratch-cv-constraint.md), [0015](../decisions/0015-external-cv-datasets-provenance.md)

## Pretrained components

**None.** Every weight in this model was trained by this project.
Initialization identical to `hand_detector` (Kaiming-normal conv,
zeros bias with −2.19 heatmap-conv bias, ones BatchNorm). No
`torchvision.models` imports. No `load_state_dict` reading foreign
weights.

## Architecture

Identical CenterNet-style design to `hand_detector_v0`:

- Stem: 2 × (3×3 conv + BN + ReLU), stride 2.
- 3 residual stages: 32 → 64 → 128 → 192 channels.
- FPN-style upsample with two transpose-conv stages + lateral skips.
- Two heads: heatmap (1 channel) + size regression (2 channels).
- Input: 320 × 320 RGB, [0, 1].
- Output stride: 4. Heatmap and size at 80 × 80.

**Parameter count:** 2,305,464 (~2.3M), measured from `best.pt` —
matches `hand_detector_v0`, confirming shared architecture with
independent weights.

## Training data

- **Source pool:** WIDER FACE (human-labeled), per ADR 0015 vetting.
  Images fetched via gdown on Modal (Google Drive consent token).
- **Manifests:** `/labeled_frames/face_bbox/external_{train,val}.json`
  on the Modal volume.
- **Provenance audit:** `docs/data/external_datasets_audit.md`.

## Training process

- **Loss:** CenterNet focal loss on heatmap + L1 size loss at GT center pixels (weight 0.1).
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4.
- **Schedule:** cosine annealing across 30 epochs.
- **Gradient clipping:** max-norm 5.0.
- **Augmentation:** random crop + horizontal flip + photometric + occasional motion blur.
- **Epochs:** 30.
- **Batch size:** 16.
- **GPU:** Modal L4.
- **Wall time:** ~52 min.

## Validation

- **Held-out set:** `/labeled_frames/face_bbox/external_val.json`.
- **Primary metric:** sum of focal + 0.1·L1 loss.
- **Final val loss:** 1.164 (best, epoch 26). Epoch 30: 1.171.
- **Trajectory:** 2.613 (epoch 1) → 1.164 (epoch 26), monotonic.
- **AP @ IoU=0.5:** not yet computed; Phase 4 end-to-end evaluation.

## Known failure modes

- WIDER FACE is heavy on crowd / small-face scenarios; the model
  may overfire on background patches in clean single-signer ASL clips
  (to be verified during Phase 4).
- Heavily side-profile or occluded faces likely the weakest set.
- Per-Fitzpatrick recall not yet measured.

## Bias notes

- WIDER FACE skews toward certain populations and lighting
  conditions; subgroup metrics will be reported after Phase 4.
- Per ADR 0012, mitigation is via additional labeled data and
  augmentation, not via importing a pretrained model.

## Browser inference

- Export: not yet wired (Phase 5).
- Target latency: ≥ 30 FPS at 320 × 320; cold start < 200 ms.
- Used in the recognition pipeline primarily as a negative-region
  exclusion signal during hand crop selection.

## Changelog

- **v0 (2026-05-21):** first production checkpoint. Final val 1.164
  in ~52 min on L4. Architecture shared with `hand_detector_v0`,
  weights independent.
