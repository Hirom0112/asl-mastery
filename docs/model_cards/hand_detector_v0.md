# Model card — hand_detector v0

> Per ADR 0012 Rule 5, every model card declares pretrained components,
> alongside architecture, parameter count, training data provenance,
> training process, validation metrics, and known failure modes.

## Overview

- **Model:** `hand_detector_v0`
- **Task:** axis-aligned hand bounding-box detection from a single RGB frame
- **Status:** trained — first production checkpoint
- **Checkpoint:** `/runs/hand_det_v0_a100_resume_20260521_145244Z/best.pt` on Modal volume `asl-mastery-data`
- **Architecture file:** `training/detectors/hand_detector.py` (`HandDetector`)
- **Training entrypoint:** `training/detectors/train.py` (local), `training/modal_app.py::train_hand_detector` (Modal)
- **Governing ADRs:** [0011](../decisions/0011-landmarks-and-templates-pivot.md), [0012](../decisions/0012-strict-from-scratch-cv-constraint.md), [0015](../decisions/0015-external-cv-datasets-provenance.md)

## Pretrained components

**None.** Every weight in this model was trained by this project.
Initialization: Kaiming-normal for conv / transposed-conv weights,
zeros for biases (with a CenterNet-standard −2.19 bias on the final
heatmap conv for focal-loss stability), ones for BatchNorm weights.
No `torchvision.models` imports. No `load_state_dict` call reading
foreign weights anywhere in the code path.

## Architecture

- CenterNet-style single-stage anchor-free detector.
- Stem: 2 × (3×3 conv + BN + ReLU), input stride 2.
- 3 residual stages with downsampling: 32 → 64 → 128 → 192 channels.
- FPN-style upsample with two transpose-conv stages, with lateral
  skips from blocks 1 and 2.
- Two heads: heatmap (1 channel) + size regression (2 channels).
- Input: 320 × 320 RGB, values in [0, 1].
- Output stride: 4. Heatmap and size are at 80 × 80.

**Parameter count:** 2,305,464 (~2.3M), measured from `best.pt`.

## Training data

- **Source pool:** external public hand-detection datasets vetted per
  ADR 0015 (human-labeled, sensor-derived, or multi-view-fit
  provenance). pretrained-model-labeled datasets excluded.
- **Manifests:** `/labeled_frames/hand_bbox/external_{train,val}.json`
  on the Modal volume.
- **Project-local labeling rubric:** `labeling/rubrics/hand_bbox.md`.
- **Provenance audit:** `docs/data/external_datasets_audit.md`.

## Training process

- **Loss:** CenterNet focal loss on heatmap + L1 loss on size at
  ground-truth center pixels. Size loss weight: 0.1.
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4.
- **Schedule:** cosine annealing over the per-stage epoch budget.
- **Gradient clipping:** max-norm 5.0.
- **Augmentation:** random crop + horizontal flip + photometric
  (brightness / contrast / saturation / hue) + occasional motion
  blur. See `training/detectors/augment.py`.
- **Schedule:** two stages. Stage 1: 16 epochs on L4, batch 16
  (run `hand_det_v0_20260521_105602Z`). Stage 2: resumed for 14
  epochs on A100 80GB, batch 64 (run
  `hand_det_v0_a100_resume_20260521_145244Z`).
- **GPU:** Modal L4 then A100 80GB.
- **Total wall time:** ~7.2 hr L4 + ~1.4 hr A100 ≈ 8.6 hr.

## Validation

- **Held-out set:** `/labeled_frames/hand_bbox/external_val.json`.
- **Primary metric:** sum of focal + 0.1·L1 loss (heatmap + size).
- **Final val loss:** 1.743 (epoch 30, best — `epoch 14` of the A100 resume run).
- **Loss trajectory (A100 resume):** 2.041 → 1.743 monotonically across 14 epochs.
- **AP @ IoU=0.5:** not yet computed. Will be measured during Phase 4
  end-to-end evaluation. Target per roadmap Slice 1.3: ≥ 0.85.

## Known failure modes

- Per-bin recall (small / medium / large hands) and per-Fitzpatrick
  recall not yet measured; to be filled in after first end-to-end
  Phase 4 evaluation.
- Candidates to expect: motion-blurred hands, hands at frame edges,
  hands partially occluded by face / body, low-light frames, hands
  on backgrounds similar to skin tone.

## Bias notes

- External corpus is dominated by certain Fitzpatrick types and
  lighting conditions. Subgroup metrics will be reported after Phase 4.
- Per ADR 0012, fairness mitigation cannot include swapping in a
  pretrained detector. Mitigation is via additional labeled data and
  augmentation choices.

## Browser inference

- Export: not yet wired (Phase 5 — `training/detectors/export_onnx.py`).
- Target latency: ≥ 30 FPS at 320 × 320 on dev laptop; cold start
  < 200 ms.

## Changelog

- **v0 (2026-05-21):** first production checkpoint. Final val loss
  1.743 across 30 epochs (16 L4 + 14 A100 resume). Trained from
  scratch on external bbox datasets vetted per ADR 0015.
