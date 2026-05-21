# Model card — hand_detector v0 (TEMPLATE, awaiting first training run)

> Per ADR 0012 Rule 5, every model card declares pretrained components,
> alongside architecture, parameter count, training data provenance,
> training process, validation metrics, and known failure modes.
> Per `docs/VOCABULARY_TRAINER_ROADMAP.md` Phase 1 Slice 1.2, this card
> is filled in after the first training run lands.

## Overview

- **Model:** `hand_detector_v0`
- **Task:** axis-aligned hand bounding-box detection from a single RGB frame
- **Status:** TEMPLATE — first training run not yet executed
- **Architecture file:** `training/detectors/hand_detector.py` (`HandDetector`)
- **Training entrypoint:** `training/detectors/train.py` (local), `training/modal_app.py::train_hand_detector` (Modal)
- **Governing ADRs:** [0011](../decisions/0011-landmarks-and-templates-pivot.md), [0012](../decisions/0012-strict-from-scratch-cv-constraint.md)

## Pretrained components

**None.** Every weight in this model was trained by this project.
Initialization: Kaiming-normal for conv / transposed-conv weights,
zeros for biases (with a CenterNet-standard −2.19 bias on the final
heatmap conv for focal-loss stability), ones for BatchNorm weights.
No `torchvision.models` imports. No `load_state_dict` call reading
foreign weights anywhere in the code path. Verified by ADR 0012's
audit greps.

## Architecture

- CenterNet-style single-stage anchor-free detector.
- Stem: 2 × (3×3 conv + BN + ReLU), input stride 2.
- 3 residual stages with downsampling: 32 → 64 → 128 → 192 channels.
- FPN-style upsample with two transpose-conv stages, with lateral
  skips from blocks 1 and 2.
- Two heads: heatmap (1 channel) + size regression (2 channels).
- Input: 320 × 320 RGB, values in [0, 1].
- Output stride: 4. Heatmap and size are at 80 × 80.

**Parameter count:** TODO — run `python -m training.detectors.hand_detector`
and fill in. Target per roadmap: 1–3M parameters.

## Training data

- **Source pool:** frames extracted from the project's cleaned
  12K-clip public-source corpus (WLASL + Lifeprint + targeted YouTube
  search) at 1 FPS via `scripts/extract_frames.py`. Plus self-recorded
  frames covering lighting, distance, and background variation.
- **Labeling rubric:** `labeling/rubrics/hand_bbox.md`.
- **Labeled-frame count:** TODO — target ~3,000 corpus + ~500–1,000
  self-recorded per Slice 1.1.
- **Manifest format:** flat JSON v1 (see
  `training/detectors/dataset.load_manifest`).
- **Train / val split:** TODO — signer-disjoint per
  `docs/DATASET.md` §4 when possible; for self-recorded frames where
  the signer is the project author, hold out a separate session.
- **Per-batch provenance:** every batch under
  `data/labeled_frames/hand_bbox/<batch_id>/` ships with a
  `PROVENANCE.md`.

## Training process

- **Loss:** CenterNet focal loss on heatmap + L1 loss on size at
  ground-truth center pixels. Size loss weight: 0.1.
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4.
- **Schedule:** cosine annealing over the full epoch budget.
- **Gradient clipping:** max-norm 5.0.
- **Augmentation:** random crop + horizontal flip + photometric
  (brightness / contrast / saturation / hue) + occasional motion
  blur. See `training/detectors/augment.py`.
- **Epochs:** TODO — initial target 60.
- **Batch size:** TODO — initial target 16 at 320 × 320.
- **GPU:** Modal — see `training/modal_app.py` GPU constant.

## Validation

- **Held-out set size:** TODO — target ~500 frames from unseen
  signers / conditions per Slice 1.3.
- **Primary metric:** AP @ IoU=0.5. Target ≥ 0.85 per Slice 1.3.
- **Secondary metrics:** AP @ IoU=0.75, per-bin recall by box scale
  (small / medium / large hands), per-Fitzpatrick recall (subgroup
  reporting per `docs/EVAL_GATE.md` §1).
- **Latest run result:** TODO.

## Known failure modes

- TODO — fill in after the first training run identifies systematic
  errors. Candidates to expect: motion-blurred hands, hands at frame
  edges, hands partially occluded by face / body, low-light frames,
  hands on backgrounds similar to skin color.

## Bias notes

- The corpus is dominated by certain Fitzpatrick types and lighting
  conditions per the source pool. Per-Fitzpatrick recall is reported
  on the held-out set; gaps inform targeted labeling in subsequent
  iterations rather than data hiding.
- Per ADR 0012, fairness mitigation cannot include swapping in a
  pretrained detector. Mitigation is via additional labeled data and
  augmentation choices.

## Browser inference

- Export: PyTorch → ONNX (TODO — Slice 1.4 to add export entrypoint).
- Target latency: ≥ 30 FPS at 320 × 320 on dev laptop; cold start
  < 200 ms.
- INT8 quantization considered if FP32 is too slow.

## Changelog

- **v0 (template, pre-first-run):** card scaffolded alongside ADR 0011
  and architecture file commit (Phase 1 Slice 1.2). All TODO fields
  are placeholders for the first real training run.
