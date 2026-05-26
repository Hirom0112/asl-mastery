# ADR 0011: Recognition architecture — from-scratch landmarks + classifier

**Status:** Accepted
**Date:** 2026-05-20
**Relates to:** [ADR 0012](./0012-strict-from-scratch-cv-constraint.md) (the constraint perimeter this architecture honors), [ADR 0015](./0015-external-cv-datasets-provenance.md) (external labeled-dataset acceptance criteria), [ADR 0005](./0005-classical-cv-allowed.md) (classical CV permitted for augmentation)

---

## Context

The recognition system must turn a webcam clip of a learner signing into a
pass/fail decision over a fixed beginner vocabulary, **with no pretrained vision
components anywhere** (brief Requirement 7, strict reading — see
[ADR 0012](./0012-strict-from-scratch-cv-constraint.md)). The public-source
corpus provides only ~13–20 clips per sign, so the architecture must use that
limited data sample-efficiently.

An end-to-end RGB classifier sized for this corpus is data-starved at that clip
count. A landmark-based pipeline is far more sample-efficient: the detectors
train on **labeled frames** (independent of sign identity), and the classifier
operates on **landmark trajectories** — a representation that already absorbs
visual variance (skin tone, lighting, background, camera position).

## Decision

A **four-stage from-scratch landmark pipeline feeding a from-scratch sign
classifier.** Every weight is trained by this project; each model has a card
under [`docs/model_cards/`](../model_cards) declaring "Pretrained components:
none" and citing a training run.

1. **Hand detector** — CenterNet-style single-stage detector, 320×320, ~2.3M params.
2. **Hand-landmark regressor** — small CNN over 224×224 hand crops, 21 keypoints/hand.
3. **Pose detector** — small CNN, 8 upper-body keypoints, providing the body-relative reference frame.
4. **Face detector** — tiny single-class detector, for framing / onboarding.
5. **Sign classifier** — per frame, the keypoints become a hand-relative feature
   vector; a 32-frame (~2 s) trajectory feeds a from-scratch temporal classifier.
   A distribution-of-templates (Mahalanobis) matcher was built as a baseline; a
   learned classifier outperformed it on the realistic confusion set and is what
   ships.

**Per-sign thresholds, precision-prioritized.** Each sign's pass threshold is
calibrated so a false-pass is rarer than a false-fail (a false-pass is worse in
a learning context). The per-sign threshold table is the artifact brief
Requirement 9 demands.

**Inference runs entirely in the browser** via ONNX Runtime Web (WebGPU, WASM
fallback): hand detector → per-hand crop → landmark regressor; pose + face
detectors in parallel; the trajectory accumulates over the capture window →
classifier → per-sign threshold → pass/fail.

## Why landmarks (not end-to-end RGB)

- **Detectors train on labeled frames, not labeled clips** — one labeled frame
  supports training for any sign, so detector accuracy is bounded by labeling
  budget, not per-sign clip count.
- **The classifier sees trajectories**, which already absorb pixel-level
  variance, so the thin per-sign clip count fits a low-dimensional distribution
  rather than teaching a CNN every intra-class variation.
- **Interpretable pass/fail.** Similarity decomposes along the Stokoe ASL
  parameters (handshape, location, movement, palm orientation), so the "worst
  parameter" of a failed attempt becomes the hint target (see
  [`docs/PEDAGOGY.md`](../PEDAGOGY.md)).

## Constraint compliance

Governed by [ADR 0012](./0012-strict-from-scratch-cv-constraint.md). Every CV
component is inside the CV perimeter and trained from scratch — no pretrained
weights, no pretrained backbones, no foreign `load_state_dict`. The
permitted-dependency list and anti-drift rules live in ADR 0012.

## Data sourcing per detector (per [ADR 0015](./0015-external-cv-datasets-provenance.md))

Detector/landmark **labels** come from public, human-/sensor-/multiview-annotated
datasets — never from a pretrained CV model's predictions (the
substantive-equivalence argument is in ADR 0015). The weights are still trained
from scratch; only the labels are external.

- **Hand detector / landmark regressor:** FreiHAND, CMU HandDB, Multiview Hand
  Pose, COCO-WholeBody (hands) — plus HaGRID via a mirror that ships only
  human-drawn boxes. ~150K–600K labeled hand instances.
- **Pose detector:** MPII Human Pose + COCO Keypoints (shoulders / elbows / wrists).
- **Face detector:** WIDER FACE.
- **Sign classifier:** trained on landmark trajectories extracted from the
  project's own ASL clip corpus (no external keypoint dataset contains ASL
  signing trajectories).

Every external dataset has a `PROVENANCE.md` and an audit entry in
[`docs/data/external_datasets_audit.md`](../data/external_datasets_audit.md).

## Consequences

- **Browser performance:** each ONNX model is small (INT8-quantizable); the four
  detectors plus the classifier run client-side. Per-attempt latency is
  dominated by the ~2 s capture window, not inference.
- **Privacy:** no third-party vision vendor and no remote inference in the
  recognition path — frames are processed locally. See
  [`docs/PRIVACY.md`](../PRIVACY.md).
- **Validation & fairness:** promotion is gated by
  [`docs/EVAL_GATE.md`](../EVAL_GATE.md), including a detector-recall floor and a
  per-Fitzpatrick accuracy-gap ceiling (the detector and landmark regressor are
  where skin-tone bias can enter, so they are the fairness-audit surfaces).
- **Pedagogy:** the three-layer hint system and the parameter-aware "worst
  parameter" hint are natural under this pipeline.

## Rejected alternatives

- **End-to-end RGB classifier.** Rejected: data-starved at ~13–20 clips/sign,
  and its per-sign thresholds (Requirement 9) would be derived from an unreliable
  classifier — below the useful-pilot bar.
- **A pretrained landmark detector + from-scratch matcher.** Rejected: a
  pretrained detector touches pixels, which the strict reading of Requirement 7
  prohibits (the whole point of
  [ADR 0012](./0012-strict-from-scratch-cv-constraint.md)).
- **Pretrained backbone warm-start, then fine-tune.** Rejected: pretrained
  weights touching pixels are out of scope regardless of whether they're later
  updated. The from-scratch story is the moat.
- **Classical-CV preprocessing → small classifier.** Rejected: classical
  thresholds (skin segmentation, optical flow) historically fail across skin tone
  and lighting; a fairness-respecting system is easier to build on landmark
  trajectories. (Classical CV remains permitted for training-time *augmentation*
  per [ADR 0005](./0005-classical-cv-allowed.md).)
- **Single multi-task model** (one forward pass for bbox + keypoints + pose +
  face). Not rejected — investigated-or-deferred; ships only if it beats the
  separate models on speed and accuracy.

## Status of the build

This architecture is **deployed**: the four detectors plus the sign classifier
ship as ONNX in `public/models/` and run in-browser, recognizing **80** beginner
signs at **81.6% top-1** (signer-disjoint). See the [README](../../README.md)
and the per-model cards under [`docs/model_cards/`](../model_cards).
