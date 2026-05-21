# ADR 0001: Recognition architecture is a small 3D CNN trained end-to-end

**Status:** Accepted — reinstated as the governing recognition-architecture decision.
**Superseded by:** ADR 0006 on 2026-05-19 following an earlier permissive reading of brief Requirement 7. **Reinstated by [ADR 0010](./0010-reversal-of-adr-0006.md) on 2026-05-20** after that permissive reading was withdrawn and the strict reading of Requirement 7 governs again. The body below is the current architectural specification under ADR 0010, not historical.
**Date:** initial scoping

---

## Context

The brief (Requirement 7) prohibits pretrained models — including
pretrained sign classifiers, pretrained hand or pose landmark
detectors, pretrained feature extractors, and pretrained general-
purpose CV backbones. This rules out the most common ASL recognition
path, which is MediaPipe hand/holistic landmarks → small classifier.

Three remaining paths are viable in principle:

- **Path A:** train our own landmark detector from scratch, then train
  a classifier on the landmarks.
- **Path B:** train a video classifier end-to-end on short clips of
  raw pixels.
- **Path C:** use classical (non-learned) CV — skin segmentation,
  optical flow, background subtraction — to produce intermediate
  representations, then train a small classifier on those.

---

## Decision

We adopt Path B: end-to-end small 3D CNN (R(2+1)D-style), trained from
scratch, on 16-frame clips at 112×112 resolution. Classical CV
techniques (Path C) are usable as data augmentation but not as core
inference.

---

## Rationale

- **Path A is research-scale work.** Training a landmark detector
  competitive with MediaPipe from scratch is a multi-year project
  with its own dataset and architecture trade-offs. Out of scope.
- **Path C is brittle.** Classical CV pipelines depend on hand-tuned
  thresholds (skin color ranges, motion thresholds) that fail across
  lighting and skin tone. Building a fairness-respecting system on
  Path C is harder than Path B.
- **Path B is the honest reading of the constraint.** It is what the
  brief implies if read strictly. It is also the most defensible in
  the writeup ("we trained a small CNN from scratch on a dataset we
  curated"), which matters for a project where the no-pretrained
  constraint is a deliberate signal of what is being evaluated.

Path B's known weakness is data hunger. We mitigate this by:

- Layering public ASL datasets (pending approval) on top of self-
  recorded data.
- Aggressive augmentation, including classical-CV-based background
  swap that does not depend on a learned model.
- A small architecture (~3M params) that does not require millions of
  clips to train usefully.

---

## Alternatives considered and rejected

- **MediaPipe Hands / Holistic → small classifier.** Banned by
  Requirement 7.
- **3D ResNet-50.** Too large to ship to the browser at <10 MB; too
  data-hungry for our likely dataset size.
- **Transformer-based video model (e.g., VideoSwin).** Even more
  data-hungry; harder to train from scratch on our scale.
- **CNN+LSTM.** Plausible alternative; slightly more interpretable
  than R(2+1)D but typically lower accuracy at the same parameter
  count. Kept as a fallback if R(2+1)D underperforms on our data.

---

## Consequences

- The dataset effort becomes the project's bottleneck. We must invest
  there.
- Browser inference latency becomes the project's secondary bottleneck.
  We commit to int8 quantization and a <10 MB model.
- Architecture-level signing of correctness (e.g., "your handshape
  was a B but should be a 5") is not available from the model
  directly. Hints are confusion-pair-based for slice 1; multi-head
  parameter prediction is a slice-2 candidate.

---

## How this is verified

The training entry point in the repo uses only `torch.nn.init.kaiming_normal_`
for weight initialization and contains no `load_state_dict` call or
external weight URL. A reviewer auditing the no-pretrained claim can
read `training/init.py` and `training/model/r2plus1d_small.py`.
