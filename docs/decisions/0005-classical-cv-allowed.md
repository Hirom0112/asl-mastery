# ADR 0005: Classical CV libraries are allowed for augmentation and quality checks, not core inference

**Status:** Accepted. Classical-CV augmentation — MOG2 background swap chief among it — is load-bearing for training-time augmentation, exactly as the body below specifies.
**Date:** 2026-05-19

---

## Context

The brief's Requirement 7 bans pretrained models — pretrained
classifiers, landmark detectors, feature extractors, and CV backbones.
It does not address classical (non-learned) computer vision: MOG2
background subtraction, dense / sparse optical flow, skin segmentation
in HSV space, perceptual hashing, and similar hand-coded algorithms.

These techniques have real value in the pipeline:

- **MOG2 background subtraction** off the 1-second empty-frame clip
  captured at session start enables background-swap augmentation,
  which is one of our main defenses against overfitting to dorm-room
  backgrounds.
- **Optical flow** and motion-energy features are candidate quality
  checks for empty / unmoving / partially-out-of-frame clips during
  recording-tool ingestion.
- **Perceptual hashing** (pHash) drives intra-signer deduplication in
  the cleaning pipeline (`docs/DATASET.md` §3).

The user confirmed verbally during Session 1 that these are allowed;
Session 2 recorded the confirmation in writing in `claude/CLAUDE.md`
§4. This ADR makes the decision permanent and reviewable on its own.

---

## Decision

Classical computer vision algorithms — implemented as hand-coded
procedures over pixel arrays, with no learned parameters — are
permitted in the system for two purposes:

1. **Training-time data augmentation.** MOG2-based background swap is
   the canonical example.
2. **Pipeline quality checks and bookkeeping.** Perceptual hashing for
   dedup, motion-energy thresholds for empty-clip rejection, framing
   sanity checks against the green box.

Classical CV is **not** permitted in the live inference path. The only
learned model used at inference time is the R(2+1)D-small classifier
defined in `docs/MODEL.md`. This keeps the no-pretrained defense
clean: the inference path is one trained-from-scratch CNN, not a
hybrid of classical preprocessing and learned classification whose
attribution is harder to reason about.

---

## Rationale

- **These algorithms are not models.** A Gaussian mixture, an HSV
  threshold, a phase-correlation tracker — these are mathematical
  procedures with no fitted parameters that depend on prior training
  data. They are categorically different from "load these weights
  somebody else trained."
- **Confining classical CV to augmentation and bookkeeping preserves
  the cleanest no-pretrained story.** "The model that decides whether
  the learner signed correctly was trained from scratch on data we
  curated; classical CV touched the *training data* but not the
  inference call" is a defense that fits on one slide.
- **The brief's letter and spirit both allow this.** Requirement 7
  bans pretrained models. Requirement 6 explicitly permits
  engineer-curated data and standard CV libraries (OpenCV, NumPy).
  Classical algorithms live unambiguously inside Requirement 6's
  scope.

---

## Alternatives considered and rejected

- **No classical CV anywhere.** Rejected. Background-swap augmentation
  is one of the highest-leverage tools we have against the dorm-room
  background-bias failure mode; giving it up would shift more burden
  onto raw data collection, which is already the project's bottleneck.
- **Classical CV in the inference path.** Rejected on the principle
  above: hybrid pipelines complicate the no-pretrained defense and
  introduce hand-tuned thresholds that historically fail across skin
  tone and lighting (see ADR 0001's argument against Path C). Keeping
  inference as one CNN over normalized RGB pixels is cleaner.

---

## Consequences

- The training pipeline imports OpenCV (or equivalent) for MOG2,
  optical flow, and pHash. The recording-tool ingestion code may use
  the same library for empty-clip detection.
- The inference bundle does not import OpenCV or any classical-CV
  library; the browser runtime is ONNX Runtime Web and standard
  canvas / WebGPU APIs only.
- `docs/MODEL.md` §7 references this ADR for the classical-CV
  exception to the no-pretrained-models defense.

---

## How this is verified

A reviewer auditing the no-pretrained claim and the inference path
can confirm:

- The browser bundle has no OpenCV or classical-CV library imports
  (checkable in the Next.js production build).
- The training pipeline's augmentation module is the only place
  classical CV appears in the learning path (`training/data/augment/`).
- The inference function in the learner app calls ONNX Runtime Web
  directly on a normalized tensor, with no intermediate classical-CV
  step.
