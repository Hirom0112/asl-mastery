# ADR 0011: Pivot from end-to-end RGB classifier to landmarks + templates architecture

**Status:** Accepted
**Date:** 2026-05-21
**Supersedes:** [ADR 0001](./0001-recognition-architecture.md) (Path B, end-to-end small 3D CNN on raw RGB) and [ADR 0010](./0010-reversal-of-adr-0006.md) (reversal-of-ADR-0006 reinstatement of ADR 0001). Both kept in the repository as historical record; both get a "Superseded by ADR 0011" header at the top.
**Carries forward:** [ADR 0008](./0008-public-data-only-training.md) (no team-recorded clips for the pilot — revisited per the licensing note below for the shipped templates), [ADR 0009](./0009-asl-citizen-v2.md) (license-aware corpus boundaries — see licensing note).
**Companion:** [ADR 0012](./0012-strict-from-scratch-cv-scoping.md) (strict from-scratch CV constraint and scoping perimeter).

---

## Context

Under the ADR 0001 / ADR 0010 architecture (an end-to-end small R(2+1)D
3D CNN trained from scratch on raw RGB video tensors), the honest
projection for v3.0 against the 75-sign slice-1b vocabulary was
**60–75 % top-1** — materially below the 85 % eval-gate floor
(`docs/EVAL_GATE.md` §1 criterion 1). The dominant bottleneck was
clip volume: ADR 0001 sized Path B at ~200 clips/sign and the merged
WLASL + ASL Citizen + Sem-Lex corpus produced ~150/sign on average
(min 55, median 133, max 571) after the v3 cleaning run on Modal
(2026-05-20 23:33 CDT). Even with the full corpus, the 3D CNN would
have remained an end-to-end-trained pixel classifier whose accuracy
is data-volume-bound and whose interpretability story is "logits in,
logits out."

**NVIDIA Signs** (NVIDIA's shipped public ASL learning prototype, per
their published model card) takes a different shape: **landmarks +
template matching with zero training data on the classifier side**.
The recognition system extracts hand and pose keypoints from each
frame, builds a distribution of per-sign templates from a small
reference corpus, and scores new attempts against those templates
with Mahalanobis (or similar) similarity. The classifier itself is
not a learned model — it's a similarity metric over a small set of
human-readable templates. Threshold calibration is per-sign and
empirical.

Reproducing that approach under the strict reading of brief
Requirement 7 — no pretrained vision components anywhere — means
**we cannot use MediaPipe Holistic, OpenPose, BlazePose, or any
pretrained landmark extractor**. The pivot therefore requires a
*from-scratch* landmark stack: a hand detector, hand landmark
detector, pose landmark detector, and face detector, each trained
from scratch on data we label ourselves. The classifier-side
data-volume problem disappears (templates need ~5–10 clips per sign,
not ~200), but it is replaced by a labeling problem on the detector
side that we have a clearer path through: bounding-box + keypoint
labels on a few thousand frames, augmentable, no semantic
disagreement between annotators the way a sign's "correctness" can
have between hearing engineers.

The supporting evidence for the pivot:

- v3.0's projected ceiling (60–75 % top-1) is comfortably below the
  85 % floor and below NVIDIA Signs' published per-sign accuracies
  on a comparable vocabulary scale.
- Every Path B accuracy gain from here would cost ~$X / iteration in
  L4 hours plus weeks of wall-clock for diminishing returns; the
  underlying ceiling does not move.
- Slice-1b's 11,153 cleaned clips and their normalized 256×256 MP4s
  remain useful under the new architecture (see "Consequences for
  the slice-1b corpus" below).
- The hint system gains a much sharper substrate: with landmarks at
  every frame, "your handshape was wrong" / "your movement was
  shorter than the reference" become checkable statements rather
  than vibes.

---

## Decision

Pivot to a **two-stage pipeline**:

**Stage 1 — from-scratch landmark stack.** Four detectors, each
trained from scratch with no pretrained weights and no pretrained
features anywhere in the perimeter defined by
[ADR 0012](./0012-strict-from-scratch-cv-scoping.md):

1. **Hand detector** — bounding boxes around hands in the frame.
2. **Hand landmark detector** — per-hand 21-keypoint regression.
3. **Pose landmark detector** — upper-body keypoint regression
   (shoulders, elbows, wrists, torso anchors).
4. **Face detector** — bounding box + a small keypoint set for
   non-manual markers (slice-2 candidate; not load-bearing for the
   first shippable build).

**Stage 2 — distribution-of-templates per sign + Mahalanobis-style
similarity + confusion-mined hint generation.** No learned classifier:

- For each of the 75 signs, build a **distribution of templates**
  from a curated subset of the slice-1b corpus — each template is
  the per-frame `(T, K)` keypoint sequence with mean / covariance /
  invariant features.
- A new attempt is scored against every per-sign template
  distribution using a Mahalanobis-style similarity metric over the
  keypoint trajectory.
- **Per-sign confidence thresholds** are calibrated empirically
  against the held-out portion of the corpus, targeting ≥ 90 %
  precision on the "pass" decision (same target as the eval gate
  criterion 4).
- **Confusion-mined hints**: build a confusion graph from the
  template-matching scores, identify the top-K confused pairs per
  sign, and author hints from the *parameter delta* between
  templates (e.g., "your movement endpoint was higher than the
  reference templates for THINK; that's the GOOD endpoint —
  THINK ends with the hand moving forward, not down").

---

## Cancellation record

- **v3.0 training run**: `v3-smoke` (the only smoke-train attempts
  ever launched). Three Modal apps were spawned:
  - `ap-S64dhuRHqpdgG3D6d73W45` at 2026-05-21 00:01 CDT — crashed on `ImportError: PyAV is not installed`.
  - `ap-3mU0Sns7mc68KCO0DcovQo` at 2026-05-21 00:03 CDT — crashed on torchvision/PyAV version mismatch after PyAV was added to requirements.
  - `ap-fqz9wvDAHSolheD4AsGh9Z` at 2026-05-21 00:04 CDT — refused to start: Modal detected `modal_app.py` was edited (linter) during build.
- **Cancellation timestamp**: 2026-05-21 00:05 CDT (manual stop of `ap-fqz9wvDAHSolheD4AsGh9Z`).
- **Total compute consumed**: approximately **2 minutes** of L4 wall-clock across all three attempts. No epoch completed in any attempt; the failures were all in setup / image build / first-batch load.
- **Promotion status**: no checkpoint promoted to production. The practice screen has continued to serve the deterministic stub fallback + ADR 0010 offline banner since the T1 production-honesty commit `06b3004` (2026-05-20). That state persists under ADR 0011 — the practice surface is untouched by the pivot until the new pipeline produces a shippable artifact.
- **Compute redirected to** Phase 1 hand-detector training under [ADR 0012](./0012-strict-from-scratch-cv-scoping.md).

---

## Consequences for the slice-1b corpus

The 11,153 cleaned per-clip MP4s (75 signs, ~150 clips/sign average)
that the v3 cleaning run produced on 2026-05-20 are **no longer
training data for a 3D-CNN sign classifier**. They are repurposed
for four roles under the new architecture:

1. **Frame pool for keypoint labeling.** Sampled frames serve as the
   labeling source for the hand detector + landmark detector training
   data. Bounding-box and keypoint annotations get authored against
   sampled frames, not whole clips.
2. **Source corpus for per-sign template distributions.** A curated
   subset of clips per sign drives the Stage-2 template distributions.
   Selection criteria: high MediaPipe-free landmark-quality (post the
   from-scratch detector running over the corpus), clean framing,
   one signer per template, no source-video duplicates.
3. **Source corpus for per-sign threshold calibration.** Held-out
   slices of the same corpus calibrate per-sign confidence thresholds
   to the ≥ 90 % precision target.
4. **Source corpus for the confusion-mined hint decision tree.** The
   template-matching confusion graph (built from the corpus held-out
   set) drives which sign pairs need authored hints and what
   parameter delta the hint should name.

The v3 manifest, vocabulary filter (`slice1b_vocabulary.json`), and
signer-disjoint splits all carry forward unchanged.

---

## Licensing note

The corpus remains a mix of three non-commercial licenses:
**Sem-Lex (CC BY-NC-SA 4.0)**, **ASL Citizen (MSR-LA, research use
only)**, **WLASL (mixed YouTube licenses, research use)**. Under
ADR 0011 we use the corpus for four *development* purposes only:
labeling input, template-distribution construction,
threshold calibration, and confusion-mined hint authoring. **None of
these produce a redistributable derivative dataset**, and **the
templates and thresholds that get bundled into a shipped ONNX
artifact are development / calibration artifacts**, not the corpus
itself.

Per the slice-2 cliff named in ADR 0008 + ADR 0009: **shipped
templates will be re-recorded with consenting signers in a later
phase**. The slice-1b corpus is for getting the pipeline to a
defensible accuracy in development; the shipped, commercially-
deployable templates require clean-license recordings authored
under the ADR 0004 instructor engagement.

---

## What stays the same

- **ADR 0002** (no placement test) — unchanged.
- **ADR 0003** (Vercel + Supabase + Cloudflare R2 deployment) —
  unchanged.
- **ADR 0004** (public sources only for slice 1; instructor
  engagement is slice-2 work) — unchanged; explicitly the path under
  which the shipped templates get re-recorded.
- **ADR 0005** (classical CV allowed) — unchanged; remains permitted
  for training-time augmentation of the labeling input.
- **ADR 0007** (Google + magic link + anonymous demo auth) —
  unchanged.
- **ADR 0008** (public-data-only training) — unchanged; explicit
  carry-forward.
- **ADR 0009** (ASL Citizen + Sem-Lex inclusion under non-commercial
  research scope) — unchanged; explicit carry-forward.
- **The mastery state machine + scheduler** (`lib/scheduler.ts`),
  **the modified-SM-2 algorithm**, **the three-layer hint
  architecture's scaffolding** (`confusion_pair_hints` table + the
  pre-attempt priming card UI), **the dashboard's forgetting-curve
  sparkline**, **the auth flow, RLS, sidebar progression**, and
  **the practice screen UI** (sidebar, reference video, camera
  preview, hint overlays, pass/fail states) all carry forward
  untouched. Only the **inference path** (camera tensor → classifier
  call) is rewired.

---

## What's superseded

- **ADR 0001** in its entirety as the active architecture. Preserved
  as the historical record of Path B's specification. Gains a
  "Superseded by ADR 0011 on 2026-05-21" header.
- **ADR 0010** in its entirety as the reversal-of-ADR-0006
  reinstatement. Preserved as the historical record of the
  short-lived (2026-05-20 → 2026-05-21) raw-RGB-from-scratch attempt.
  Gains a "Superseded by ADR 0011 on 2026-05-21" header.
- **The training pipeline at `training/classifier/cnn.py`,
  `training/classifier/dataset_video.py`, the T4-rewritten
  `train.py` / `validate.py` / `export.py` / `augment.py`**, and the
  T3-rewritten `training/data/clean.py`. All archived under
  `ml/archive/v3.0/code_v3_3dcnn_unbuilt/` and replaced by the new
  detector-training pipeline that lands incrementally under
  `training/detectors/`.

---

## Rejected alternatives

- **Keep iterating on the 3D CNN until top-1 hits 85 %.** Rejected.
  The accuracy ceiling under the available data is the wall; more
  hyperparameter sweeps, deeper architectures, or more data
  augmentation move it by 1–3 pp each, not 10–25 pp.
- **Adopt MediaPipe (or any pretrained landmark detector) and a
  small classifier on top.** Rejected. This was the ADR 0006 path
  whose permissive reading of Requirement 7 was withdrawn on
  2026-05-20 (ADR 0010). The strict reading governs; pretrained
  vision components are out.
- **Train our own landmark detector AND a learned classifier on the
  landmarks.** Considered. The classifier remains data-volume-bound
  (just on landmark sequences instead of pixels). The template
  approach is more transparent, has better hint substrate, and
  matches NVIDIA Signs' shipped pattern — adopted instead.
- **Skip the from-scratch detector and use classical CV (skin
  segmentation + optical flow + contour tracking) for landmarks.**
  Rejected — this was Path C in ADR 0001 and was rejected there as
  brittle across skin tone and lighting. ADR 0005 keeps classical CV
  permitted for augmentation; not for the inference path.

---

## How this is verified

- The `training/classifier/` files that backed ADR 0001 / ADR 0010
  (`cnn.py`, `dataset_video.py`, etc.) are deleted from the live
  tree and present only in `ml/archive/v3.0/code_v3_3dcnn_unbuilt/`.
- A new `training/detectors/` directory holds the from-scratch
  detector code as it lands incrementally per
  [ADR 0012](./0012-strict-from-scratch-cv-scoping.md)'s perimeter.
- `docs/MODEL.md` and `docs/ARCHITECTURE.md` get rewritten in a
  later commit (T-pivot equivalent of T2 documentation surgery) to
  reflect the new pipeline; this ADR is the entry point.
- The cancellation record above is auditable against Modal's app
  history (the three `ap-*` ids and timestamps).
