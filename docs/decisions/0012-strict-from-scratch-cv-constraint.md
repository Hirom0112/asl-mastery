# ADR 0012: Strict from-scratch CV constraint — scope, perimeter, declared exceptions

**Status:** Accepted
**Date:** 2026-05-20
**Relates to:** [ADR 0010](./0010-reversal-of-adr-0006.md) (strict reading of Requirement 7 governs), [ADR 0011](./0011-landmarks-and-templates-pivot.md) (recognition architecture pivot to landmarks + templates), [ADR 0005](./0005-classical-cv-allowed.md) (classical CV permitted), [ADR 0008](./0008-public-data-only-training.md) (public-data-only training)

---

## Context

ADR 0010 restored the strict reading of brief Requirement 7: no
pretrained vision components anywhere in the recognition pipeline.
ADR 0011 then pivoted the recognition architecture from end-to-end
RGB classification to a four-stage from-scratch landmark + template
pipeline (hand detector → hand landmark regressor → pose detector →
face detector → trajectory matcher / optional learned head). Both
ADRs assume a constraint that has never been written down in one
place at the level of detail needed to keep it from drifting over
the next 11–13 months of work.

This ADR is that document. It is the load-bearing constraint ADR.
On the bad days — when a from-scratch detector is not converging
and a pretrained hand-tracking package is one keystroke away — this is
the ADR to re-read.

The brief's Requirement 7 is the source authority. Its strict
reading prohibits pretrained models that touch pixels in the
recognition system. It does not prohibit non-CV libraries,
programming frameworks, data-processing libraries, or general
machine-learning libraries — the brief explicitly permits these.
The constraint is therefore **strict-but-scoped**: maximally strict
inside the CV perimeter, scoped to the CV perimeter.

---

## Decision

### The CV perimeter

The CV perimeter is every component, weight, or learned parameter
that participates in turning camera pixels into recognition output.
Inside this perimeter, **every weight is trained by this project
on documented data**. No pretrained models. No pretrained backbones.
No pretrained classifiers. No pretrained landmark detectors. No
pretrained feature extractors. No `load_state_dict` calls reading
weights this project did not produce. No external weight URLs.

Components inside the CV perimeter (all from-scratch, per ADR 0011):

1. **Hand detector** — single-stage anchor-free CNN, ~1–3M params, CenterNet-style head, trained on project-labeled hand bounding boxes.
2. **Hand landmark regressor** — small CNN backbone + coordinate regression head, 21 keypoints, trained on project-labeled hand keypoints.
3. **Pose detector** — small CNN, 8 upper-body keypoints, trained on project-labeled pose annotations.
4. **Face detector** — single-class tiny CNN, trained on project-labeled face bounding boxes.
5. **Sign matcher** — distribution-of-templates over landmark trajectories, derived statistically from project-extracted trajectories. Optional learned head (TCN or 1D conv stack), if added, is trained from scratch on project-extracted landmark sequences.

### Permitted dependencies outside the CV perimeter

These dependencies contain no pretrained neural weights and do not
contradict the constraint:

- **Programming frameworks:** React, Vite, Three.js (rendering only — no ML), Node.js, Next.js if/when reintroduced.
- **Data and storage:** SQLite (`better-sqlite3`), IndexedDB, ONNX Runtime Web (an inference runtime, not a model).
- **Machine-learning libraries (training infrastructure, no shipped weights):** PyTorch, NumPy, torchvision *transforms only* (image transforms — no `torchvision.models.*` imports), `onnx`, `onnxruntime`, Modal SDK.
- **Labeling tools:** CVAT or LabelStudio (used to produce *our* labels; any model-assisted pre-labeling uses a model *we trained* in the bootstrapping loop, never a third-party pretrained one).
- **Classical (non-learned) CV** per ADR 0005: MOG2 background subtraction, optical flow, skin segmentation in HSV space, perceptual hashing, contour detection. Hand-coded algorithms with no fitted parameters — categorically different from pretrained weights.
- **3D assets:** Ready Player Me / Mixamo avatar meshes and rigs. These are art assets, not ML models — no neural network generates the avatar's motion. Animations are procedural (IK retarget from our templates) or hand-keyed in Blender.

### Pretrained dependencies outside the CV perimeter (declared, not hidden)

The following dependency contains pretrained neural network weights
that were not trained by this project, and is used **outside the CV
/ recognition stack**:

- **Web Speech API (browser-native TTS).** Used to produce warm
  spoken hint audio in the multimodal feedback layer (Phase 6,
  Slice 6.3). The underlying TTS model is provided by the user's
  browser / operating system. It does not process camera input,
  does not contribute to sign recognition, and is invoked only
  *after* the recognition system has produced a pass/fail decision
  and the hint logic has selected hint text. Used post-recognition,
  for output only.

No other pretrained neural models are used anywhere in the project.
The recognition pipeline — hand detector, hand landmark regressor,
pose detector, face detector, sign matcher — is 100% from-scratch,
with every weight trained by this project on documented data.

If a fully-zero-pretrained-anything constraint is required by a
reviewer, the TTS can be removed and replaced with subtitle-only
hint display. The subtitle bubble (Slice 6.5) already exists and
already carries the full hint text; the TTS is additive on top of
it. Removing TTS would reduce the warmth of the feedback experience
but is technically a single-flag change. This stricter alternative
is technically feasible and explicitly offered.

### Anti-drift rules

These rules exist because the failure mode this ADR guards against
is incremental drift, not a single deliberate violation. The
hardest constraint to hold is not technical — it is the constraint
to not drift.

1. **No pretrained-CV package in `package.json`, and no pretrained-CV
   import in any shipped code path, ever.** The recognition code imports
   zero pretrained models. (The only pretrained-CV name anywhere in the
   tree is `@mediapipe/tasks-vision`, an unused transitive dependency
   that `@react-three/drei` declares for face-tracking helpers we never
   use — it is not in `package.json`, not installed, and never imported.)
   Any new pretrained-CV import is a violation to be reviewed under this ADR.
2. **No `torchvision.models` imports.** `torchvision.transforms` is
   permitted (data augmentation); `torchvision.models` is not (it
   exposes pretrained backbones).
3. **No `load_state_dict` call in any model file reads weights
   this project did not produce.** Checkpoints loaded inside a
   training resume are fine; weight URLs pointing at someone
   else's hosted weights are not.
4. **No "we'll swap it later" stubs in the recognition path.** The
   detector is built before the pedagogy layer is developed. Real
   landmarks must be flowing from the real model before any
   pedagogy code that consumes landmarks is written. This rule is
   restated from the operating rules in
   `docs/VOCABULARY_TRAINER_ROADMAP.md`.
5. **Every model card under `docs/model_cards/` declares
   "Pretrained components: none"** in a fixed-format field, alongside
   architecture, parameter count, training data provenance, training
   process, validation metrics, and known failure modes. A reviewer
   reading any single model card can confirm the constraint without
   reading any other document.
6. **Every new ADR that touches the recognition path references
   ADR 0012 by number** and confirms compliance, or explicitly
   marks itself as proposing a relaxation (which would then have
   to be reviewed against this ADR's reasoning).

### Audit surface

A reviewer auditing the constraint should be able to verify it from
a small fixed set of artifacts:

- `docs/decisions/0012-strict-from-scratch-cv-constraint.md` — this document.
- `docs/model_cards/*.md` — one card per model in the CV perimeter, each declaring "Pretrained components: none."
- `package.json` — no pretrained-model packages declared.
- `training/requirements.txt` — no pretrained-model packages.
- The recognition code (`lib/inference/`, `training/detectors/`) imports no pretrained CV — a search for pretrained-model imports returns nothing. (The lockfile lists one unused `@mediapipe/tasks-vision`, a transitive dependency of `@react-three/drei`'s face-tracking helpers, which is never imported or installed.)
- `grep -r 'torchvision.models' --include='*.py' .` — returns nothing.
- `grep -r 'load_state_dict' --include='*.py' .` — returns only project-internal checkpoint loads (training resume), each commented with the originating training run.
- Every model's weight artifact has a corresponding Modal training-run record.

---

## Rejected alternatives

- **Permit a pretrained landmark detector and document why.**
  Rejected. ADR 0006 took this path under the permissive reading of
  Requirement 7, and ADR 0010 reversed it. Re-litigating that
  decision here would be wasted motion.
- **Permit a pretrained backbone (ImageNet ResNet, Kinetics video
  model) for warm-start only, then fine-tune from scratch.**
  Rejected. The strict reading does not distinguish "warm start"
  from "use as-is" — pretrained weights touching pixels are out of
  scope regardless of whether they are subsequently updated. This
  also collapses the from-scratch story: a reviewer asked "did you
  train this from scratch?" would have to answer "the architecture
  yes, the weights partially," which is the answer this constraint
  exists to prevent.
- **Permit Web Speech API silently and not declare it.** Rejected.
  Hiding a dependency is the failure mode that this whole project
  exists to demonstrate against. The strictness of the constraint
  is the moat; the honesty of the declaration is what makes the
  strictness credible.

---

## Consequences

- The timeline reflects this constraint. The 11–13-month estimate
  in `docs/VOCABULARY_TRAINER_ROADMAP.md` exists because every CV
  weight is trained by us on data we labeled. Relaxing the
  constraint would shorten the timeline by many months and would
  also destroy the project's story.
- The first four months of work produce **training pipelines,
  labeled datasets, and model cards** — not a finished product. The
  Friday Question answer for Months 1–4 is "the training pipeline
  and the labeled dataset I am building, plus the model cards for
  what I have trained so far." This is the work. The product
  follows.
- If a CV component will not converge despite the budgeted
  iteration time, the response is **scope reduction** (75 → 50
  signs, simpler architecture, fewer keypoints, accept lower
  validation accuracy and document it) **not relaxation of the
  constraint**. Brief Requirement 8 explicitly demands documented
  limitations, not eliminated ones.
- A reviewer who reads only this ADR knows: what is from-scratch,
  what is not, where the perimeter is drawn, and how to verify
  compliance. That is the artifact this ADR exists to be.
