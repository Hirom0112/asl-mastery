# ADR 0011: Pivot to from-scratch landmarks + templates recognition pipeline

**Status:** Accepted
**Date:** 2026-05-20
**Supersedes:** [ADR 0001](./0001-recognition-architecture.md) (end-to-end small 3D CNN over raw RGB, "Path B"), [ADR 0010](./0010-reversal-of-adr-0006.md) (which reinstated ADR 0001 under the strict reading of Requirement 7)
**Relates to:** [ADR 0012](./0012-strict-from-scratch-cv-constraint.md) (the constraint perimeter this pivot honors), [ADR 0015](./0015-external-cv-datasets-provenance.md) (external labeled dataset acceptance criteria), [ADR 0005](./0005-classical-cv-allowed.md), [ADR 0008](./0008-public-data-only-training.md)

---

## Context

ADR 0010 reinstated ADR 0001 Path B — end-to-end small 3D CNN over
raw RGB — after the permissive reading of brief Requirement 7 was
withdrawn. Path B is **constraint-compliant** (every weight trained
from scratch on public-source raw video) but its **honest outcome
estimate** in ADR 0010 was 30–50% top-1 accuracy on 75 signs,
against an 85% eval-gate floor (`docs/EVAL_GATE.md`). The dominant
cause was per-sign clip count: Path B was sized for ~200 clips/sign
and the public-source corpus delivers 13–20 clips/sign across the
frozen 75-sign vocabulary (`docs/VOCABULARY.md` — "Frozen 75-sign
trainer vocabulary").

The question this ADR answers: given the strict no-pretrained-CV
constraint and a 13–20 clips/sign data ceiling, is end-to-end RGB
classification the right architecture, or is there a
constraint-compliant pipeline that uses the limited data more
sample-efficiently and produces a more pedagogically useful output?

The answer the project is committing to is a **four-stage
from-scratch landmark pipeline followed by a distribution-of-templates
matcher** (NVIDIA-style, but with every model trained by this
project). The pipeline:

1. **Hand detector** — from-scratch CenterNet-style single-stage detector, 320×320 input, ~1–3M params.
2. **Hand landmark regressor** — from-scratch small CNN over 224×224 hand crops, regresses 21 keypoints per hand.
3. **Pose detector** — from-scratch small CNN, 8 upper-body keypoints, provides the body-relative reference frame.
4. **Face detector** — from-scratch tiny single-class detector, for framing UI and onboarding calibration.
5. **Sign matcher** — distribution-of-templates over landmark trajectories per sign; Mahalanobis-distance softmax; per-sign precision-prioritizing thresholds. Optional learned head (TCN or 1D conv) over landmark sequences if templates underperform.

This pivot does **not** relax the from-scratch constraint. It
re-distributes the from-scratch budget across four small detectors
plus a statistical matcher, instead of one large end-to-end
classifier. The constraint perimeter is documented in
`docs/decisions/0012-strict-from-scratch-cv-constraint.md` (this
ADR's load-bearing companion).

---

## Decision

The recognition architecture pivots from ADR 0001 Path B
(end-to-end small 3D CNN over raw RGB) to a **four-stage
from-scratch landmark pipeline + distribution-of-templates matcher**.

1. **Four CV models, all from-scratch.** Architectures, parameter
   targets, training data, and validation metrics per
   `docs/VOCABULARY_TRAINER_ROADMAP.md` Phases 1–3. No pretrained
   weights. No pretrained backbones. No `@mediapipe/*`. Every weight
   has a Modal training run and a model card under
   `docs/model_cards/`.
2. **Sign matcher: distribution-of-templates over landmark
   trajectories.** Per sign, cluster landmark trajectories from that
   sign's training clips, compute mean trajectory + per-keypoint
   per-timestep variance, and store as the template distribution.
   At inference, compute Mahalanobis distance from the learner's
   trajectory to each template; softmax over distances → per-sign
   confidence.
3. **Per-sign thresholds, precision-prioritized.** Each sign's
   pass threshold is calibrated against positive (correct clips
   matching their own template) and negative (other-sign clips
   matching this template) distributions. Precision is prioritized
   over recall — false-pass is worse than false-fail in a learning
   context. Per-sign thresholds and the data behind them are the
   artifact brief Requirement 9 demands.
4. **Optional learned head.** If templates alone underperform, a
   small temporal model (TCN or 1D conv stack) is trained from
   scratch on the project-extracted landmark sequences with their
   class labels. Evaluated head-to-head with templates-only; ship
   whichever is better; document the choice in the recognition
   model card.
5. **Inference pipeline runs entirely in the browser via ONNX
   Runtime Web.** Hand detector → for each detected hand, crop → hand
   landmark regressor → 21 keypoints; in parallel, pose detector and
   face detector. Landmark trajectory accumulated over a 1–3 second
   window, then fed to the sign matcher.
6. **All artifacts under ADR 0010's Path B (v1.0.1, v2.0.0,
   v2.1.0, and any v3.x produced under ADR 0010) remain
   deactivated.** They are not architecturally compatible with this
   pipeline. They stay in the repository as historical records per
   ADR 0010's "no silent revisions" commitment.

---

## Why this pivot now

The data ceiling. ADR 0010 named 30–50% top-1 as the honest
outcome for Path B on 13–20 clips/sign. A landmark pipeline is
substantially more sample-efficient:

- Each detector is trained on **labeled frames**, not labeled
  clips. One labeled frame supports detector training for any
  sign; the 12K-clip corpus yields 100K+ frame samples after
  1-FPS extraction (`docs/VOCABULARY_TRAINER_ROADMAP.md` Phase 1,
  Slice 1.1). The detector accuracy ceiling is set by labeling
  budget, not by per-sign clip count.
- The landmark regressor is trained on **labeled hand crops** —
  also independent of sign identity. Bootstrapping (label-train-pre-label-correct,
  iterated) brings the realistic 120-hour labeling budget down by
  ~40% per generation per `docs/VOCABULARY_TRAINER_ROADMAP.md`
  Phase 2, Slice 2.1.
- The template matcher operates on **landmark trajectories**, a
  representation that already absorbs visual variance (skin tone,
  lighting, background, camera position). The per-sign 13–20
  clips/sign is now used to fit a low-dimensional trajectory
  distribution, not to teach a CNN every pixel-level intra-class
  variation.

The pivot also unlocks **interpretable pass/fail decomposition**.
At inference, similarity is computed not against a black-box CNN
logit but against a per-keypoint, per-timestep trajectory
distribution that can be decomposed along the four Stokoe ASL
parameters (handshape, location, movement, palm orientation —
`docs/VOCABULARY_TRAINER_ROADMAP.md` Phase 5, Slice 5.2). The
"worst parameter" of a failed attempt becomes the hint target.
This decomposition was acknowledged as harder under raw-RGB Path B
in ADR 0010's "Consequences → Hint system" section. It becomes
natural under this pipeline.

---

## Constraint compliance

This pivot is governed by ADR 0012. Every CV component listed in
the Decision section is inside the CV perimeter and trained from
scratch. The complete list of permitted dependencies, the declared
TTS exception (Web Speech API, post-recognition output only), and
the anti-drift rules (no `@mediapipe/*`, no `torchvision.models`,
no foreign `load_state_dict`, no "we'll swap it later" stubs in
the recognition path) live in ADR 0012 and govern every commit
that touches this pipeline.

The pivot does not introduce a single new pretrained dependency.
What it introduces is four small models the project must label
data for and train, in place of one larger model that consumed
raw video.

---

## Rejected alternatives

- **Stay on ADR 0001 Path B, accept 30–50% accuracy, ship with a
  documented limitation.** Rejected. ADR 0010 left this open as
  the honest outcome. Brief Requirement 8 permits documented
  limitations, but Requirement 9 demands per-sign calibrated
  thresholds — and per-sign thresholds derived from a 30–50%
  accuracy classifier are not pedagogically useful. The learner
  experience falls below the "useful pilot" bar in brief Section 4.
- **Pretrained landmark detector (a pretrained landmark detector) plus
  from-scratch matcher on top.** Rejected. This is exactly ADR 0006
  under a different name. ADR 0010 reversed ADR 0006 under the
  strict reading of Requirement 7. Re-litigating that decision via
  the back door is bad-faith engagement with the brief.
- **Pretrained backbone (ImageNet or Kinetics) warm-start for the
  end-to-end classifier, then fine-tune.** Rejected by ADR 0012:
  pretrained weights touching pixels are out of scope under the
  strict reading regardless of whether they are subsequently
  updated. The from-scratch story is the moat.
- **Hybrid pipeline: classical CV preprocessing (skin segmentation,
  optical flow) → small from-scratch classifier over the features.**
  Rejected in ADR 0001 as Path C and rejected again here. Classical
  CV thresholds historically fail across skin tone and lighting;
  building a fairness-respecting system on that substrate is harder
  than building one on landmark trajectories.
- **Train a single multi-task model that jointly predicts hand
  bbox + hand keypoints + pose + face from one forward pass, instead
  of four separate models.** Not rejected — deferred to the
  optional stretch in `docs/VOCABULARY_TRAINER_ROADMAP.md` Phase 3,
  Slice 3.3. If it beats the separate models on speed and accuracy,
  it ships. If not, the four-model pipeline is the production
  choice and the multi-task experiment is documented as
  investigated-and-not-pursued.

---

## Consequences

### Timeline and scope

- Total realistic build budget moves to **11–13 months** per
  `docs/VOCABULARY_TRAINER_ROADMAP.md`. First real detector output
  at end of Month 4; all four CV components working at end of
  Month 7; 75-sign vocabulary end-to-end with pedagogy at Month 10;
  polish (avatar, TTS, mastery, validation) Months 12–13. This is
  substantially longer than the original Path B timeline and is
  driven by the labeling budget (especially Phase 2, ~120 hours of
  hand-keypoint labeling).
- If iteration time runs out on any single CV component, the
  documented escape is **scope reduction** (75 → 50 signs, smaller
  architectures, fewer keypoints, accept lower validation accuracy
  and document) per ADR 0012's anti-drift rule 1 and brief
  Requirement 8. Adding a pretrained component to escape is the
  failure mode this whole project exists to demonstrate against
  and is forbidden.

### Data pipeline (`docs/DATASET.md`, `docs/ROADMAP.md` Phase 3)

- The raw 12K-clip public-source corpus stays. It now feeds two
  consumers: frame sampling (1 FPS) for detector / landmark
  training, and per-clip trajectory extraction for template
  fitting.
- A new **labeled-frames** artifact class enters the pipeline:
  hand bboxes, hand keypoints, pose keypoints, face bboxes. These
  are produced under `/labeling` and stored under `/data/labeled_frames`
  with per-batch provenance per `/data/README.md`.
- Per-clip landmark trajectories (`/data/templates` inputs) are
  generated by running the trained hand + pose detectors across
  the full 12K-clip corpus once at the start of Phase 4. Storage:
  tens of MB; trivial compared to the video.
- The cleaning pipeline (`training/data/clean.py` /
  `dataset/clean/`) stays as-is for raw clip preparation. It no
  longer feeds an RGB classifier directly; it feeds the labeled-
  frame sampling step and the trajectory-extraction step.

### Training (`docs/MODEL.md`, `training/`)

- Five training entrypoints replace the single end-to-end CNN
  training entrypoint: `train_hand_detector`,
  `train_hand_landmarks`, `train_pose`, `train_face`, and
  optionally `train_sign_head`. All on Modal. All from scratch.
  Each gets its own model card under `docs/model_cards/`.
- The `training/classifier/cnn.py` referenced in ADR 0010 (the
  R(2+1)D end-to-end CNN) is **not** built. It becomes a
  not-pursued artifact named in this ADR and the recognition model
  card.
- Per-GPU iteration cost drops sharply per training run (four
  small models on small inputs vs. one bigger model on video
  tensors) but rises in aggregate because there are five training
  loops and a labeling pipeline to maintain. Net effect: more
  human time, less GPU time.

### Inference (`docs/ARCHITECTURE.md` §2.3)

- The per-attempt browser pipeline becomes: webcam frame → hand
  detector (320×320) → per-hand crop → hand landmark regressor
  (224×224) → 21 keypoints per hand. In parallel: pose detector,
  face detector. Landmark stream accumulates over a 1–3 second
  window and is fed to the template matcher.
- End-to-end browser latency target: ≥20 FPS on a dev laptop with
  all four detectors running (Phase 2 Slice 2.4 benchmark gate).
  Per-attempt classification latency: dominated by the 1–3 second
  capture window, not by inference.
- Browser bundle target: each ONNX model under ~5 MB after INT8
  quantization; total under ~20 MB for the four detectors plus
  template manifest. Looser than ADR 0006's 5 MB budget but
  acceptable for a localhost-only app (ADR 0012's "local only,
  indefinitely" rule).

### Pedagogy (`docs/ARCHITECTURE.md` §5, `docs/PEDAGOGY.md`)

- The three-layer hint system survives. The 120 already-seeded
  confusion-pair hints (architecture-agnostic) survive.
- Layer-3 parameter-aware hints become **natural** under this
  pipeline rather than a research problem. The four Stokoe ASL
  parameters (handshape, location, movement, palm orientation) map
  directly to features of the landmark trajectory and the per-
  keypoint variance structure. The "worst parameter" of a failed
  attempt is the hint target; the hint copy comes from the
  template library in Phase 5 Slice 5.3.

### Validation and fairness (`docs/EVAL_GATE.md`)

- Hard criterion 10 (MediaPipe landmark detection success ≥ 95%)
  stays removed per ADR 0010 — MediaPipe is not in the pipeline.
  A replacement criterion 10 is introduced: **own-hand-detector
  recall ≥ 90% on the held-out validation set, broken out by
  Fitzpatrick demographic.** The criterion measures the
  project-trained detector instead of the third-party one. Hard
  criteria 1–9 are unchanged.
- The per-Fitzpatrick ≤ 10 pp accuracy gap (criterion 3) becomes
  somewhat easier than under Path B raw-RGB, because the
  trajectory representation has already absorbed pixel-level
  variance. It is still a top-line eval-gate criterion; the
  detector and landmark regressor are the surfaces where skin-tone
  bias can enter and are the surfaces the fairness audit measures.

### Privacy (`docs/PRIVACY.md`)

- Stronger than under ADR 0010. There is no third-party vendor
  (no third-party model CDN) in the inference path. There is no remote
  inference. Per ADR 0012, the entire app runs on localhost
  indefinitely until that constraint is explicitly reversed.

### Data sourcing — landmark training (amended 2026-05-21 under ADR 0015)

The original Decision section above assumes that **all** labeled data for the
four detectors is produced by the project (labeled frames sampled from our
12K-clip corpus, plus self-recorded supplements). The 11–13-month timeline in
`docs/VOCABULARY_TRAINER_ROADMAP.md` is sized accordingly, with the dominant
single line item being ~120 hours of from-scratch hand-keypoint labeling in
Phase 2 Slice 2.1.

The provenance audit at `docs/data/external_datasets_audit.md` (2026-05-21)
identifies multiple **public labeled CV datasets** whose annotations were
produced by humans, by physical sensors, or by the dataset authors' own
multi-view fitting pipelines — i.e., not by any third-party pretrained CV
model. ADR 0015 codifies the acceptance criteria for these datasets and the
exclusion of pretrained-model-labeled / OpenPose-labeled / pretrained-model-labeled
sources (substantive equivalence + audit credibility, per ADR 0015 §
"Why pretrained-model-labeled datasets are excluded").

Folding the RECOMMENDED external datasets into the training pipeline does
**not** relax ADR 0012's from-scratch CV constraint. Every weight in every
detector is still trained from scratch — the change is only in the *labels*
the loss is computed against, and those labels remain provenance-clean per
ADR 0015's six criteria.

**Sources used per detector (subject to user approval at download time):**

- **Hand detector (Phase 1):** bounding boxes derived from the hand-keypoint
  datasets accepted under ADR 0015 (FreiHAND, CMU HandDB manual + multiview,
  Multiview Hand Pose, InterHand2.6M human_annot, COCO-WholeBody), plus
  ~500–1,000 self-recorded frames for camera-setup and Fitzpatrick coverage
  gaps. Total available: ~150K–600K labeled hand instances (audit memo §
  "Combined-corpus summary"), well above the original Slice 1.1 target.
- **Hand landmark regressor (Phase 2):** same source list as the hand
  detector. Direct 21-keypoint annotations available across all listed
  sources after topology normalization (Phase 2 Slice 2.1's bootstrapping
  iteration is now an optional refinement, not the primary annotation
  mechanism).
- **Pose detector (Phase 3 Slice 3.1):** upper-body keypoints from MPII
  Human Pose (~40K people) and COCO Keypoints (~250K person instances).
  Total available: ~290K person instances with shoulders / elbows / wrists.
  Supplemental self-recording reduced to ~100–200 frames for the user's
  webcam framing gap.
- **Face detector (Phase 3 Slice 3.2):** WIDER FACE (~393K face boxes
  across 32K images). Supplemental self-recording reduced to ~50–100 frames
  for the onboarding framing-check UX in the user's camera setup.
- **Sign matcher (Phase 4):** unchanged — still built from the project's
  12K-clip ASL corpus (WLASL + Lifeprint + targeted YouTube + ASL Citizen
  under ADR 0009). The matcher consumes landmark trajectories produced by
  the detectors above; external hand-keypoint datasets do not contain ASL
  signing trajectories and are not used here.

**Revised total supplemental labeling workload:** ~450–900 self-recorded
frames across all four detectors, ~5–15 hours total at the documented
per-frame labeling rate. Down from the original ~120-hour Phase 2 cliff.

**Timeline implication:** Phase 2 in `docs/VOCABULARY_TRAINER_ROADMAP.md`
compresses substantially. The 11–13-month total roadmap budget is **not**
re-baselined in this amendment because the savings should flow into deeper
iteration on the harder phases (Phase 4 template matcher, Phase 5
parameter-aware hints, Phase 6 avatar / TTS / correction animation) rather
than into shipping earlier. The roadmap text remains the authoritative
schedule until a separate ADR re-baselines it.

**Audit trail:** Every external dataset used will have an entry in
`docs/data/external_datasets_audit.md`, a corresponding download script
under `scripts/datasets/`, and a `data/external/<dataset_name>/PROVENANCE.md`
recording the source URL, license, SHA256 of the downloaded archives, and
the ADR 0015 audit-entry name. Every model card under `docs/model_cards/`
for a detector trained on external data references its `PROVENANCE.md` entries
by path. ADR 0012's "Pretrained components: none" declaration on each model
card stands unchanged.

---

### Pre-pivot infrastructure that stays dormant

Per the active project direction (localhost-first, deploy-when-ready):

- The Supabase project (`ehrqwtvrmejozwlybndl`) remains linked but
  is not written to during local development of the new pipeline.
- The Cloudflare R2 buckets (models + references public, raw
  training data private) remain in place.
- The Vercel project remains linked, not redeployed. The
  production URL continues to serve the deactivated v2.1.0
  artifact behind ADR 0010's stub fallback + offline banner until
  a v3 model under this ADR is shipped and approved.

No infrastructure is unlinked or deleted. Reactivation is a single
deploy when the new pipeline clears the eval gate.

---

## How this is verified

- `grep -r 'mediapipe' --include='*.ts' --include='*.tsx' --include='*.py' .` returns zero hits.
- `grep -r '@mediapipe' package.json training/requirements.txt` returns zero hits.
- `grep -r 'torchvision.models' --include='*.py' .` returns zero hits.
- Four model cards exist under `docs/model_cards/`: `hand_detector_v1.md`, `hand_landmarks_v1.md`, `pose_v1.md`, `face_v1.md`. Each declares "Pretrained components: none." Each cites a Modal training run.
- A fifth model card `sign_matcher_v1.md` documents the template fitting procedure, the per-sign threshold table (brief Requirement 9 artifact), and the optional learned-head experiment.
- The inference path in `/web` (the Next.js app at the repo root) wires the four ONNX models through ONNX Runtime Web with no third-party model fetches.
- `docs/validation/v3.md` (when produced) names this ADR by number and reports per-sign accuracy, per-Fitzpatrick accuracy, false-pass / false-fail rates, and the per-sign threshold calibration data.
- The frozen 75-sign vocabulary at `dataset/slice1b_vocabulary.json` is the vocabulary the templates, thresholds, and hints are built against (`docs/VOCABULARY.md` — "Frozen 75-sign trainer vocabulary").
