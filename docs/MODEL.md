# Model

> ⚠️ **SUPERSEDED — read [`STATUS.md`](../STATUS.md) first.**
> This document describes a single-classifier architecture (either the
> ADR 0006 BiLSTM-on-MediaPipe-keypoints, or the ADR 0010-reinstated
> end-to-end R(2+1)D 3D CNN on raw RGB). Both are superseded by
> [ADR 0011](decisions/0011-landmarks-and-templates-pivot.md), which
> replaces the single classifier with a **four-stage from-scratch
> landmark + templates pipeline** (hand detector → hand landmark
> regressor → pose regressor → face detector → trajectory template
> matcher). Architecture sources for the new pipeline live in
> `training/detectors/`. A v2 model card will land once first training
> runs produce results.

> Model architecture, training procedure, and evidence that no
> pretrained vision components — landmark detectors, classifiers,
> backbones — appear anywhere in the pipeline.
>
> Governing decisions: [ADR 0001](./decisions/0001-recognition-architecture.md)
> (Path B, end-to-end small 3D CNN trained from scratch on raw RGB)
> reinstated by [ADR 0010](./decisions/0010-reversal-of-adr-0006.md)
> on 2026-05-20 after the earlier permissive reading of brief
> Requirement 7 (ADR 0006) was withdrawn. Classical CV permitted
> for training-time augmentation only, per
> [ADR 0005](./decisions/0005-classical-cv-allowed.md).

---

## 1. Architecture

**Single-stage end-to-end small 3D CNN, trained from scratch on raw
RGB video.** No landmark extractor, no pretrained backbone, no
pretrained anything.

- **Topology:** R(2+1)D-style factored 3D convolution. Each 3D conv
  block is split into a `1 × 3 × 3` spatial conv composed with a
  `3 × 1 × 1` temporal conv. The factorization gives nearly the
  representational capacity of full 3D conv at materially lower
  parameter count and faster wall-clock per epoch on commodity GPUs
  (Tran et al. 2018, "A Closer Look at Spatiotemporal
  Convolutions"). The reference architecture stays small by design
  — ASL recognition is not Kinetics, and overfitting risk dominates
  under-fitting risk at our data scale.
- **Input tensor:** `(B, T, H, W, 3)` float32 with `T = 16` frames.
  H and W are decided during T4 training based on browser inference
  latency; initial target H = W = 96 (small enough to hit the
  ≤ 300 ms inference budget on a Chromebook-class device) with
  H = W = 112 held in reserve as the larger variant.
- **Output:** logits over the 75-class slice-1 vocabulary
  (`docs/VOCABULARY.md`). Temperature-scaled at calibration time
  (see §4). Per-sign confidence thresholds bundled into the
  classifier config.
- **Parameters:** target ~5–10 M total. The exact channel widths /
  block counts are tuned in T4 to land in that range while clearing
  the inference latency budget. Reference channel ladder:
  `[64, 128, 256, 512]` with two R(2+1)D blocks per stage, global
  spatiotemporal average pool, then a linear classifier head.
- **Initialization:** Kaiming-normal (`torch.nn.init.kaiming_normal_`)
  on all conv and linear weights; biases zero; norms default
  (PyTorch). The init is implemented inline in the model module
  (`training/classifier/cnn.py`, built in T4); the old
  `training/classifier/init.py` file from the BiLSTM era is deleted
  in T3 commit 2.

The whole inference path is one model. No classical CV preprocessing
step in front of the classifier (per [ADR 0005](./decisions/0005-classical-cv-allowed.md)'s
"training-time only" boundary). No landmark extractor. The bytes
that go in are pixels; the bytes that come out are logits.

---

## 2. Training procedure

- **Optimizer:** AdamW, learning rate `1e-3`, weight decay `1e-4`.
  Initial LR is held under review during T5 — Path B CNNs can be
  finicky at the small data scale we ship under (see §8).
- **Scheduler:** cosine annealing across the full run.
- **Batch size:** decided in T4 against GPU memory; reference
  starting point is 32 on an L4 (24 GB) at H = W = 96 and T = 16.
- **Epochs:** 60 max with early-stopping patience 8 epochs on
  validation top-1.
- **Loss:** cross-entropy with label smoothing 0.1.
- **Class balance:** weighted sampling so each class appears
  roughly equally per epoch.
- **Mixed precision:** enabled (`torch.cuda.amp`) for memory
  headroom; explicitly disabled for the export-time forward pass
  to keep ONNX parity tight.
- **Reproducibility:** every Python random source seeded; W&B run
  records git commit, dataset version hash, full hyperparameter
  config. Dataset manifests record source-clip identifiers per
  example so a training run is reconstructible from artifacts.

Hardware: Modal L4 GPU per session 11 plan; A10G as the fallback
if L4 cycles run long. Expected training time per run: **hours, not
minutes** — the classifier is materially larger than the BiLSTM
that shipped under ADR 0006, and the inputs are raw video tensors
rather than precomputed keypoint sequences. Iteration cadence is
correspondingly slower; the T4 plan optimizes for fewer-better
runs rather than many speculative ones.

---

## 3. Augmentation stack

Training-time only, applied at the **pixel level** under
[ADR 0005](./decisions/0005-classical-cv-allowed.md)'s
classical-CV-permitted scope. None of these augmentations introduce
pretrained components; all are hand-coded procedures over pixel
arrays.

- **Random spatial crop.** Crop a `(H, W)` region from the
  256 × 256 cleaned source frame, with small jitter in offset.
  Models slight framing inconsistency between learners.
- **Color jitter, brightness, contrast.** Standard pixel-level
  augmentation. Models the dorm-room lighting variance that the
  classifier would otherwise overfit to in the WLASL +
  ASL Citizen + Sem-Lex source distribution.
- **MOG2 background swap (classical CV, ADR 0005).** Mixture-of-
  Gaussians background subtraction on a moving-average frame
  generates a foreground mask per clip; the masked foreground is
  composited onto random replacement backgrounds drawn from a
  small bank of plain / textured / cluttered references. Defends
  against background-bias overfitting — the canonical Path B
  failure mode for dorm-room training data.
- **Small affine.** ≤ ±10 ° in-plane rotation, ≤ ±5 % scale,
  ≤ ±3 % translation. Models camera tilt and minor framing
  drift.
- **Conditional horizontal flip.** Applied only on signs marked
  `flippable: true` in `docs/VOCABULARY.md`. Two-handed asymmetric
  signs and signs whose handedness encodes meaning are never
  flipped.

The keypoint-level augmentations from the superseded ADR 0006
landmark architecture (per-keypoint coordinate jitter, temporal
stretch on keypoint sequences, keypoint dropout) are deprecated —
they had no meaning on pixel inputs and the modules that
implemented them are deleted in T3 commit 2.

---

## 4. Calibration

- **Temperature scaling** on the validation set after training. A
  single scalar `T` minimizes negative log-likelihood when logits
  are divided by `T` before softmax (Guo et al. 2017).
- **Per-sign confidence threshold** derived from validation
  precision-recall curves. Target: ≥ 90 % precision per sign on
  the "pass" decision. Calibration JSON is bundled into the
  classifier config served from R2 alongside the ONNX artifact.
- **Retry leniency.** On second and third attempts at the same
  prompt within a session, the threshold drops slightly (decay
  schedule lives in app config). Prevents demoralizing repeat
  failures while keeping the first-attempt bar honest.

---

## 5. Confusion analysis

After training, the validation confusion matrix is dumped as
`confusion_matrix.json`. The top 2 – 3 confusion targets per sign
drive the confusion-pair hint system (`confusion_pair_hints`
table). The 120 confusion-pair hints already seeded against the
75-sign vocabulary (authored from ASL-LEX 2.0 phonological
features) are architecture-agnostic; they remain valid under the
reverted ADR 0001 Path B model. Pairs that v3.0 starts surfacing
but the v2.x-derived hint catalog doesn't cover are flagged for
authoring in a future session.

---

## 6. Export and quantization

- **PyTorch → ONNX** via `torch.onnx.export` with dynamic axes for
  the batch dimension and the temporal axis.
- **Parity check:** verify ONNX output matches PyTorch float32
  output within tolerance (max absolute error < 1e-4 on a
  100-clip held-out video tensor sample).
- **INT8 quantization** is **required** under Path B (unlike the
  ≤ 1 MB BiLSTM that shipped under ADR 0006, the 3D CNN is large
  enough that float32 would blow the bundle target). Calibration
  set is 256 representative clips; per-tensor symmetric quantization
  on weights, per-channel on activations where ONNX Runtime Web
  supports it. The eval-gate check refuses promotion if the
  quantized model regresses > 1 pp top-1 vs the float32 sibling.
- **Classifier artifact target: ≤ 10 MB after INT8 quantization.**

---

## 7. No-pretrained-pipeline evidence

Under the strict reading of brief Requirement 7 (governing again
since 2026-05-20 per [ADR 0010](./decisions/0010-reversal-of-adr-0006.md)),
**no pretrained vision components appear anywhere in the
pipeline**. The audit surface a reviewer can verify:

- **The classifier is trained entirely from scratch.** Weights
  initialize from `torch.nn.init.kaiming_normal_` for all conv and
  linear layers; biases zero; norms default. No `load_state_dict`
  call in the training entry point. No external classifier weight
  URL anywhere in `training/`.
- **No landmark detector.** The pipeline does not import MediaPipe,
  OpenPose, BlazePose, MMPose, or any equivalent library — in the
  training pipeline or in the frontend. T3 commit 1 deletes
  `lib/mediapipe/*`, `lib/keypoints.ts`, and
  `hooks/use-landmark-extractor.ts`, and removes
  `@mediapipe/tasks-vision` from `package.json`. T3 commit 2
  deletes `training/keypoints.py`, the MediaPipe extraction stage
  in `training/data/clean.py`, and the `mediapipe==0.10.18` pin
  from `training/requirements.txt`.
- **No pretrained image / video backbone.** Neither at training
  time (no `torchvision.models.video.*` weight downloads, no
  `transformers` import for video models) nor at inference time
  (the ONNX artifact is the from-scratch classifier, not a
  fine-tuned head on a pretrained backbone).
- **Classical CV is permitted under ADR 0005** for training-time
  augmentation (MOG2 background swap, color jitter, etc.) and for
  pipeline quality checks (perceptual hashing for dedup). These
  are hand-coded algorithms with no fitted parameters from prior
  training data; they are categorically different from "load these
  weights somebody else trained" and live unambiguously inside
  Requirement 6's permitted scope.
- **Public ASL datasets as raw video are permitted** per ADR 0008
  (WLASL, MS-ASL) and ADR 0009 (ASL Citizen, MSR-LA research use
  for slice 1). Raw video is data; it is not a pretrained
  component.

A reviewer auditing the no-pretrained claim should inspect
`training/classifier/cnn.py` and `training/data/clean.py` (after
T3 + T4 land), and confirm the absence of MediaPipe / pretrained-
weight imports in both. The CI workflow `.github/workflows/eval-gate.yml`
fires the audit on any PR touching `docs/validation/`.

---

## 8. Performance targets

Under Path B (reinstated by ADR 0010) the deployed bundle is larger
and inference is slower than the ADR 0006 landmark-based targets.
These numbers replace the §8 targets that served the superseded
landmark architecture:

- **First-visit client bundle download** (INT8-quantized 3D CNN
  classifier; no MediaPipe runtime to ship): ≤ 5 seconds over
  reasonable broadband. Combined size target ≤ 10 MB.
- **Cached-visit warm-up:** ≤ 1 second.
- **Per-clip classifier inference** (video tensor in, logits out,
  WebGPU or WASM): ≤ 300 ms.
- **End-to-end "submit attempt" to result UI:** ≤ 1 second.

If targets are missed, the response is to shrink the classifier
(reduce channel widths, drop a stage, lower H × W) — not to drop
quality elsewhere.

**Expected v3.0 accuracy ceiling.** ADR 0001 sized Path B for ~200
clips/sign. Across the merged WLASL + ASL Citizen + Sem-Lex source
corpus filtered to the 75-sign slice-1 vocabulary we have ~30 – 90
clips/sign (varies by sign). The honest estimate for v3.0 is **30 –
50 % top-1 accuracy**, well below the 85 % eval-gate floor. The
slice-1 acceptance pattern from v1.0.1 / v2.0.0 / v2.1.0 continues:
name the gap, do not paper over it. If iteration in T5 genuinely
stalls below 50 % top-1 on more than a handful of signs, the
escalation path is a scope-relief ask (smaller vocabulary, lower
floor, or commitment to the ADR 0004 instructor engagement) rather
than more iteration cycles.

---

## 9. Versioning

Every trained model has:

- A semantic-ish version string: `v{major}.{minor}.{patch}`.
- A content hash of the ONNX artifact (sha256, recorded in the
  artifact's `manifest.json`).
- A bundled config (per-sign thresholds, class list, normalization
  params, the `H × W` the model expects).
- A bundled validation report (`docs/validation/v<N>.md` plus the
  JSON sibling consumed by the eval-gate enforcer).
- A row in the `model_versions` Postgres table with `is_active`
  flag. Exactly one row at a time can be active (unique partial
  index in `supabase/migrations/20260519100000_init.sql`).

Promotion to active is manual for the pilot; the eval gate
(`docs/EVAL_GATE.md`) is the human's checklist. Slice-2 candidate:
automated promotion gated by the same checklist in CI.

**Current state (2026-05-20):** the rows for v1.0.1, v2.0.0, and
v2.1.0 are all `is_active = false` per the migration applied in
T1 (`06b3004`). No model is active; the practice screen serves a
deterministic stub and an honest offline banner naming ADR 0010
until v3.0 ships under the reverted architecture.
