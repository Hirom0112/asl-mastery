# ADR 0010: Reversal of ADR 0006 — strict reading of Requirement 7 governs again

**Status:** Accepted
**Date:** 2026-05-20
**Supersedes:** [ADR 0006](./0006-recognition-architecture-revised.md) (landmark-based recognition + pretrained landmark extractor permitted)
**Reinstates:** [ADR 0001](./0001-recognition-architecture.md) (end-to-end small 3D CNN trained from scratch on raw RGB)

---

## Context

ADR 0006 was written on 2026-05-19 against an earlier permissive
reading of brief Requirement 7. That reading held that Requirement 7
restricted pretrained ASL pipelines and pretrained sign classifiers
but *permitted* pretrained general-purpose landmark detectors
(MediaPipe Hands, MediaPipe Holistic, OpenPose, BlazePose). On the
strength of that reading, the architecture pivoted from ADR 0001's
end-to-end small 3D CNN over raw RGB (Path B) to a two-stage
landmark-based pipeline: MediaPipe Holistic for keypoint extraction
in the browser, then a from-scratch BiLSTM classifier over keypoint
sequences. Three artifacts shipped on this architecture — v1.0.1,
v2.0.0, v2.1.0.

On **2026-05-20** the earlier permissive reading was withdrawn. The
**strict reading of Requirement 7 governs again**:

- No pretrained vision components anywhere in the pipeline. Pretrained
  landmark detectors are pretrained vision models and are therefore
  out of scope, regardless of whether they are ASL-specific.
- Pretrained ASL classifiers and ASL feature extractors remain out of
  scope (this was uncontested under ADR 0006 and remains so).
- Classical (non-learned) CV remains permitted per ADR 0005: MOG2
  background subtraction, optical flow, skin segmentation in HSV
  space, perceptual hashing, contour detection. These are hand-coded
  algorithms with no fitted parameters; they are categorically
  different from "load these weights somebody else trained."
- Public ASL datasets as raw video remain permitted: WLASL, ASL
  Citizen (under MSR-LA for slice-1 research use per ADR 0009), and
  Sem-Lex (under CC BY-NC-SA 4.0).

---

## Decision

The architecture reverts to **ADR 0001 Path B**: end-to-end small 3D
CNN trained from scratch on raw RGB frames.

1. **MediaPipe Holistic is removed from the pipeline.** No
   browser-side landmark extraction. No training-time landmark
   extraction. No `@mediapipe/tasks-vision` dep. No
   `mediapipe==0.10.18` pin.
2. **The classifier takes raw RGB video tensors directly.** Input
   shape `(B, T=16, H, W, 3)` where H and W are decided during T4
   training based on browser inference latency (initial target: 96
   or 112).
3. **Architecture:** R(2+1)D-style small 3D CNN (factored 3D
   convolution: 1×3×3 spatial composed with 3×1×1 temporal). Target
   ~5–10M parameters. Kaiming-normal initialization throughout. No
   `load_state_dict` call. No external classifier weight URL.
4. **Augmentation moves back to pixel space** under ADR 0005:
   random spatial crop, color jitter, brightness/contrast,
   MOG2-based background swap (using the empty-frame clip captured
   at session start in the slice-2 recording tool; in slice-1 we
   apply random backgrounds drawn from a small bank), small affine
   transforms, conditional horizontal flip on `flippable: true`
   signs.
5. **ONNX export with INT8 quantization** to keep the deployed
   bundle under the original Path B target (≤ 10 MB total). The
   browser inference runtime stays as ONNX Runtime Web with WebGPU
   → WASM execution providers.
6. **Three existing artifacts (v1.0.1, v2.0.0, v2.1.0) are
   deactivated.** Migration
   `supabase/migrations/20260520200000_deactivate_all_models.sql`
   set `is_active = false` on every row of `model_versions`
   (applied in T1, `06b3004`). The artifacts and rows remain in R2
   and in Postgres as historical records — no silent revisions.
7. **The practice screen serves a deterministic stub fallback +
   honest offline banner** until v3.0 is rebuilt under the new
   constraint. The offline-banner copy explicitly names this ADR.

---

## What stays the same

The system-level architecture, every operational commitment, and
every architecture-agnostic surface is unchanged:

- The five logical surfaces in `docs/ARCHITECTURE.md` §1.
- The mastery state machine and scheduler (`lib/scheduler.ts`,
  `docs/ARCHITECTURE.md` §4) — pure functions, architecture-
  agnostic.
- The three-layer hint system (`docs/ARCHITECTURE.md` §5). The 120
  confusion-pair hints already authored from ASL-LEX 2.0
  phonological features (`supabase/migrations/...seed_confusion_hints_v2.sql`)
  are architecture-independent and survive.
- The eval gate hard and soft criteria (`docs/EVAL_GATE.md`), with
  one removal: criterion 10 (MediaPipe per-frame detection-success
  ≥ 95%) is dropped because there is no MediaPipe in the pipeline
  to measure.
- Signer-disjoint splits (`docs/DATASET.md` §4) and the existing
  signer-to-split manifests in `training/splits/`. The classifier
  changes; the splits do not.
- Fairness commitments and per-demographic accuracy reporting
  (`docs/DATASET.md` §5, `docs/EVAL_GATE.md` §1). Under raw-RGB
  Path B the classifier *can* in principle learn skin tone as a
  spurious feature; per-Fitzpatrick reporting and the ≤ 10 pp gap
  criterion are therefore more — not less — important.
- Privacy architecture: local inference, no frames leaving the
  device (`docs/PRIVACY.md` §1, §2). The privacy posture is
  strictly stronger under Path B because there is no third-party
  vendor (Google's MediaPipe CDN) in the inference path.
- Deployment platform: Vercel + Supabase + Cloudflare R2 (ADR 0003).
- Public-sources-only sourcing for the pilot (ADR 0004).
- Public-data-only training (ADR 0008). The data the new pipeline
  consumes is raw video from the same datasets, just without the
  MediaPipe extraction stage between ingest and training.
- ASL Citizen inclusion (ADR 0009). The MSR-LA research-purpose
  scope of slice-1 use is unchanged; the slice-2 commercial-cliff
  is unchanged. The accuracy projection in ADR 0009 was computed
  under the landmark architecture and no longer applies — the
  inclusion decision (raw video, license-permitted for the pilot)
  stands.
- Classical CV permitted per ADR 0005. **Load-bearing again** for
  slice-1 augmentation, since landmark normalization no longer
  absorbs visual variance for us.

---

## What's superseded

ADR 0006 in its entirety. The landmark-based architecture is no
longer the chosen recognition path. ADR 0006 is *not* deleted; it
remains in the repository as the historical record of the
architecture that shipped v1.0.1 / v2.0.0 / v2.1.0 under the now-
withdrawn permissive reading. ADR 0006 gains a `Superseded by
ADR 0010 on 2026-05-20` line below its `Status` header so a reader
who lands on the old ADR is routed forward.

ADR 0001 is **reinstated** as the governing recognition-architecture
decision. ADR 0001's header gains a `Superseded by ADR 0006 on
2026-05-19, reinstated by ADR 0010 on 2026-05-20` line.

---

## Consequences

### Data pipeline (`docs/DATASET.md`, `docs/ROADMAP.md` Phase 3)

- Per-sign clip target reverts to **~200 clips/sign** (ADR 0001 Path
  B sizing). Available raw video across WLASL + ASL Citizen +
  Sem-Lex gives ~30–90 clips/sign across our 75-sign vocabulary —
  well below the Path B target. This is the dominant accuracy
  ceiling and is named in the T5 honest-outcome estimate.
- Cleaning pipeline drops the MediaPipe Holistic extraction stage.
  Output is per-clip MP4 + per-clip metadata. No `.npy` keypoint
  tensors.
- Raw clips remain retained in R2 / on the Modal volume; they are
  the actual training inputs again, not intermediate artifacts.

### Training (`docs/MODEL.md`, `docs/ROADMAP.md` Phase 4)

- Training time per run rises from minutes (BiLSTM on keypoints) to
  hours (3D CNN on video tensors). GPU rental cost rises
  correspondingly. Iteration cadence slows.
- The BiLSTM / Transformer classifiers in `training/classifier/model.py`
  are replaced by `training/classifier/cnn.py` (the new R(2+1)D-style
  3D CNN). The old file remains until T3 commit 2 deletes it.

### Augmentation

- Keypoint-level augmentations (coordinate jitter, temporal stretch,
  keypoint dropout) are deprecated. They were meaningful only on
  keypoint inputs; they have no analogue under raw-RGB Path B.
- Pixel-level augmentations return as the slice-1 augmentation
  surface: random spatial crop, color jitter, brightness/contrast,
  MOG2-based background swap (classical CV under ADR 0005), small
  affine transforms, conditional horizontal flip via `flippable:
  true`.

### Inference (`docs/ARCHITECTURE.md` §2.3)

- The per-attempt pipeline loses the MediaPipe extraction step.
  Frames are captured, cropped to the green box, resized to H×W,
  and fed directly as `(16, H, W, 3)` to the classifier.
- End-to-end latency budget loosens from 600 ms (the tightened
  ADR 0006 target) back to roughly 1 s (the ADR 0001 Path B
  target). Per-clip classifier inference target: ≤ 300 ms.
- Browser bundle target loosens from ≤ 5 MB combined (MediaPipe
  Tasks Web + classifier) to ≤ 10 MB classifier-only, because the
  classifier is now substantially larger (~5–10M params float32,
  INT8 quantized for shipping).

### Hint system (`docs/ARCHITECTURE.md` §5)

- Slice-1 hint behavior is unchanged. The three layers and the
  confusion-pair lookup all survive. The 120 already-seeded
  confusion-pair hints remain valid — they reference (target,
  predicted) sign pairs, which are architecture-agnostic.
- Slice-2 parameter-aware hints (`docs/ARCHITECTURE.md` §5 closing
  paragraph) become harder under raw-RGB Path B than they were
  under landmarks. The keypoint sequences that already encoded
  handshape geometry / location / palm orientation are gone; a
  parameter-prediction head would need to learn those parameters
  from pixels rather than read them off coordinates. This is
  acknowledged but not blocking — multi-head classification from
  raw video is a researched problem with workable approaches.

### Fairness reporting (`docs/EVAL_GATE.md`)

- Hard criterion 10 (MediaPipe landmark detection success ≥ 95% of
  test clips, broken out by demographic) is removed. There is no
  MediaPipe in the pipeline to measure.
- The remaining nine hard criteria are unchanged. The per-Fitzpatrick
  ≤ 10 pp gap criterion (criterion 3) becomes more important under
  raw-RGB Path B because the classifier can now see skin tone
  directly. Augmentation choice (background swap, color jitter) is
  the architectural defense.

### No-pretrained-pipeline evidence (`docs/MODEL.md` §7)

- The audit surface returns to ADR 0001's specification:
  `training/classifier/cnn.py` (new) plus the init logic embedded in
  it. No `load_state_dict` calls. No external classifier weight URLs.
  No MediaPipe imports anywhere — neither in the training pipeline
  (`training/data/clean.py` loses its MediaPipe extraction stage,
  `training/keypoints.py` is deleted) nor in the frontend
  (`lib/mediapipe/*`, `lib/keypoints.ts`,
  `hooks/use-landmark-extractor.ts` are deleted; the
  `@mediapipe/tasks-vision` dep is removed).

---

## Rejected alternatives

- **Keep the landmark-based architecture and argue Requirement 7's
  strict reading is wrong.** Rejected. The strict reading is the
  defensible reading. Arguing the permissive reading after it has
  been withdrawn is bad-faith engagement with the brief.
- **Train our own landmark detector from scratch (ADR 0001's old
  Path A).** Rejected in ADR 0001 as research-scale work; remains
  so. Building a competitive hand/pose detector from scratch is a
  multi-year project. Not in scope.
- **Hybrid pipeline: classical CV preprocessing (skin segmentation,
  optical flow) → small classifier over the features.** Rejected in
  ADR 0001 as Path C — classical CV thresholds historically fail
  across skin tone and lighting; building a fairness-respecting
  system on that substrate is harder than building it on Path B.
- **Use a pretrained image classifier (e.g., ImageNet ResNet) as a
  feature extractor.** Rejected for the same reason as MediaPipe:
  pretrained vision component, out of scope under the strict
  reading.
- **Train a 3D CNN but warm-start from a pretrained video model
  (Kinetics, SSv2).** Rejected. Pretrained component.

---

## Honesty about the reversal

This is the second project-level course correction on recognition
architecture. ADR 0001 chose Path B under the strict reading.
ADR 0006 pivoted to landmarks under the permissive reading.
ADR 0010 reverts to Path B under the strict reading restored.

The history is kept visible on purpose. ADR 0001 / ADR 0006 / ADR
0010 form a chain a reader can walk in either direction. The
v1.0.1 / v2.0.0 / v2.1.0 artifacts and their validation reports
(`docs/validation/v1.md`, `docs/validation/v2.md`) stay in the
repo as historical records of what was shipped under ADR 0006,
each annotated as superseded under this ADR. No silent revisions.

The honest outcome estimate for v3.0 under this reversal: **30–50%
top-1 accuracy on 75 signs**, well below the 85% eval-gate floor.
ADR 0001 Path B was sized for ~200 clips/sign; we have ~30–90/sign
in our public-source corpus. The slice-1 acceptance pattern from
v1.0.1 / v2.0.0 / v2.1.0 continues: name the gap, do not paper
over it. The slice-2 paths to closing the gap (ADR 0004 instructor
engagement, ADR 0008 recording-tool framework) remain unchanged.

---

## How this is verified

- No MediaPipe imports anywhere in the repository after T3:
  `grep -r 'mediapipe' --include='*.ts' --include='*.tsx' --include='*.py'`
  returns nothing.
- `@mediapipe/tasks-vision` removed from `package.json` after T3
  commit 1.
- `mediapipe==0.10.18` removed from `training/requirements.txt`
  after T3 commit 2.
- `training/classifier/cnn.py` (new in T4) uses
  `torch.nn.init.kaiming_normal_` for all conv/linear weights and
  contains no `load_state_dict` call and no external weight URL.
- `lib/inference/classifier.ts` (rewired in T3) takes a video
  tensor of shape `(B, T=16, H, W, 3)` and feeds it to ONNX Runtime
  Web directly. No classical CV step in the inference path,
  honoring ADR 0005's "classical CV at training-time only" boundary.
- The validation report `docs/validation/v3.md` (new in T5) names
  this ADR by number and discloses the data-quality ceiling that
  produced its accuracy.
