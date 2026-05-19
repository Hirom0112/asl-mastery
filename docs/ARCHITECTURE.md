# Architecture

> The system-level view of every component and how data flows between
> them. This document is technical, but every component is justified
> against the pedagogical theory in `PEDAGOGY.md`. If a piece of the
> architecture cannot be defended pedagogically, it does not belong.

---

## 1. High-level shape

The system has five logical surfaces:

1. **Learner app** (browser) — where the learner practices.
2. **Recording tool** (browser, admin-gated) — where contributors
   record training video.
3. **Inference runtime** (browser) — runs the trained recognition model
   on the learner's device. Never leaves the device during normal use.
4. **Backend** (Next.js API routes + Postgres + object storage) — auth,
   progress, model artifact serving, observability event sink. Never
   receives video frames during normal use.
5. **Training pipeline** (offline, Python/PyTorch on a GPU) — produces
   versioned model artifacts that the inference runtime downloads.

The defining architectural commitment is that the learner's video stays
on the learner's device. Everything else flows from that.

---

## 2. Components, top to bottom

### 2.1 Learner app

Built with Next.js 14 (App Router), React 18, TypeScript, Tailwind CSS,
shadcn/ui. Deployed on Vercel.

Responsibilities:

- Authentication via Supabase Auth (email magic links).
- Onboarding flow that explains the system and captures handedness.
- Practice screen orchestration: prompt → capture → inference →
  evaluation → feedback → state update.
- Mastery dashboard rendering.
- Camera access, MediaRecorder usage, frame capture to canvas, frame
  tensor preparation.
- ONNX Runtime Web loading of the active model version from R2,
  with IndexedDB caching after first download.
- Local-only inference; results plus metadata are what gets posted to
  the backend.
- Settings, consent flows, account management.

Why this stack: TypeScript and React mirror Superbuilders' stack
(Patrick is TS/React/Rust). Next.js App Router gives us a single
codebase for the app, marketing pages, and the admin recording tool.
Tailwind plus shadcn/ui lets us hit production-grade visual polish
without ornament-for-ornament's-sake design overhead. Supabase Auth
is well-supported by Next.js out of the box.

### 2.2 Recording tool

Same Next.js codebase, gated route at `/admin/record`. Only authorized
contributor accounts can reach it.

Responsibilities:

- Display the green-box framing the learner app uses, so training data
  matches deployment conditions exactly.
- Prompt the contributor with the sign to record and a reference video
  loop.
- Capture 2-second clips at the highest resolution the contributor's
  camera supports (we want 720p+ originals so we can re-process later
  if preprocessing changes).
- Capture a 1-second empty-frame clip at session start (contributor
  steps out of frame) for use in background-subtraction-based
  augmentation.
- Capture per-clip metadata: signer id, sign id, timestamp, self-rated
  lighting and background, handedness, sleeve length, optional
  Fitzpatrick scale (with consent), self-rated correctness 1-5.
- Upload clips to a dedicated R2 bucket via signed URLs; index in
  Postgres with full metadata.

This is the only path by which video leaves a device, and only
contributors with explicit consent and admin access use it. Learner
sessions never go through this path.

### 2.3 Inference runtime

ONNX Runtime Web. WebGPU backend primary, WebGL fallback, WebAssembly
final fallback for older browsers.

Pipeline per attempt:

1. The MediaRecorder or getUserMedia stream feeds a hidden video
   element.
2. On user clicking "Record attempt," a 2-second capture window opens.
3. Frames are drawn to an offscreen canvas at the source rate, then
   sampled to exactly 16 frames over the window (frame-rate
   normalization).
4. Each frame is cropped to the green-box region in screen coordinates,
   resized to 112×112 (bilinear), un-mirrored if the preview was
   mirrored, horizontally flipped if the learner is left-handed.
5. Pixel values are scaled to [-1, 1].
6. The (16, 112, 112, 3) tensor is fed to the ONNX model.
7. Logits come out, temperature-scaled, softmaxed to probabilities.
8. The top prediction is compared to the prompted sign.
9. Pass/fail decision uses the per-sign confidence threshold stored in
   the model's bundled config.
10. The result, predicted class id, confidence, and model version are
    posted to the backend along with attempt metadata (timestamp, time
    to attempt, etc.) — never the frames.

Performance targets (mid-range laptop, integrated GPU):

- First-visit model download: ≤ 3 seconds
- Cached visit model warm-up: ≤ 500 ms
- Per-attempt inference: ≤ 300 ms
- End-to-end "submit" to result UI: ≤ 1 second

### 2.4 Backend

Next.js API routes (server actions for mutations). Postgres via
Supabase. Object storage via Cloudflare R2.

Schema (Postgres):

```
users
  id (uuid, pk)
  email (citext, unique)
  handedness (enum: right, left, ambidextrous, unspecified)
  fitzpatrick_scale (int, nullable, with consent)
  created_at, updated_at

vocabulary_items
  id (text, pk — e.g. "thank_you")
  display_gloss (text — e.g. "THANK-YOU")
  reference_video_url (text)
  parameters (jsonb — handshape, location, palm_orientation, movement, nmm)
  pre_attempt_hint (text)
  generic_failure_hint (text)
  flippable (bool — safe to horizontally augment in training)

confusion_pair_hints
  id (uuid, pk)
  target_sign_id (fk vocabulary_items)
  predicted_sign_id (fk vocabulary_items)
  hint (text)
  created_at

model_versions
  id (text, pk — e.g. "v1.3.2")
  artifact_url (text — R2 path)
  config_url (text — R2 path with thresholds, classes, normalization)
  metrics_url (text — R2 path with validation report)
  is_active (bool — exactly one true row)
  promoted_at, promoted_by

attempts
  id (uuid, pk)
  user_id (fk users)
  vocab_id (fk vocabulary_items)
  prompted_at (timestamptz)
  submitted_at (timestamptz)
  predicted_class_id (fk vocabulary_items)
  confidence (float)
  passed (bool)
  hint_shown (text, nullable)
  hint_source (enum: confusion_pair, generic_failure, none)
  learner_disagreed (bool, default false — "I think I did this right" feedback)
  model_version_id (fk model_versions)

mastery_state
  user_id (fk users)
  vocab_id (fk vocabulary_items)
  status (enum: untouched, learning, reviewing, mastered)
  ease (float)
  interval_days (float)
  next_review_at (timestamptz)
  consecutive_passes (int)
  total_attempts (int)
  total_passes (int)
  last_attempt_at (timestamptz)
  pk: (user_id, vocab_id)
```

Endpoints (server actions, not REST):

- `signUp`, `signIn`, `signOut` — Supabase Auth wrappers.
- `updateHandedness`, `updateConsents`.
- `getNextItem(userId)` — runs the scheduler; returns the next sign
  to practice.
- `recordAttempt(userId, attempt)` — writes the attempt row, updates
  mastery_state, returns the hint if failed.
- `flagAttempt(attemptId)` — sets learner_disagreed = true.
- `getMasteryDashboard(userId)` — aggregates mastery_state for the UI.
- `getActiveModelVersion()` — returns model artifact URLs for the
  client to load.

R2 buckets:

- `models/` — model artifacts by version. Public read via signed
  URLs (long TTL).
- `references/` — canonical reference videos. Slice 1: WLASL/MS-ASL clips with attribution (per ADR 0004). Slice 2: instructor-recorded replacements. Public read.
- `raw-training-data/` — contributor uploads from the recording tool.
  Private; access via signed URLs only for the training pipeline.

### 2.5 Training pipeline

Python 3.11+, PyTorch, ONNX, Weights & Biases for experiment tracking.
Runs on a rented GPU (RTX 4090 or A100, hourly cloud rental).

Pipeline stages:

1. **Dataset assembly.** Pull clips from R2 by version. Apply the
   signer-disjoint split assignment from the committed manifest.
2. **Cleaning.** Trim to sign window, normalize framing (crop to green
   box), normalize frame rate to 30 fps, normalize length to 16 frames,
   compute and dedupe by perceptual hash.
3. **Augmentation** (training only). Spatial jitter, color jitter,
   brightness/contrast, Gaussian noise, gamma adjustment, temporal
   crop, speed variation, frame dropout. Background swap via MOG2 mask
   for clips with empty-frame reference. Horizontal flip only for signs
   marked `flippable: true`.
4. **Training.** R(2+1)D-small as specified in `MODEL.md`. AdamW
   optimizer, cosine annealing, label smoothing 0.1, weighted sampling
   for class balance, 80 epochs with early stopping. Track every
   hyperparameter, dataset version hash, and git commit in W&B.
5. **Calibration.** Temperature scaling on validation set. Per-sign
   confidence threshold tuning to ≥90% precision target.
6. **Confusion analysis.** Extract top-3 confusion pairs per sign from
   the validation confusion matrix; write to a JSON config for the
   hint system to consume.
7. **Export.** PyTorch → ONNX. Verify outputs match within tolerance.
8. **Quantization.** Post-training dynamic int8 quantization. Verify
   accuracy delta ≤ 1% versus float32 on validation set.
9. **Validation report generation.** Run the held-out test set; produce
   `VALIDATION.md` content (overall accuracy, per-sign, per-condition,
   per-demographic, full confusion matrix, reliability diagram, known
   limitations).
10. **Artifact bundling.** Model file + config (thresholds, class list,
    normalization params) + validation report + dataset manifest hash.
    Upload to R2 under a new version id.
11. **Promotion.** A human compares the new version's validation report
    against the eval gate criteria; if passing, runs the promote
    command, which flips `is_active` in `model_versions` and triggers
    a client cache invalidation.

The promotion step is intentionally manual for this pilot. Automated
promotion is a slice-2 candidate; for slice 1, a human eye on the eval
gate is the right tradeoff.

---

## 3. Data flow — the learner's path

```
  ┌─────────────────────────────────────────┐
  │ Learner opens app, signs in             │
  └─────────────────┬───────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │ Client loads active model artifact      │
  │ from R2, cached in IndexedDB if seen    │
  └─────────────────┬───────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │ Client calls getNextItem(userId)        │
  │ Backend scheduler reads mastery_state,  │
  │ returns the next vocab_item             │
  └─────────────────┬───────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │ Practice screen: prompt + reference     │
  │ video + pre-attempt parameter card      │
  └─────────────────┬───────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │ Learner presses Record. Countdown,      │
  │ 2-second capture window, frames sampled │
  │ to 16, cropped, resized, normalized     │
  └─────────────────┬───────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │ Inference runs locally. Logits →        │
  │ temperature scale → softmax → top class │
  │ + confidence                            │
  └─────────────────┬───────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │ Pass/fail decided via per-sign          │
  │ threshold. Hint selected if fail.       │
  └─────────────────┬───────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │ Client posts attempt metadata           │
  │ (NOT frames) to backend                 │
  └─────────────────┬───────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │ Backend writes attempts row, updates    │
  │ mastery_state via modified-SM-2.        │
  │ Returns updated state to client.        │
  └─────────────────┬───────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │ Client renders result. If pass:         │
  │ celebration + next-item button.         │
  │ If fail: targeted hint + retry button + │
  │ reference video replay option.          │
  └─────────────────────────────────────────┘
```

The only data leaving the device during practice is the metadata in
the second-to-last box. No frames, no clips, no images.

---

## 4. The scheduler

Modified SM-2 algorithm. The classic SM-2 was designed for declarative
recall; we adapt it for motor skill, where:

- Failures should reset the interval more aggressively (motor skills
  decay faster when wrong patterns are reinforced).
- Mastery requires multiple successful retrievals at long intervals,
  not just three consecutive passes.

State per (user, sign):

- `ease` ∈ [0.13, 2.5], default 1.3.
- `interval_days` (float), default 0.
- `consecutive_passes` (int), default 0.
- `status` ∈ {untouched, learning, reviewing, mastered}.

Transitions:

- **Pass when status = untouched** → status becomes "learning",
  interval = 0 (practice again same session), ease unchanged.
- **Pass when status = learning** → consecutive_passes += 1; if
  consecutive_passes ≥ 2 in current session, status = "reviewing",
  interval = 1, ease unchanged.
- **Pass when status = reviewing** → consecutive_passes += 1, interval
  *= ease, ease += 0.1 (capped at 2.5). If interval ≥ 7 days and
  consecutive_passes ≥ 3, status = "mastered".
- **Fail (any status above untouched)** → consecutive_passes = 0,
  ease -= 0.2 (floored at 0.13), interval = max(interval * 0.3, 0).
  Status drops one level (mastered → reviewing → learning).

This is intentionally hand-rolled rather than imported. We want every
constant defensible against motor-learning research (`PEDAGOGY.md`)
and tunable from the validation data we collect.

Item selection per request to `getNextItem`:

1. Items with `next_review_at <= now` and `status != untouched` are
   review-due. Pick the most overdue one.
2. If no review-due items and learner's rolling pass rate over the last
   10 attempts is ≥ 70%, introduce a new untouched item.
3. If learner's rolling pass rate is < 70%, prioritize a struggling
   "learning" item rather than introducing new content. This is
   cognitive load theory in code form: do not pile new content on a
   learner who is already overloaded.

---

## 5. The hint system

Three layers, in priority order:

**Layer A — pre-attempt priming card.** Shown before recording. Lists
the five sign parameters concisely. For slice 1, authored against
ASL-LEX 2.0 parameter codes and Lifeprint per-sign instructional
notes (per ADR 0004); fluent-signer validation deferred to slice 2. This is
cognitive load theory applied: priming attention before the attempt
reduces extraneous load during the attempt.

**Layer B — confusion-pair-aware hint.** When the model's top
prediction is a different sign with reasonable confidence (e.g.,
predicted class confidence ≥ 0.4 and predicted class ≠ target), look
up the confusion pair (target_sign, predicted_sign) and serve the
authored hint. Example: "You signed something that looks like GOOD.
The difference: GOOD moves down to your other palm; THANK-YOU moves
forward, away from you."

**Layer C — generic per-sign failure hint.** Used when the model has
low confidence everywhere, or when no confusion-pair hint exists for
the predicted/target combination. Each sign has one authored generic
hint.

Slice 1 hint copy is authored against ASL-LEX 2.0 phonological data without Deaf-signer review; Deaf-signer review of every hint is a slice-2 production-deployment requirement (ADR 0004).
Hints reference ASL's five sign parameters explicitly (handshape,
location, palm orientation, movement, non-manual markers) so the
learner builds linguistic vocabulary alongside performance skill.

---

## 6. The eval gate, in the architecture

The eval gate is not just a document — it is a check executed by the
training pipeline at the end of every training run and visible in CI
when a new model version is proposed for promotion.

Mechanics:

1. The training pipeline runs the held-out test set against the
   candidate model.
2. It computes every metric required by `EVAL_GATE.md`.
3. It writes a structured report (`validation_report.json` plus a
   human-readable `VALIDATION.md`).
4. The promotion command refuses to flip `is_active` unless the report
   passes every criterion.

This makes "no vibes-based AI" a property of the deployment
infrastructure, not a wish.

---

## 7. Privacy architecture

The architectural commitments — not just policy statements:

- Inference runs in the browser. The model file is downloaded; frames
  never leave.
- No third-party analytics receives canvas data; PostHog is configured
  to ignore DOM elements with class `no-track-canvas`, and the camera
  preview canvas carries that class.
- Sentry is configured with `Replay` either disabled or with
  `maskAllInputs: true, blockAllMedia: true` so camera streams are
  never captured in error replays.
- The `raw-training-data/` R2 bucket is private; only the training
  pipeline (server-side, with a credential not present in the
  client) can read from it.
- Contributors to training data sign a consent form before being
  granted access to the recording tool; consent records are stored in
  Postgres with timestamps.
- Optional demographic data (Fitzpatrick scale) requires affirmative
  opt-in with explicit explanation of why we ask (model fairness
  evaluation); never required.
- Account deletion removes all attempts and mastery rows; if the user
  was a contributor, their clips are deleted from R2 and removed from
  future training runs.

---

## 8. Observability

PostHog for product analytics (event names: `attempt_recorded`,
`mastery_reached`, `session_started`, `session_ended`, `hint_shown`,
`learner_disagreed_with_result`). No video, no images, no PII beyond
what is necessary.

Sentry for error tracking with masking rules as above.

Custom dashboards in PostHog or a simple Next.js admin page:

- Per-sign pass rate distribution.
- Per-user pass rate distribution (with alerts on users < 30% over
  50+ attempts).
- Inference latency percentiles.
- Model load time percentiles.
- Hint efficacy per confusion pair (hint shown → next-attempt pass
  rate).
- Learner-disagreed flag rate per sign (signals false-negative-heavy
  signs that need data collection or retraining).

---

## 9. Why this architecture, in one paragraph

We are building a system whose entire design — not its marketing copy
— is shaped by the pedagogical commitment that mastery, not
engagement, is the goal. Local inference protects the learner's video,
which makes the system trustworthy enough to deploy in school
contexts. The mastery state model and scheduler encode the spaced-
retrieval research directly in code. The hint system layers progress
from generic to specific feedback in lockstep with what the model
knows. The eval gate makes AI quality a contract rather than a
promise. Self-paced exit on mastery makes the "kids get off the app"
principle a literal database state. Every architectural choice has
both a software justification and a pedagogical one, as required by
the two-theories principle (`PEDAGOGY.md`).
