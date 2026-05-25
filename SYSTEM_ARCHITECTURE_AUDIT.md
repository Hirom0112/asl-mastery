# Mastered — System Architecture Audit (presentation reference)

> Built from a direct read of the codebase on 2026-05-25 (branch `main`, HEAD `e670863`). Every claim is cited to a file/line or a doc. Where the code and the docs disagree, the code wins and the disagreement is flagged. Anything not found is marked **NOT FOUND IN CODEBASE** rather than guessed.
>
> Canonical source-of-truth files: `STATUS.md`, `TRAINING.md` (gitignored handoff), `avatar.md` (gitignored handoff), `docs/decisions/0011`, `docs/decisions/0012`.

---

## 1. Project one-liner

**Non-technical:** Mastered is a web app that teaches American Sign Language one sign at a time — a 3D avatar shows you the sign, you sign it back into your webcam, and the app tells you whether you got it right and decides when to make you practice it again until you've mastered it.

**Technical:** A browser-based, mastery-driven ASL vocabulary trainer (Next.js 16 App Router + Supabase + react-three-fiber) whose recognition stack is a **strict from-scratch computer-vision pipeline** — four small CNNs the project trained itself (hand detector → hand-landmark regressor → upper-body pose regressor, plus a face detector for framing) feed a learned temporal classifier (a dilated TCN) over 21-keypoint hand trajectories, all exported to ONNX and run **entirely in the learner's browser** via `onnxruntime-web` (WebGPU→WASM). No MediaPipe, no pretrained backbones, no remote inference — webcam frames never leave the device. A modified-SM-2 spaced-repetition scheduler in Postgres governs what to practice next, and a Cloudflare-R2-served / Vercel-hosted deployment keeps per-learner cost near zero. The hard constraint (ADR 0012: every recognition weight trained in-house on provenance-audited public labels) is the project's deliberate moat, traded against accuracy and timeline.

---

## 2. The user journey (happy path)

1. **Landing → auth** (`app/page.tsx`, `app/sign-in/page.tsx`). Three sign-in routes (ADR 0007, `lib/auth/actions.ts`): email magic-link (`signInWithOtp`), Google OAuth (`signInWithOAuth`), or **anonymous demo** (`signInAnonymously`, `lib/auth/actions.ts:43-50`) which redirects straight to `/practice`. OAuth/magic-link returns through `app/auth/callback/route.ts` (`exchangeCodeForSession`).
2. **Onboarding** (`app/welcome/page.tsx`, `lib/auth/profile.ts:45-73`). First-time users land on `/welcome`; collects **display name** (stored in `auth.user_metadata.display_name`) + **handedness** (right/left). `completeOnboarding` sets `users.onboarded_at` then redirects to `/practice`. The practice page redirects un-onboarded users back here (`app/practice/page.tsx:51-53`).
3. **Practice page load (server)** (`app/practice/page.tsx`, `export const dynamic = "force-dynamic"`). Server component: `getUser()` (redirect to sign-in if absent), resolve which sign to show — explicit `?sign=` (gated by `isSignUnlocked`) / `?skip=1` override / else the scheduler's `getNextItem()` — then in parallel loads active model row, the user's handedness/onboarding, and the full rank-ordered vocab progression for the sidebar.
4. **Practice screen render** (`components/practice/runner.tsx`). Three-column layout: sidebar (category accordion of 80 signs) | center (the **3D avatar** demonstrating the prompted sign) | right (the learner's **camera** + Record). On mount it **preloads the four ONNX models** (`preloadKeypointModels`, `runner.tsx:66-68`) so the first attempt isn't cold.
5. **Avatar demonstration** (`components/practice/sign-avatar.tsx`). Production now plays **3D-LEX optical-mocap GLBs** (`public/3dlex/<sign>.glb`, 70 of 80 signs) on a Ready-Player-Me rig; the 10 signs without mocap render the figure at rest (`avatar.md` handoff 2026-05-25). Rendered with react-three-fiber / drei `useGLTF` + `useAnimations`.
6. **Record** (`components/practice/camera-capture.tsx`). `getUserMedia({video:{720×720, frameRate:30}})`. On Record: a **3-2-1 countdown** (≈700 ms each), then a **2-second capture** grabbing **16 frames evenly across 2000 ms** (`CAPTURE_MS=2000`, `VIDEO_TEMPORAL_LENGTH=16` → ~8 fps sampling) as full-resolution `ImageBitmap`s. Frames are non-mirrored for the model (left-handed users are mirrored to the training convention, `camera-capture.tsx:179-184`). **Frames never leave the device** (privacy: hidden canvas tagged `no-track-canvas`).
7. **In-browser recognition** (`lib/inference/keypoint-predict.ts` → `keypoints.ts` → `sign_matcher.ts`). For each captured frame: hand detector (320²) → per-hand crop → landmark regressor (224²) → 21 keypoints; full-frame pose (256²) → 8 keypoints. Frames with no detected hand are dropped (`drop_handless_frames`). Each kept frame → **108-D hand-relative feature vector** (`frameToFeaturesV2`). The sequence is NaN-aware resampled to **32 timesteps**, fed to `sign_classifier_v3.onnx` → softmax over 80 classes.
8. **Pass/fail decision** (`keypoint-predict.ts:122-135`). **Pedagogically forgiving:** the attempt passes if the prompted sign is in the **top-4** of the ranked softmax (rationale: an 80-class softmax puts even a correct answer at only ~0.3–0.5; top-5 accuracy is ~93%). On any inference error it falls back to `stubPredict` (`runner.tsx:85-88`).
9. **Result + persistence** (`runner.tsx:92-117`, `lib/scheduler/actions.ts:126-180`). `recordAttempt` inserts an **attempts** row (metadata only — sign id, predicted id, confidence, pass/fail, timestamps, hint) and **upserts mastery_state** via the modified-SM-2 update. UI shows pass ("Nice — <sign>") or fail ("Not quite") with a confidence read-out; "Skip for now" appears only after 3 misses (`SKIP_AFTER_FAILS=3`); a "I think I did this right" button flags a false negative (`flagAttempt`).
10. **Next item** (`router.refresh()` → step 3 re-runs the scheduler).

**Latency expectations (from code/docs, mostly *targets* not measured):**
- Capture window: **fixed ~2.0 s + ~2.1 s countdown** (the dominant, intentional latency).
- Per-attempt inference: **~1–3 s "Analyzing…"** — ~16 frames × up to 4 ONNX forward passes each, stated in `TRAINING.md` WEB SHIP notes as "tunable, untested in-browser."
- Doc targets (largely **stale/aspirational**, ADR 0011 §"Inference" / `docs/EVAL_GATE.md` criterion 5): ≥20 FPS detector throughput; end-to-end p95 ≤ 600 ms per attempt (the *recognition* part, not the capture window). **No measured p50/p95/p99 latency exists in the repo** — flag as a gap.

---

## 3. Tech stack — full inventory

### Frontend
| Item | Version | Why | Replaces / rejected |
|---|---|---|---|
| Next.js (App Router) | **16.2.6** (`package.json:36`) | SSR practice page, server actions as the API, Vercel-native | ADR 0003 rejected Railway/all-Vercel-storage; docs say "Next 14" but code is 16 (stale doc) |
| React | **19.2.4** | UI | docs say React 18 (stale) |
| TypeScript | ^5 (strict, `tsconfig.json`) | type safety; `pnpm typecheck` in CI | — |
| Tailwind CSS | v4 (`@tailwindcss/postcss`) + `tw-animate-css` | styling (terracotta editorial theme) | — |
| shadcn/ui | ^4.7.0, style `base-nova` (`components.json`) | component primitives over `@base-ui/react` ^1.5.0 | — |
| react-three-fiber / drei / postprocessing | 9.6.1 / 10.7.7 / 3.0.4; **three** ^0.184.0 | the 3D practice avatar (GLB mocap playback) | retargeting onto X-Bot via `SkeletonUtils.retargetClip` failed (`avatar.md` dead-ends) |
| lucide-react | ^1.16.0 | icons | — |
| **State management** | React hooks + Server Components only | per-attempt state machine in `runner.tsx` (`useState`/`useTransition`) | **No Redux/Zustand/Jotai** — none in deps |
| Build tooling | Next build (Turbopack default), **pnpm 10** | — | — |

### Backend / API layer
- **Language/runtime:** TypeScript on Node (Next.js server). **No separate API server** — the "API" is **Next.js Server Actions** (`"use server"` modules: `lib/scheduler/actions.ts`, `lib/auth/actions.ts`, `lib/auth/profile.ts`). ADR/ARCHITECTURE: "server actions, not REST."
- **Only HTTP route handler:** `app/auth/callback/route.ts` (OAuth code exchange). `middleware.ts` runs Supabase session refresh on every non-static request (`lib/db/middleware.ts`).
- **Hosting:** Vercel (ADR 0003).

### ML / inference layer
- **Inference framework:** `onnxruntime-web` ^1.26.0 (`package.json:37`), **in-browser**, execution providers `["webgpu","wasm"]` (WebGPU primary, WASM fallback) — `keypoints.ts:54`, `keypoint-predict.ts:50`.
- **Models served:** 4 ONNX files in `public/models/` — `hand_detector_v0.onnx` (9.2 MB), `hand_landmarks_v0.onnx` (17.1 MB), `pose_detector_v0.onnx` (17.0 MB), `sign_classifier_v3.onnx` (8.4 MB) + `sign_classifier_v3.config.json`. **Loaded directly from `/public` — no DB `model_versions` row needed** (`keypoint-predict.ts:14,23`).
- **Training framework:** PyTorch **2.4.1**, torchvision 0.19.1 (**transforms only**, no `torchvision.models`), onnx 1.17.0, onnxruntime 1.20.0, wandb 0.18.5, scipy, OpenCV-headless, PyAV (`training/requirements.txt`). **MediaPipe explicitly removed** (`requirements.txt:13`).
- **Serving approach:** static ONNX from CDN (`public/`) → client-side session, **fp32, no quantization** on the four live models (`training/detectors/export_onnx.py`, opset 17). (INT8 `quantize_dynamic` exists only in the **dead** 3D-CNN path `training/classifier/export.py`.)

### Data layer
- **Postgres:** Supabase (`@supabase/ssr`, `@supabase/supabase-js`). 6 tables (see §4). Access control = **Row-Level Security**, not app code.
- **Object storage:** Cloudflare **R2** — buckets `asl-mastery-models`, `asl-mastery-references`, `asl-mastery-raw-training-data` (`.env.example:13-24`). Chosen for **zero egress** (ADR 0003). Reference videos also served from an `r2.dev` public URL.
- **Training data volume:** Modal network volume `asl-mastery-data` (~100 GB; on-disk `data/` measured **104 GB**, excluded from deploy via `.vercelignore`). Near a 500k-inode cap (operational constraint in `TRAINING.md` §6).
- **Browser cache:** ONNX sessions cached per page lifetime (module-level promises); ARCHITECTURE mentions IndexedDB artifact cache.
- **Vector store:** **None.** (Recognition is keypoint-classification, not embedding retrieval.)

### Real-time / streaming
- **Camera:** `getUserMedia` 720×720@30 (`camera-capture.tsx:106`). **No WebRTC, no WebSocket, no server streaming** — capture and inference are entirely local and synchronous within the browser tab.
- **MediaPipe:** **forbidden** (ADR 0012 anti-drift rule 1; verified `grep mediapipe` returns nothing in shipped code).
- A `StreamingMatcher` class exists (`lib/inference/sign_matcher.ts:473`, rolling 48-frame ring buffer for frame-by-frame live scoring) but the **shipped path uses the batched `predictFromFrames`**, not streaming.

### Auth
- **Supabase Auth**: email magic-link + Google OAuth + **anonymous demo sessions** (ADR 0007). Anonymous and permanent users are treated identically by RLS (`rls.sql` policies are pure `auth.uid() = user_id`). Sessions refreshed in middleware. Inactive anonymous users cleaned after 30 days by `cleanup_inactive_anonymous_users()` (**defined but must be scheduled manually via pg_cron** — `20260519110100`).

### Infra / hosting
- **Vercel** (Next.js app + serverless) · **Supabase** (Postgres + Auth) · **Cloudflare R2** (artifacts/references) · **Modal** (serverless GPU training: L4/A100/H100/A10G). Total pilot infra **< $50** (ADR 0003 §cost); cumulative Modal training spend **~$50** (`STATUS.md`).
- No Kubernetes / no container orchestration for the app (Vercel-managed). No CDN config beyond Vercel + R2.

### Observability
- **PostHog** product analytics (`NEXT_PUBLIC_POSTHOG_KEY`), configured to **never receive video** (canvas tagged `no-track-canvas`; named-event allowlist) — `docs/PRIVACY.md`, ARCHITECTURE §8 events: `attempt_recorded`, `mastery_reached`, `hint_shown`, `learner_disagreed_with_result`, etc.
- **Sentry** errors (`SENTRY_DSN`), session replays disabled or fully masked (`docs/PRIVACY.md`).
- **wandb** for training-run tracking (`training/requirements.txt`).
- ⚠️ These are **declared in `.env.example` and docs**; no Sentry/PostHog SDK appears in `package.json` dependencies — instrumentation may be **planned/partial**, not wired. Flag for verification.

### CI/CD
- **GitHub Actions** (`.github/workflows/`): `ci.yml` on push + PR to `main` runs `format:check → lint → typecheck → test (vitest) → build`; `eval-gate.yml` runs `scripts/check_eval_gate.py` when a `docs/validation/v*.md` changes (refuses promotion unless hard criteria pass).
- **Husky** pre-commit → `lint-staged` (Prettier + ESLint --fix). Prettier (printWidth 100, double quotes, trailing comma all). ESLint extends `eslint-config-next`, ignores `training/`, `dataset/`, `runs/`, `artifacts/`.

---

## 4. System architecture — components and data flow

### Components

| Component | Responsibility | Inputs → Outputs | Sync/async | Where it runs |
|---|---|---|---|---|
| **Practice page (RSC)** `app/practice/page.tsx` | Resolve next sign, gather sidebar/progression, gate auth | URL params + session → props for runner | sync server render | Vercel (server) |
| **PracticeRunner (client)** `components/practice/runner.tsx` | Per-attempt state machine; orchestrate capture→predict→record | user clicks → attempt result | sync UI, async actions | Browser |
| **CameraCapture** `camera-capture.tsx` | Webcam, countdown, 16-frame grab | `getUserMedia` → `ImageBitmap[]` + video tensor | async | Browser |
| **SignAvatar** `sign-avatar.tsx` | Demonstrate the prompted sign | `signId` → GLB animation | sync render | Browser (WebGL) |
| **Keypoint pipeline** `keypoints.ts` | Run detector/landmark/pose ONNX per frame | `ImageBitmap` → `RawFrame{hands,pose}` | async (ONNX) | Browser (WebGPU/WASM) |
| **Feature + classifier** `keypoint-predict.ts`, `sign_matcher.ts` | 108-D features, resample, classify | `RawFrame[]` → top-k + pass/fail | async | Browser |
| **Scheduler actions** `lib/scheduler/actions.ts` | SRS read/write (`getNextItem`, `recordAttempt`, `flagAttempt`, `getItemById`) | attempt data → DB rows | async server action | Vercel (server) |
| **Scheduler core** `lib/scheduler.ts` | Pure modified-SM-2 functions | mastery state + outcome → next state | sync, pure | Vercel (server) |
| **Vocab progression** `lib/scheduler/vocab-progression.ts` | Rank-ordered unlock state for sidebar | DB → `{mastered\|current\|locked}[]` | async | Vercel (server) |
| **Supabase Postgres** | Users, vocab, attempts, mastery, models, hints | SQL via RLS | async | Supabase |
| **Cloudflare R2** | Model artifacts + reference videos | HTTPS GET | async | Cloudflare edge |
| **Auth/middleware** `middleware.ts`, `lib/db/*` | Session refresh, RLS-scoped clients | cookies → refreshed session | async | Vercel edge/server |
| **Modal training** `training/modal_app.py` | Train all CV models, extract trajectories, export ONNX | labeled data → `.pt` → `.onnx` | offline batch | Modal GPU |
| **SMPLest-X extractor** `training/smplestx_modal.py` | Avatar motion (SMPL-X) — **not recognition** | clip → `.npz` params | offline batch | Modal A10G |

### Data-flow contracts (numbered, arrows you can draw)

1. **Browser → Vercel (RSC load).** `GET /practice?sign=&skip=` → server reads Supabase session (cookies). Response: rendered HTML + serialized `NextItem` (`{vocabId, displayGloss, category, referenceVideoUrl, preAttemptHint, flippable}`) + progression array. **Transport HTTP, sync render.**
2. **Browser → R2/`public` (model fetch).** On mount, 4 × `InferenceSession.create("/models/*.onnx")` + `fetch("/models/sign_classifier_v3.config.json")`. **HTTP GET, async, cached per page.** (Live models load from Vercel `/public`, not R2; R2 is the historical `model_versions.artifact_url` path, currently unused.)
3. **Camera → frames (in-process).** `startCapture()` → `{frames: ImageBitmap[16], frameWidth, frameHeight, promptedAtIso, submittedAtIso}`. **Local, async, no network.**
4. **Frames → keypoints (in-process, ONNX).** Per frame: detector input `Float32Array(1,3,320,320)` RGB[0,1] → outputs `heatmap(1,1,80,80)` + `size(1,2,80,80)`; peak-pick (3×3 max-pool, threshold 0.3, 2nd hand 0.5, IoU dedup 0.5, max 2 hands) → boxes; per box landmark input `(1,3,224,224)` → `coords(1,21,2)∈[0,1]`; pose input `(1,3,256,256)` → `coords(1,8,2)`. Yields `RawFrame{hands:[{keypoints:[21][2]}], pose:[8]}`. **Local, async.**
5. **Keypoints → features → classifier (in-process).** Drop handless frames → `frameToFeaturesV2` → 108-D rows → `resampleTrajectory(n→32)` → NaN→0 → ONNX classifier input `Tensor(float32,[1,32,108])` → output logits `[80]` → temperature softmax → ranked top-k; pass = target in top-4. **Local, async.**
6. **Browser → Vercel (recordAttempt server action).** RPC-style POST of `{vocabId, promptedAtIso, submittedAtIso, predictedClassId, confidence, passed, hintShown, hintSource, mediapipeDetectionFailed:false, modelVersionId}` → server. **Transport: Next server action (HTTP), async.**
7. **Vercel → Postgres (write).** `INSERT attempts` then `UPSERT mastery_state` (computed by `applyPass`/`applyFail`). **Two separate statements, NOT a single transaction** (flagged). RLS scopes to `auth.uid()`. Returns `{attemptId, newStatus, reachedMastery}`.
8. **Browser → Vercel (flagAttempt).** `UPDATE attempts SET learner_disagreed=true` (guard trigger permits only this column). **Async server action.**
9. **Offline: Modal → R2/repo.** Training runs write `.pt` to the Modal volume; `export_onnx.py` → `.onnx`; artifacts copied into `public/models/` and committed (commit `fcad132`). **Batch, out-of-band.**

### Tables (`supabase/migrations/20260519100000_init.sql` + later)
- **users** — `id (=auth.users)`, `email`, `handedness`, `fitzpatrick (1-6 nullable)`, `onboarded_at`. Populated by `handle_new_auth_user()` trigger (fires for anonymous too).
- **vocabulary_items** — `id (slug)`, `display_gloss`, `category`, `static_or_movement`, `flippable`, `difficulty_rank`, `is_active_for_practice`, `reference_video_url`, `pre_attempt_hint`, `generic_failure_hint`, `parameters jsonb`. Currently **80 active sem-lex signs** (`20260524000000`), re-ranked by learning difficulty (`20260525000000`); 10 signs in category `pending` (no avatar mocap).
- **attempts** — append-only metadata log: `passed`, `confidence`, `predicted_class_id`, `hint_shown`, `learner_disagreed`, `mediapipe_detection_failed`, `model_version_id`. **No frames/keypoints stored** (privacy).
- **mastery_state** — SRS state per (user, sign): `status`, `ease (def 1.3)`, `interval_days`, `next_review_at`, `consecutive_passes`, `total_attempts/passes`.
- **model_versions** — `id`, `artifact_url`, `config_url`, `is_active` (partial-unique: ≤1 active). **All rows currently inactive** (`20260520200000`); the live recognizer bypasses this and loads from `/public/models`.
- **confusion_pair_hints** — 120 seeded `(target, predicted) → hint` rows (Layer-B pedagogy).

---

## 5. ML system — deep dive

### 5.1 Models (all from-scratch; params **computed**, not estimated)

**(A) Hand detector — `hand_detector_v0.onnx`** (`training/detectors/hand_detector.py:50-149`)
- CenterNet-style anchor-free single-stage CNN. Stem (320→160) + 3 residual stages (32→64→128→192, stride-2) down to 20², FPN-style transpose-conv upsample + lateral skips back to 80², two heads.
- **Input** `(1,3,320,320)` RGB[0,1]; **outputs** `heatmap(1,1,80,80)` logits + `size(1,2,80,80)`, **stride 4**.
- **Params: 2,300,707 (~2.3M)**. Heatmap final-conv bias init −2.19 (focal stability).
- **Weights:** from scratch; trained on hand bboxes from FreiHAND + CMU HandDB + COCO-WholeBody + HaGRID (HF mirror) + face-negatives (WFLW). Model card val loss **1.743**; **AP@IoU=0.5 never computed** (target ≥ 0.85).

**(B) Hand-landmark regressor — `hand_landmarks_v0.onnx`** (`hand_landmarks.py:50-130`)
- Small ResNet (stem + 4 residual stages 64→128→192→256) + GAP + MLP coord head; direct coordinate regression.
- **Input** `(1,3,224,224)`; **outputs** `coords(1,21,2)∈[0,1]` (+ visibility, + optional depth head when `predict_z=True` — **off in the exported 2D model**).
- **Params: 4,286,239 (~4.3M, 2D)**. Keypoint-agnostic (reused for 98-pt face landmarks and 8-pt pose).
- **Metrics (model card):** v0 = **12.82 px @ 224** (FreiHAND/CMU, green-screen). The retrained **v2_combined** (+COCO-WholeBody, 117,982 hands) = **10.84 px clean / 21.61 px in-the-wild**. ⚠️ **The browser ships v0** — see §6 train/serve skew.

**(C) Pose regressor — `pose_detector_v0.onnx`** (`pose_detector.py:21-70`)
- Same backbone family; **input `(1,3,256,256)` → `coords(1,8,2)`** (nose, neck, R/L shoulder, R/L elbow, R/L wrist). **Params: 4,260,920 (~4.3M).** Provides the body-relative reference frame for the `location` features. (Ablation: pose-only collapses; it's marginal — `TRAINING.md` §3.)

**(D) Face detector — `face_detector_v0.onnx`** (`face_detector.py:18` — literal `class FaceDetector(HandDetector): pass`)
- Identical 2.3M CenterNet arch, independently trained on **WIDER FACE**. **Framing/UI only — does NOT participate in recognition** (ADR 0011). **Not in the browser `public/models/` set** (only used in the Python live demo).

**(E) HandshapeEncoder — PARKED/killed** (`handshape_encoder.py:47-111`)
- Pre-activation ResNet → 512-D → MLP → **128-D L2-normed** embedding; NT-Xent contrastive on 182K hand crops. **Params: 11,319,712 (~11.3M).** Best val 2.77; linear-probe 7.18% (5.7× chance). **Killed**: appending its 128-D/hand made features 356-D and accuracy *collapsed* to 7.5% (`TRAINING.md` §1). Not shipped.

**(F) Sign classifier — `sign_classifier_v3.onnx` (the deployed recognizer)** (`sign_classifier.py:67-110`)
- **Dilated Temporal Convolutional Network (TCN):** `BlockLayerNorm` → `Conv1d(F→hidden, k=1)` → 5 residual `TemporalBlock`s (each 2× `Conv1d(k=3, dilation=2^i)` + GELU + GroupNorm(8) + Dropout 0.2 + residual) → global avg-pool → LayerNorm/Linear/GELU/Linear head. `nan_to_num` at forward entry.
- **Input** `(1,32,108)`; **output** `(1,80)` logits. **Params: 2,088,744 (~2.09M)** at the deployed `hidden=256, num_blocks=5` (note: the class docstring's "~1M / hidden=192" is stale — every training call hardcodes 256/5).
- **A transformer variant** (`TransformerSignClassifier`, ~0.98M / ~2.24M) was A/B'd (Phase 4.7) and **parked** — both ≈ the TCN (~11% at 100-D). "Temporal attention is not the bottleneck; feature representation is" (`STATUS.md`).

### 5.2 Feature extraction / preprocessing (the contract that matters)
The classifier never sees pixels — it sees a **108-D hand-relative feature vector per frame**, the v2 "`norm=hand`" schema (`sign_matcher.ts:163-222`, mirrors `fit_templates._frame_to_features_v2`). Per hand slot (46 dims): `[handshape 0:42 = (kp_i − kp0)/‖kp9−kp0‖]` (wrist-origin, hand-scaled) `[location 42:44 = (kp0 − bodyAnchor)/shoulderScale]` `[orientation 44:46 = unit(kp9−kp0)]`. Two slots (slot 0 if wrist.x < anchor, else slot 1) + 8 pose kpts ×2 = **108**. Body anchor = neck → shoulder-mid → nose fallback; scale = shoulder pixel distance (floor 1.0). Missing → NaN, zeroed before the net.
- **This normalization was the single biggest win.** v1 body-relative norm (100-D) plateaued at ~12% top-1; the v2 hand-relative norm jumped sem_lex signer-disjoint to **70.9%** "for free" (`STATUS.md`, memory `normalization_was_the_bottleneck`). Handshape was a sub-pixel wiggle under body-norm.
- Trajectory: drop handless frames → linear NaN-aware resample to **T=32** (`resampleTrajectory`).

### 5.3 Training pipeline
- **Source data (recognition):** ~12.7K ASL clips (`v4` manifest: sem_lex 7,657 + asl_citizen 2,979 + wlasl 772 + msasl 450 + lifeprint 235 + ytsearch 560 + project_clean 99); current training vocab `dataset/sem_lex_top80_vocabulary.json` (80 signs, on-disk floor 99 clips/sign, median 140, 25+ signers). **Detector/landmark labels** come from provenance-audited public datasets (FreiHAND, CMU HandDB, COCO-WholeBody, MPII, WIDER FACE, Multiview Hand, WFLW; ADR 0015) — human/sensor/multi-view-fit labels, **never MediaPipe/OpenPose-generated**.
- **Trajectory extraction** (`extract_trajectories_v2.py`): batched detector→landmark→pose over every clip, idle-frame trim, quality filter (drops degenerate bboxes; raised 2nd-hand threshold 0.15, max aspect 3.5 → 0% degenerate in v7 vs 48% in v6). v3 trajectories re-extracted with the v2_combined landmarker.
- **Classifier training** (`train_classifier.py`): `AdamW(lr=1e-3, wd=1e-4)`, `CosineAnnealingLR`, **CrossEntropy with label_smoothing=0.05** (class-weighting *hurt* val), batch 256, 80 epochs, early-stop patience 8, bf16 autocast, grad-clip 5.0. Aug: global scale [0.9,1.1], xy jitter σ=5px, horizontal mirror p=0.5 (slot-swap), time-warp ±15%. Splits are **signer-disjoint** (join trajectory→manifest `signer_id`). Trained on Modal **L4**.
- **Detectors:** `AdamW`, warmup→cosine, focal + 0.5·GIoU size loss (`losses.py`); losses forced to **fp32** to dodge a bf16 `log(0)→NaN` trap. Hardware: hand detector **A100/H100**, landmarks 2D **L4** / 3D & packed **A100**, pose **H100**, face **L4**. Speed win: uint8 H→D transfer + pin/persistent/prefetch dropped landmark epochs 159s→59s (2.7×).
- **ONNX export** (`export_onnx.py`): opset 17, constant-folded, dynamic batch, **fp32, no quantization**; parity-checked vs onnxruntime (atol 5e-4 / classifier 5e-6).
- **Modal:** app `asl-mastery-training`, volume `asl-mastery-data`, ~50 entrypoints across L4/A100/H100/A10G. Cumulative spend ~$50.

### 5.4 Inference pipeline (serving)
- **Where:** 100% in the **browser** (`onnxruntime-web`, WebGPU→WASM). No server/edge/GPU inference.
- **Batching:** none — one frame at a time, ~16 frames × up to 4 ONNX runs.
- **Quantization:** none on live models (fp32).
- **Latency:** capture 2 s + ~1–3 s analyze (untested in-browser).
- **Throughput:** single-user, single-attempt; no concurrency model (inference is on the client).

### 5.5 Post-processing / decoding
Temperature softmax (T=1.0 in config) → rank → **pass if target in top-4**; `perSignThresholds` is `{}` (empty) and `DEFAULT_THRESHOLD=0.3` is essentially display-only. No-hands-anywhere → graceful "try again" (`passed:false`).

### 5.6 Evaluation
- **Headline:** **75.8% top-1 / 92.7% top-5** on `sem_lex_top80`, signer-disjoint (11,307 train / 1,856 val), v3 classifier on v2_combined-extracted trajectories (`TRAINING.md` WEB SHIP; memory `normalization_was_the_bottleneck`).
- **Eval gate** (`docs/EVAL_GATE.md`): top-1 ≥ 85% floor; no sign < 60%; per-Fitzpatrick gap ≤ 10pp; pass-decision precision ≥ 90%; p95 ≤ 600 ms; ECE ≤ 0.05; no-pretrained evidence intact. The 85% floor is **not met** and is **acknowledged unreachable on the full 80** (~83% asymptote — `memory/project_path_to_85_percent`).
- **Benchmark comparison:** the archived **MediaPipe + BiLSTM** pipeline hit **67% top-1 / 85% top-3** on the same vocab (`runs/v2-007/validation.json`) — the from-scratch v3 now **beats** that proof-point. Detector/landmark cards: hand det val loss 1.743 (AP uncomputed), landmark 12.82 px (v0) / 10.84–21.61 px (v2).

---

## 6. Known ML problems and limitations (brutally honest)

1. **⚠️ Train/serve skew on the landmark model.** The classifier was trained on trajectories re-extracted with **hand_landmarks_v2_combined** (10.84 px clean / 21.61 px in-the-wild), but the browser ships `public/models/hand_landmarks_v0.onnx`, which is **byte-identical (md5 `951cac41…`) to the May-21 v0 export (12.82 px, FreiHAND/CMU only)**. So in-browser features come from a *different, worse, more OOD* landmarker than the one the 75.8% was measured on. **Tried:** v2_combined was trained and used for extraction; the browser ONNX was apparently never re-exported. **Net:** real in-browser accuracy is likely below 75.8% and is **untested in-browser** (`TRAINING.md` WEB SHIP: "untested in-browser… expect 2-3 iteration rounds"). *This is the #1 thing to fix and the most likely question.*
2. **Detector undercalibration.** Max heatmap prob ~0.27 on real ASL frames → extraction runs at threshold **0.02** (`extract_trajectories_v2.py:82-86`); ~48% of an earlier trajectory set had degenerate 1-px boxes. **Tried:** raised thresholds + aspect/size quality filter + IoU dedup + hand-tracking → 0% degenerate in v7. A from-scratch **detector retrain** on combined hand-boxes is *built but parked* (user declined).
3. **In-the-wild landmark error 21.61 px.** Green-screen-trained landmarks are OOD on real hands (bunched dots, palm-forward, close-ups, two-hand). **Tried:** v2_combined (+COCO-WholeBody) closed clean error to 10.84 px and added an in-the-wild number; scale aug widened to 0.65–1.40 for close-ups.
4. **Two-hand contact signs underperform ~9 pts** (0.661 vs 0.753). **Diagnosed** as mostly **sign similarity / homonyms** (NICE = CLEAN are the *same* sign in ASL), *not* two-hand tracking (`memory/per_sign_v2_contact_vs_similarity`). Lever = handshape discrimination + vocab curation, not tracking.
5. **85% top-1 is unreachable on the full 80-sign vocab.** The 64 "easy" signs already sit at 0.83 and dominate the mean; perfectly fixing the worst 16 only reaches 82.8%; clip-count↔accuracy r=+0.15 (data volume isn't the lever). **Path:** explicit handshape + inter-hand **contact** geometry features (#1 lever, $0) + augmentation + conservative curation → ~85% on a *curated* ~76-sign vocab. **Not yet built.**
6. **Pose anchor uses the full frame in the browser**, but the Python demo crops pose to a face/upper-body box (`keypoints.ts:13-17,215-229`). This can shift the body-relative `location` features the classifier relies on. **Untested in-browser** (explicit TODO at top of `keypoints.ts`).
7. **Left-handed mirroring is unhandled in the browser path.** Capture mirrors left-handed input to the training convention (`camera-capture.tsx:179-184`), but `TRAINING.md` flags "left-handed mirroring unhandled" as a post-deploy verify item — the feature-level mirror in `frameToFeaturesV2` is not applied at predict time.
8. **Encoder / handshape embedding killed.** 356-D (kpts + 128-D embed) collapsed to 7.5% (vs 100-D's 11.1%). Single-frame handshape info is real but insufficient (linear probe 7.18%). Parked.
9. **Z/3D depth shelved.** 3D landmark v0/v1 trained (13.82/13.98 px, depth err 0.0997/0.0950), but `zcheck_corpus` showed depth on real corpus clips is **noise-dominated** (jitter > signal) → would hurt the classifier. Shelved as a precision upgrade, not deployed.
10. **Forgiving pass = false-positive risk.** Top-4 acceptance (`keypoint-predict.ts:132-134`) means a learner can "pass" while in 4th place. **Conscious pedagogical choice** (practice app, not exam; an 80-class softmax puts correct answers at 0.3–0.5), but it weakens recognition rigor.
11. **Cold start / memory.** Four ONNX models (~51 MB total) load on mount; preloaded to avoid first-attempt stall. No measured memory ceiling; WebGPU unavailability silently falls back to WASM (slower).
12. **Silent fallbacks.** Any inference exception → `stubPredict` (deterministic ~80% cosmetic pass, `runner.tsx:85-88`) — a *real* miss could be masked by the stub if the pipeline throws. No-hands-anywhere returns a clean fail.
13. **Open production bug (not ML):** `/practice` crashes intermittently with a **server-side** error (digest `3063673210`, `TRAINING.md` top block) — suspected `nextSignId`/`getItemById` on a locked/`pending`/missing sign after the recent vocab+rerank migrations. Diagnosis steps documented; fix not yet landed.

---

## 7. Tradeoffs we've consciously made

| Tradeoff | What we chose | What we gave up | Why |
|---|---|---|---|
| **From-scratch CV vs accuracy/speed** | Train **every** recognition weight in-house (ADR 0012) | MediaPipe's instant 21-pt landmarks + ~67/85 baseline; months of timeline | The from-scratch story **is the deliverable/moat**; ADR 0012 is the "don't `npm install @mediapipe/hands` on the bad days" guard |
| **On-device vs server inference** | 100% browser ONNX | batching, bigger models, server GPU throughput | Privacy (frames never leave device, `docs/PRIVACY.md`) + R2 zero-egress + no inference cost |
| **Landmark pipeline vs end-to-end 3D-CNN** | 4 small detectors + TCN over trajectories (ADR 0011) | one simple model | Sample efficiency: 13–90 clips/sign can't train an RGB 3D-CNN (honest 30–50%); landmarks reuse per-frame labels + absorb pixel variance |
| **Learned TCN vs Mahalanobis templates** | TCN classifier head | template interpretability | Templates hit an architectural ceiling (P=19.7/R=28.5%); TCN substantially better. Template matcher kept as fallback (`StreamingMatcher`) |
| **fp32 vs INT8 in browser** | fp32 for the 4 live models | ~4× smaller/faster bundle | Parity simplicity; localhost-class budget (~20 MB target). INT8 exists only in the dead 3D-CNN path |
| **Server actions vs REST API** | Next.js server actions (monolith) | a separate API surface/service boundary | One Next app on Vercel; no microservices |
| **Forgiving top-4 vs strict top-1 pass** | top-4 acceptance | recognition rigor / false-positive resistance | Pedagogy: an 80-way softmax + ~93% top-5 means strict top-1 flickers on correct attempts |
| **Avatar: buy mocap vs build motion** | 3D-LEX optical mocap GLBs (non-commercial license) | full ownership; commercial licensing certainty | Avatar is *reference content*, explicitly outside the strict-CV perimeter → can use clean external 3D motion ($0 GPU). Own SMPL-X extraction was the fallback |
| **Z/3D depth: ship vs shelve** | 2D only | per-finger depth, hardest curl signs | Depth on real corpus is noise-dominated → would *hurt* accuracy |
| **Cost vs performance** | L4/A100 spot-style Modal, detached runs, ~$50 total | dedicated GPUs | "Confirm cost before any Modal GPU launch" is a hard user rule (`memory/feedback_modal_cost_confirm`) |

---

## 8. What's NOT in this version (deferred / future work)

- **`model_versions` promotion** — all rows deactivated (`20260520200000`); live models bypass the DB and load from `/public`. `20260521000000_promote_v3_0_0.sql.template` awaits an eval-gate pass + ONNX SHA256.
- **Per-sign calibrated thresholds** — `perSignThresholds: {}` empty; Brief Requirement 9 artifact not yet generated for the 80-sign vocab.
- **Face detector / face landmarks in the browser** — trained (98-pt WFLW, 16.76 px) but **not wired** into the browser recognition path; gated on a confusion analysis of face-located signs.
- **Path-to-85% features** — explicit handshape curl/spread + inter-hand contact-distance features (the #1 accuracy lever) **not built**.
- **Handshape encoder + Z/3D depth** — built, **killed/shelved**.
- **Transformer classifier head + from-scratch detector retrain** — built, **A/B'd, parked**.
- **Phase 5 pedagogy** — only Layer-A (pre-attempt) + a generic Layer-C failure hint are live; **Layer-B confusion-pair hints don't fire** (no active model row historically; 120 hints seeded but a top-80 confusion matrix isn't authored), and **parameter-aware (Stokoe) hints** are unbuilt.
- **Avatar:** 10 of 80 signs have **no mocap** (render at rest); face-contact signs clip fingers slightly into the face; only ~5/70 spot-QA'd. Fingers driven by mocap now, but the older keypoint/SMPL-X finger paths are dead ends.
- **Reference videos for the new top-80** — `20260524010000_seed_reference_videos_top80.sql.draft` is **not applied**; some of the 55 new signs may lack a reference video.
- **Recording tool** (admin contributor capture), **TTS** (Web Speech API, declared ADR-0012 exception), **bias/Fitzpatrick eval** (Phase 7), **AP@IoU detector metrics** — all unbuilt.
- **Observability SDKs** — PostHog/Sentry are in env/docs but not in `package.json`; likely not yet wired.
- **The entire ADR-0010 3D-CNN path** (`training/classifier/*`, `lib/inference/classifier.ts`) is **dead code** kept for history (the live path is `keypoint-predict.ts`).

---

## 9. The 60-second pitch

> "Mastered teaches ASL the way a good tutor would — show the sign, watch you do it, and bring it back exactly when you're about to forget it. The interesting constraint we set ourselves: **no pretrained vision, anywhere.** Everyone reaches for MediaPipe; we trained every recognition weight from scratch — our own hand detector, hand-landmark regressor, and pose model, all small CNNs — on public datasets we provenance-audited. Those feed a from-scratch temporal CNN over 21-point hand trajectories, exported to ONNX and running **entirely in the browser** — webcam frames never leave your device. The whole stack is a single Next.js app on Vercel with Supabase for auth and a spaced-repetition scheduler in Postgres, and Cloudflare R2 for model files, so per-learner cost is basically zero. The thing that unlocked us wasn't a bigger model — it was a feature-normalization fix: making hand keypoints **hand-relative instead of body-relative** took us from 12% to ~76% top-1, 93% top-5 on 80 signs, signer-disjoint. That already beats the MediaPipe baseline we benchmark against. We're honest that strict top-1 of 85% isn't reachable on this vocabulary without explicit handshape and finger-contact features — that's exactly what's next, along with re-exporting the improved landmark model that's currently lagging the one we trained on."

---

## 10. Anticipated questions (10 hardest, with tight answers)

1. **"Why not just use MediaPipe? It's free, accurate, and instant."**
   Deliberate constraint (ADR 0012). The from-scratch story is the deliverable, not the recognizer per se. We benchmark *against* a MediaPipe+BiLSTM pipeline that hit 67/85 — our from-scratch v3 now beats it at 75.8/92.7. MediaPipe is forbidden in shipped code and CI-grep-verified.

2. **"What's your p99 latency?"**
   We don't have measured p50/p95/p99 yet — honest gap. The dominant latency is the fixed 2 s capture + ~2 s countdown by design; recognition adds ~1–3 s of in-browser ONNX (16 frames × ≤4 models), untested in-browser. The eval-gate *target* is ≤600 ms for the recognition step. Measuring this is on the list.

3. **"What happens when the model is wrong?"**
   Two cases. Soft: pass is top-4, so near-misses still pass (pedagogical, ~93% top-5). The learner can flag a false negative ("I think I did this right" → `learner_disagreed`), which we mine for weak signs. Hard: any inference exception falls back to a deterministic stub (~80% cosmetic pass) — which is a known risk: a thrown pipeline could mask a real miss.

4. **"How does this scale?"**
   Inference is on the client, so recognition scales for free — it's the user's GPU. The server only does auth + ~2 small DB writes per attempt (RLS-scoped). The bottleneck is Postgres write throughput and Vercel function concurrency, both generous for a pilot. Training is offline on Modal and doesn't touch the request path.

5. **"Is your 75.8% real, or leaked?"**
   Signer-disjoint split (clips joined to manifest `signer_id`), 11,307 train / 1,856 val — a signer in val never appears in train. We confirmed clean ≈ leaky, so leakage wasn't the inflator. **Caveat I'll volunteer:** that number was measured with the *v2_combined* landmarker; the browser currently ships the older *v0* landmarker (byte-identical to the May-21 export), so live accuracy is likely lower until we re-export — that's my top fix.

6. **"Why a TCN and not a transformer or LSTM?"**
   We A/B'd a transformer (Phase 4.7): both ~11% at 100-D, ≈ the TCN. Conclusion: temporal attention isn't the bottleneck — **feature representation is**. The TCN is ~2M params, parallelizable, and avoided the Mahalanobis matcher's magnet pathology. We kept the template matcher as a fallback.

7. **"What actually moved the needle from 12% to 76%?"**
   Not the model — normalization. Hand keypoints were normalized body-relative (neck anchor, shoulder scale), so handshape was a sub-pixel wiggle the classifier couldn't see. Switching to **hand-relative** features (wrist-origin, ‖kp9−kp0‖ scale) + explicit palm orientation + body-relative *location* gave +59 points essentially for free.

8. **"Where does your training data come from, and is it clean?"**
   Recognition trajectories: ~12.7K public ASL clips (Sem-Lex, ASL-Citizen, WLASL, MS-ASL, Lifeprint). Detector/landmark *labels*: FreiHAND, CMU HandDB, COCO-WholeBody, MPII, WIDER FACE, WFLW — all human/sensor/multi-view-fit annotations, **never MediaPipe/OpenPose-generated** (ADR 0015 acceptance criteria, per-dataset PROVENANCE.md, SHA256s).

9. **"Why can't you hit the 85% you set as the gate?"**
   Arithmetic: 64 of 80 signs already sit at 0.83 and dominate the mean; perfectly fixing the worst 16 tops out at 82.8%, and clip count barely correlates with accuracy (r=+0.15). 85% needs lifting the whole field via features — explicit handshape geometry and inter-hand contact distance — on a slightly curated vocab (merge true homonyms like NICE/CLEAN). We document the limitation rather than relax the constraint to escape it.

10. **"Privacy — you're running a camera; where do the frames go?"**
    Nowhere. Inference is 100% in-browser; the capture canvas is tagged `no-track-canvas` so PostHog can't grab it; only attempt *metadata* (sign id, predicted id, confidence, pass/fail, timestamps) is written to Postgres. No third-party vision vendor in the path (no MediaPipe CDN). Account deletion cascades via FK. This is stronger than the MediaPipe architecture would allow.

---

## Files / sources I wish I'd had (gaps to fill manually)

1. **A measured in-browser latency + accuracy report.** Everything past "the model exists" (the 75.8% was offline, on v2_combined trajectories). No `docs/validation/v3.md` for the keypoint pipeline exists; `runs/*` hold offline metrics only.
2. **The exact `sign_classifier_v3` training run record** (which trajectory dir, seed, epoch, and *which* landmarker its trajectories used) — inferred from `TRAINING.md`/memory, not from a committed run config in the repo.
3. **Provenance of `public/models/hand_landmarks_v0.onnx`** — confirmed md5-identical to the v0 export, but I couldn't find the script/commit that decided to ship v0 rather than re-export v2_combined. Confirm whether this is intentional or an oversight before presenting #5/#1.
4. **W&B run dashboards** — referenced (`wandb` in requirements) but not in-repo; the loss curves / per-Fitzpatrick breakdowns live there.
5. **Confirmation that PostHog/Sentry are actually wired** — they're in `.env.example` and docs but absent from `package.json`; the instrumentation code wasn't found.
6. **The `/practice` crash root cause** — the digest (`3063673210`) needs a dev-mode stack or `vercel logs` to localize; not resolvable from static reading.
7. **`docs/validation/v2.md` and `runs/v2-007/validation.json`** numbers (the 67/85 MediaPipe baseline and the 67.07/84.94 v2.0.0 RGB number) — cited across docs but I read them secondhand via the model cards/README, not the JSON itself.
8. **SignAvatars / 3D-LEX license terms** — non-commercial; confirm acceptability before any commercial framing of the avatar.
