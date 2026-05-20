# ASL Mastery

A mastery-based skill acquisition system, instrumented for measurable
learning outcomes, using ASL vocabulary as the controlled testbed.

Live pilot: **https://asl-mastery.vercel.app**

The brief calls it "an ASL learning app for college ASL 1 learners."
That is the surface of the project. The substance is a pedagogical
architecture that could generalize to any skill, for any school, for
any learner — measured by **time-to-mastery, not engagement**.

---

## What is here

### Substance under evaluation

- A spaced-retrieval scheduler grounded in published learning science
  (`docs/PEDAGOGY.md`, citations verified against primary literature).
- A mastery state machine with an explicit *exit*: a sign reaching
  the `mastered` state is actively removed from practice rotation.
  The opposite of streak mechanics.
- A three-layer hint architecture (`docs/ARCHITECTURE.md` §5) where
  every hint traces to either a measurable model signal or an
  authored per-sign source. No vibes-based feedback.
- An **eval gate** (`docs/EVAL_GATE.md`) that any model artifact
  must pass before promotion. The gate is enforced in the training
  pipeline, not aspirational.
- Local browser inference: video frames never leave the device.
  MediaPipe Holistic + a from-scratch BiLSTM classifier per ADR 0006.
- **Honest scope disclosure**: vocabulary curated by hearing engineers
  from public sources (ADR 0004), training data drawn from WLASL +
  ASL Citizen only (ADR 0008 amended by ADR 0009). The ASL Citizen
  inclusion at v2.x is logged with its MSR-LA non-commercial
  constraint; slice-2 commercial deployment requires a re-train per
  ADR 0009. Slice-2 commitments to a Deaf-instructor engagement are
  named explicitly. The current v2.0.0 model lifts top-1 from v1.0.1's
  17.86% to **67.07% top-1 / 84.94% top-3 on 75 signs**, below the 85%
  eval-gate floor — `docs/validation/v2.md` is the contract.

### What this is not

- Not engagement software. No streaks, no DAU/MAU mechanics, no
  push-notification re-engagement, no gamification beyond mastery
  progression.
- Not a placement-test app. The brief assumes true beginners; the
  scheduler handles fast learners through the ordinary loop
  (ADR 0002).
- Not multi-language sign support, not sentence/phrase recognition.
  Out of scope per brief.

---

## How the pieces fit

```
  Learner browser                                 Server (Next.js on Vercel)
  ─────────────────                               ───────────────────────────
  ┌──────────────────────┐                        ┌────────────────────────┐
  │ getUserMedia → video │                        │ Supabase Auth (Google, │
  │ Green-box framing    │                        │ magic link, anonymous) │
  │ 2-second capture     │                        └────────────────────────┘
  │ MediaPipe Holistic   │                        ┌────────────────────────┐
  │ ONNX Runtime Web     │   ── attempt metadata ─▶│ Postgres (Supabase)   │
  │ (classifier inference)│   (NOT frames or       │ users / vocabulary /  │
  │ Pass/fail + hint     │    keypoints)          │ attempts / mastery /  │
  └──────────────────────┘                        │ model_versions / hints│
            ▲                                     └────────────────────────┘
            │
            └─── ONNX artifact + config ─── Cloudflare R2
                                            asl-mastery-models/
```

Detailed architecture lives in `docs/ARCHITECTURE.md`.

---

## Repo layout

```
app/                       Next.js App Router routes
  page.tsx                 landing page
  sign-in/                 auth UI
  practice/                practice screen (camera + extractor + classifier)
  dashboard/               mastery dashboard with forgetting-curve sparklines
  settings/                handedness, Fitzpatrick, account delete
  auth/callback/           OAuth + magic-link exchange
components/                React UI
hooks/                     useLandmarkExtractor (browser MediaPipe wrapper)
lib/
  db/                      supabase-js client (server/browser/admin) + middleware
  auth/                    auth + profile server actions
  scheduler.ts             pure modified-SM-2 logic (tested)
  scheduler/               server actions + read-only queries
  inference/               ONNX Runtime Web wrapper + stub fallback
  mediapipe/               HolisticLandmarker loader + extractor
  keypoints.ts             single source of truth for the (T, K=150) tensor
middleware.ts              Supabase session-refresh middleware

training/                  Python pipeline (Phase 3d / 3f / 4)
  data/                    WLASL+MS-ASL ingestion + cleaning + filter
  classifier/              BiLSTM + Transformer, init, augment, train,
                           validate, export
  keypoints.py             Python mirror of lib/keypoints.ts
  requirements.txt         pinned deps (mediapipe 0.10.18 matches lib/mediapipe)

supabase/migrations/       declarative SQL (init, RLS, vocabulary seed)
docs/
  ARCHITECTURE.md          system architecture
  ROADMAP.md               phases + exit criteria
  PEDAGOGY.md              learning theory (verified citations)
  MODEL.md                 classifier spec + no-pretrained evidence
  DATASET.md               sources, cleaning, splits, fairness
  EVAL_GATE.md             promotion criteria
  PRIVACY.md               data-handling commitments
  VOCABULARY.md            the 96-sign list with full cross-reference
  decisions/               ADRs 0001–0008
  research/                per-source citation notes
claude/                    persistent project context for Claude Code
  CLAUDE.md
  SESSION_LOG.md
```

---

## Running locally

Prerequisites: `node@22`, `pnpm@10`, a Supabase project, and the
env-var names listed in `.env.example` populated in `.env.local`.

```sh
pnpm install
pnpm dev               # http://localhost:3000
pnpm test              # vitest
pnpm typecheck
pnpm lint
pnpm format:check
pnpm build
```

Migrations (link once per machine):

```sh
supabase link --project-ref <your-project-ref>
supabase db push
supabase gen types typescript --linked --schema public 2>/dev/null > lib/db/database.types.ts
```

Training pipeline (Python, Phase 3d/3f/4):

```sh
cd training/
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Phase 3d — ingest public datasets
python -m training.data.ingest_wlasl --wlasl-json /path/to/WLASL_v0.3.json --output dataset/raw/wlasl
python -m training.data.filter_vocabulary --manifests dataset/raw/wlasl_manifest.json --output dataset/slice1_vocabulary.json --floor 15

# Phase 3f — clean + extract MediaPipe keypoints
python -m training.data.clean --raw-manifest dataset/raw/wlasl_manifest.json --filter dataset/slice1_vocabulary.json --output dataset/clean/v1 --version v1

# Phase 4 — train + validate + export
python -m training.classifier.train --manifest dataset/clean/v1/dataset_v1_manifest.json --output runs/v1-001/
python -m training.classifier.validate --manifest dataset/clean/v1/dataset_v1_manifest.json --checkpoint runs/v1-001/best.pt --output runs/v1-001/
python -m training.classifier.export --checkpoint runs/v1-001/best.pt --validation runs/v1-001/validation.json --output artifacts/v1.0.0/
```

---

## Honest scope disclosure (read this before judging the model)

The pedagogical and architectural commitments above are the substance
of what this project demonstrates. The recognition model itself is
pilot-grade under documented controlled conditions, with three
explicit limitations:

1. **Vocabulary, hints, and reference videos were curated by hearing
   engineers** against public corpora (Lifeprint, ASL-LEX 2.0, WLASL,
   ASL Citizen). No Deaf instructor reviewed them for slice 1.
   Confusion-pair hints were authored mechanically from ASL-LEX 2.0
   phonological features.
   See [`docs/decisions/0004-public-sources-only.md`](docs/decisions/0004-public-sources-only.md).
2. **Training data was drawn from public datasets only.** No member
   of the project team is a fluent ASL signer; we declined to record
   our own clips because training on non-signer-authored data would
   teach the model wrong signs — worse than less data, it would be
   misleading data. See
   [`docs/decisions/0008-public-data-only-training.md`](docs/decisions/0008-public-data-only-training.md).
3. **The v2.x model includes ASL Citizen under MSR-LA license**
   (non-commercial research). The slice-1 pilot is non-commercial and
   fits within MSR-LA's research-purpose clause, but the current
   v2.x weights cannot be deployed commercially without a re-train.
   See [`docs/decisions/0009-asl-citizen-v2.md`](docs/decisions/0009-asl-citizen-v2.md).

**Current artifact: v2.0.0** at `artifacts/v2.0.0/` — 67.07% top-1 /
84.94% top-3 on 75 signs (lifted from v1.0.1's 17.86% top-1 via the
Phase 9 data + model improvements). **Below the 85% eval-gate floor;
not promotable under the gate's own criteria.** Documented and
recorded in [`docs/validation/v2.md`](docs/validation/v2.md). The
eval-gate enforcer's purpose is precisely this: refuse to let a
sub-floor model masquerade as a passing one. Slice-2 closes the
gap via instructor-recorded data + the recording tool framework in
ADR 0004 / 0008 / 0009.

Slice-2 production-deployment work addresses all three limitations:
paid Deaf-instructor review of every sign + hint, instructor-recorded
canonical reference videos, instructor-recorded training supplement
in our green-box framing, plus retraining the classifier on
license-clean data so it ships commercially. The recording tool's
specification (`docs/ARCHITECTURE.md` §2.2) is preserved as the
slice-2 framework target — built but not deployed in slice 1.

The validation report names every limitation explicitly. The
README and the demo walkthrough do not claim what the system
cannot defend.

---

## Decision records

Every non-trivial decision has an ADR. The ones load-bearing for
understanding the project:

- [`0001-recognition-architecture.md`](docs/decisions/0001-recognition-architecture.md) — original Path B (superseded; preserved as history)
- [`0002-no-placement-test.md`](docs/decisions/0002-no-placement-test.md) — why we skip placement tests
- [`0003-deployment-platform.md`](docs/decisions/0003-deployment-platform.md) — Vercel + Supabase + R2
- [`0004-public-sources-only.md`](docs/decisions/0004-public-sources-only.md) — no instructor for slice 1
- [`0005-classical-cv-allowed.md`](docs/decisions/0005-classical-cv-allowed.md) — classical CV scope
- [`0006-recognition-architecture-revised.md`](docs/decisions/0006-recognition-architecture-revised.md) — pivot to landmark-based recognition
- [`0007-auth-providers-and-demo.md`](docs/decisions/0007-auth-providers-and-demo.md) — Google + magic link + anonymous demo
- [`0008-public-data-only-training.md`](docs/decisions/0008-public-data-only-training.md) — public-data-only training for slice 1
- [`0009-asl-citizen-v2.md`](docs/decisions/0009-asl-citizen-v2.md) — adding ASL Citizen (MSR-LA) at v2.x; slice-2 commercial re-train cliff named

---

## Built for

[Superbuilders](https://superbuilders.school) — Patrick Skinner (GM)
and Frank Yang (tech lead), evaluating against mission alignment,
agency, and engineering skill, in that order.

Mission alignment is non-negotiable and ranked above engineering
skill. Every architectural choice in this repo has both a software
justification and a pedagogical one (the two-theories principle in
`claude/CLAUDE.md` §3).
