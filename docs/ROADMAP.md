# Roadmap

> What we are building and in what order. This is the document we return
> to whenever the question "what do we work on next?" comes up.

The roadmap is organized in phases rather than days, because the user
has stated the goal is a production-grade system at scale, not a
two-day sprint. Each phase has an exit criterion. A phase is not done
until its exit criterion is met.

---

## Phase 0 — Lock the unknowns (blocking)

**Goal:** Resolve every open question that would force a re-build if
answered wrong.

**Work:**

1. Send the director the three written questions:
   - Alphabet/fingerspelling inclusion.
   - Public ASL dataset usage (WLASL, MS-ASL, ASL-LEX, HandSpeak).
   - Classical CV libraries (already verbal, want it in writing).
   - Server-side training with browser-side inference.
2. Confirm demo audience (Patrick/Frank only vs. real learners) so we
   know whether to build full auth or guest mode.
3. Commit to the candidate vocabulary list drawn from Lifeprint
   Units 1–6 and cross-checked against ASL-LEX 2.0 (phonological
   coverage) and WLASL/MS-ASL (clip availability). Output:
   `docs/VOCABULARY.md`. Deaf-signer sign-off is a slice-2
   requirement (ADR 0004).

Instructor contracting is intentionally absent from Phase 0; ADR 0004
defers it to slice 2 (production deployment), with pilot sources drawn
from ASL-LEX 2.0, Lifeprint, WLASL, and MS-ASL.

**Exit criterion:** Written confirmations on dataset / classical-CV /
training-location items received (already received per CLAUDE.md §4);
vocabulary list committed to `docs/VOCABULARY.md`.

---

## Phase 1 — Documentation and decision records

**Goal:** Every non-trivial decision is written down with rationale
before any code is written that depends on it.

**Work:**

1. Complete `ARCHITECTURE.md`, `MODEL.md`, `DATASET.md`, `PEDAGOGY.md`,
   `EVAL_GATE.md`, `PRIVACY.md`.
2. Verify every citation in `PEDAGOGY.md`. Any claim that cannot be
   sourced gets removed or marked PENDING CITATION.
3. Build the research index in `docs/research/` with one file per
   primary source we rely on.
4. Architecture Decision Records (`docs/decisions/`) for: the
   recognition architecture choice, the no-placement-test decision, the
   mastery state model, the hint system layering, the signer-disjoint
   split, the eval-gate criteria.

**Exit criterion:** README references all docs and each doc is complete
or carries explicit PENDING markers we can audit.

---

## Phase 2 — Repo scaffolding and infra

**Goal:** A deployed app shell where future work can land without
infrastructure friction.

**Work:**

1. Next.js 14 + App Router + TypeScript + Tailwind + shadcn/ui project,
   pushed to GitHub.
2. Supabase project: auth (email magic link), Postgres schema for
   users, vocabulary_items, attempts, mastery, model_versions.
3. Vercel deployment from main; preview deployments per PR.
4. Cloudflare R2 (or S3) bucket for model artifacts and any
   consented-upload data; signed URLs only.
5. ESLint, Prettier, Husky pre-commit, GitHub Actions for typecheck
   and lint on every PR.
6. Sentry for error tracking; PostHog for product analytics (configured
   to never receive video data).
7. A minimal "hello" deployed page that proves the whole pipeline works
   end to end.

**Exit criterion:** A reviewer can clone the repo, run `pnpm install &&
pnpm dev`, and hit a working page; the same code is live on a Vercel URL.

---

## Phase 3 — Recording tool and dataset assembly

**Goal:** Have enough training data to start training meaningful
models.

**Work:**

1. **Recording tool** (admin-gated route in the same Next.js app, not a
   separate app). Captures 2-second clips at 720p+ with the green-box
   framing the learner app will use. Tags every clip with metadata
   (signer id, sign id, timestamp, lighting, background, handedness,
   sleeve length, optional Fitzpatrick scale with consent, signer's
   self-rated correctness). Captures a 1-second empty-frame clip at
   session start for MOG2 background-subtraction augmentation.
2. **Public dataset ingestion.** Assuming Phase 0 approves, download
   WLASL and MS-ASL clip metadata; filter to our vocabulary; pull only
   the clips we need; store in R2 with provenance metadata; record
   license terms per dataset in `docs/DATASET.md`.
3. **Canonical reference selection from public datasets.** Per ADR
   0004, select the best per-sign reference clip from WLASL/MS-ASL,
   record attribution, mirror to R2, and use as the learner-facing
   reference video plus anchor positive in training. Slice-2 candidate:
   paid 3–4 hour session with a Deaf ASL instructor re-recording all
   75–100 signs in our standardized green-box framing to replace the
   public-dataset references.
4. **Self-recorded supplement.** Team members and friends record 30 min
   each, distributed across signs. Goal: 100+ clips per sign minimum
   across all sources, with diversity targets for skin tone, lighting,
   background, and clothing tracked in a coverage spreadsheet.
5. **Cleaning pipeline.** Python scripts that read raw clips, trim to
   the sign window, normalize framing (crop to green box), normalize
   frame rate to 30 fps, normalize length to the model's 16-frame
   window, compute a perceptual hash for dedup, and write the cleaned
   clips to a versioned dataset directory in R2 with a manifest.
6. **Signer-disjoint split assignment.** Each signer gets assigned to
   train, validation, or test once and forever; the split is committed
   as a JSON manifest in the repo.

**Exit criterion:** A versioned dataset (v1) exists with ≥100 clips per
sign on average, diverse signer demographics tracked in metadata, and a
documented cleaning pipeline that is reproducible from a single
command.

---

## Phase 4 — Model training and validation

**Goal:** A model that passes our eval gate.

**Work:**

1. **Training pipeline** in PyTorch, with Weights & Biases experiment
   tracking. Includes the R(2+1)D-small architecture, AdamW + cosine
   annealing, label smoothing, weighted class sampling, and the full
   augmentation stack (spatial, temporal, color, classical-CV-based
   background swap).
2. **Validation harness** that runs against the held-out test split,
   produces per-sign accuracy, per-condition accuracy, per-demographic
   accuracy (where consented data allows), full confusion matrix, and
   reliability diagram for calibration.
3. **Temperature scaling** post-training; per-sign confidence
   thresholds derived from validation (≥90% precision target).
4. **ONNX export and int8 quantization**; verify quantized model
   matches float32 predictions on a sample within tolerance.
5. **Confusion-pair extraction.** From validation confusion matrix,
   pull the top 2–3 confusions per sign; these drive the hint system.
6. **Iteration loop.** Train v0 on whatever data is available;
   identify worst-performing signs; collect more data for those signs;
   retrain v1; repeat until eval gate passes.

**Exit criterion:** A model artifact (v1 or later) that meets
`EVAL_GATE.md` criteria, with a frozen validation report committed to
the repo.

---

## Phase 5 — Learner app

**Goal:** A learner can sign in, practice, get pass/fail with targeted
hints, see mastery progress, and have progress saved across sessions.

**Work:**

1. **Auth flow.** Sign up, log in, sign out. Email magic links. Capture
   handedness on first session (one tap, with a "I'll set this later"
   escape).
2. **Onboarding.** No placement test. A 60-second walkthrough screen
   that explains how the system works: prompts, recording, pass/fail,
   hints, mastery progression, the "you can leave when you're done"
   principle. Camera/mic permission handled here, not mid-practice.
3. **Practice screen.** Prompt + reference video loop + pre-attempt
   parameter card. Green-box camera preview (mirrored selfie view).
   Countdown, 2-second capture, "analyzing" state, result with hint or
   passage to next item.
4. **Scheduler.** Modified SM-2 algorithm picks the next item from the
   learner's mastery state. New items introduced when rolling pass rate
   is high enough; struggling items prioritized.
5. **Mastery dashboard.** Signs mastered, signs in active learning,
   signs in spaced review, signs not yet started. The mastery decay
   visualization (a sparkline per sign showing the predicted retention
   curve and the next scheduled review).
6. **Self-paced exit screen.** When a sign hits mastered, a celebratory
   screen, the sign is removed from active rotation, and the learner is
   shown the long-interval review schedule.
7. **Session-end summary.** Total time spent, signs touched, mastery
   transitions during the session.
8. **Settings.** Handedness, optional demographic disclosures with
   clear consent language, camera device selection, sign-out, account
   delete.
9. **Error and edge-case handling.** Camera denied, no camera detected,
   model load failure, network loss mid-session, attempt timeout.

**Exit criterion:** A reviewer can sign up, complete a practice session,
log out, log back in, and see preserved progress. All five sign
parameters of the practice flow (prompt, capture, evaluate, feedback,
state update) work end to end with the trained model running in the
browser.

---

## Phase 6 — Polish, branding, accessibility

**Goal:** The app looks and feels like it belongs in Superbuilders'
product portfolio.

**Work:**

1. Pull brand colors and typography from Alpha School and Superbuilders
   marketing surfaces. Apply through Tailwind theme tokens.
2. Microcopy review across the entire app. Pedagogically aware
   language: "mastered," "in review," "ready to retire" rather than
   "completed," "locked," "next level." No engagement-app vocabulary.
3. Accessibility pass: keyboard navigation, focus management, screen
   reader labels, ARIA roles, color contrast WCAG AA minimum (AAA
   where reasonable). Captions on the reference videos — non-negotiable
   given the population.
4. Loading and error states across every async action.
5. Mobile responsiveness for the marketing/landing pages; the practice
   screen itself targets desktop/laptop because camera framing and
   processing favor it.
6. A small case-study landing page that walks Patrick through the
   project: the pedagogical theory, the architecture, the model, the
   results, the limitations, the slice-2 roadmap. Embeds a short
   walkthrough video.

**Exit criterion:** Tested in a browser the team did not develop in (per
hiring partner guide). Lighthouse accessibility score ≥95.

---

## Phase 7 — Eval gate, monitoring, observability

**Goal:** The system is operationally trustworthy.

**Work:**

1. Eval gate enforced in CI: any new model artifact PR must include the
   validation report, and the report must meet the criteria in
   `EVAL_GATE.md`, or the PR cannot merge.
2. Production monitoring dashboards: per-sign pass rate, per-user pass
   rate distribution, inference latency p50/p95/p99, model load time,
   hint efficacy per confusion pair.
3. Per-user fairness alerting: if any user's pass rate stays below 30%
   across 50+ attempts, surface for manual review (likely demographic
   mismatch in training data).
4. "I think I did this right" feedback button on every fail; clicks are
   logged against the attempt for later review. Gold data for
   identifying false negatives.

**Exit criterion:** Dashboards are live; alerts have been test-fired;
the feedback loop is closed from learner click to dataset gap
identification.

---

## Phase 8 — Presentation readiness

**Goal:** The user can present this project to Patrick and Frank with
no surprise questions.

**Work:**

1. README is complete and reads as a senior-engineer's writeup, not a
   project tutorial.
2. Recorded walkthrough video (3–5 minutes) covering: the pedagogical
   theory in one paragraph, the live demo, the validation report, the
   eval gate, the slice-2 roadmap.
3. Talking-point document covering every question we anticipate (see
   `claude/CLAUDE.md` and the presentation prep list in conversation
   history): why ASL, what's the theory, how do you know it works, why
   no-pretrained matters to you, what failure mode worries you most,
   what would you do with three more months, how does this scale to a
   billion kids.
4. **Promote the six `VERIFIED` (Tier 1) pedagogy citations in
   `docs/research/README.md` to `DEEP-VERIFIED` (Tier 2).** For each
   of Bloom 1984, Hattie & Timperley 2007 (+ Wisniewski et al. 2019),
   Karpicke & Roediger 2008, Cepeda et al. 2006, Shea & Morgan 1979,
   and Sweller 1988/2010, locate the published source, pull an exact
   quote with page or section reference, and commit a per-source file
   at `docs/research/<slug>.md`. Any citation that appears in the
   README, the recorded walkthrough, or the talking-point document
   must be `DEEP-VERIFIED` before the demo to Patrick.
5. Two mock interviews with Jon or Derek per the hiring partner guide.

**Exit criterion:** The user has presented this twice out loud to a real
listener and received feedback.

---

## What is intentionally not on this roadmap

- Slice-2 features (multi-head parameter-aware model, instructor
  dashboard, expanded vocabulary) are *talked about* but not built
  in this scope. They live in their own document and become a
  follow-up roadmap if the user proceeds.
- Multi-language sign support — explicitly out of scope per brief.
- Sentence/phrase recognition — explicitly out of scope per brief.
- Production-scale public deployment with thousands of users — out of
  scope per brief, but the architecture decisions made here do not
  preclude it.

---

## Working agreement between user and Claude

- The user drives sequencing. Claude proposes; the user decides what is
  next.
- Each session begins by reading `claude/CLAUDE.md` and
  `claude/SESSION_LOG.md`. Each session ends by appending to
  `claude/SESSION_LOG.md`.
- When ambiguity arises about an already-made decision, Claude reads
  the decision record in `docs/decisions/` before re-litigating it.
- "Production-grade" is the bar. Slices are scope tools, not quality
  excuses.
