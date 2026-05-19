# Session Log

Append-only log of decisions, threads, and context. Written at the end
of each working session so future-Claude can reconstruct where we left
off.

---

## Session 1 — initial scoping

**Decisions locked:**

- Project framed as "mastery-based skill acquisition system using ASL
  as testbed," not "ASL learning app."
- Path B (end-to-end small 3D CNN) is the model approach.
- Stack: Next.js + TypeScript + Tailwind frontend; ONNX Runtime Web for
  inference; Supabase for auth and progress; Vercel for hosting; R2 for
  model artifacts; PyTorch for training.
- Selfie-mirror preview, un-mirror before model input.
- Left-handed signers flagged at signup; flip frame at inference.
- Green box framing solves the multi-signer / pet / roommate edge cases.
- Classical CV libraries are usable for augmentation and quality checks
  (per user's director, pending written confirmation).
- Placement / fluency test REMOVED — anti-pedagogical for true
  beginners; scheduler handles fast learners via natural mastery
  progression.
- Two slice-1 thoughtful extras: mastery decay visualization, self-paced
  exit on mastery reached.

**Open threads at end of session:**

- Need written confirmation on alphabet/fingerspelling inclusion.
- Need written confirmation on public ASL dataset usage.
- Need written confirmation on classical CV libraries.
- Need to know if demo audience includes real learners.
- Need to commit to the candidate vocabulary list (waiting on item 1).

**Files created this session:**

- `claude/CLAUDE.md`
- `claude/SESSION_LOG.md` (this file)
- `docs/ROADMAP.md`
- `docs/ARCHITECTURE.md`
- `docs/PEDAGOGY.md` (skeleton, citations pending verification)
- `docs/MODEL.md`
- `docs/DATASET.md`
- `docs/EVAL_GATE.md`
- `docs/PRIVACY.md`
- `docs/research/README.md`
- `docs/decisions/0001-recognition-architecture.md`
- `docs/decisions/0002-no-placement-test.md`

## Session 2 — citation verification, deployment decision, constraints locked

**Decisions locked (now committed in CLAUDE.md):**

- **Alphabet/fingerspelling EXCLUDED.** Vocabulary must be life-usable
  content signs only. User confirmed.
- **Public ASL datasets ALLOWED** (WLASL, MS-ASL, ASL-LEX). User confirmed.
- **Classical CV libraries ALLOWED.** User confirmed.
- **Training is server-side, inference is browser-side.** Defended in
  ADR 0003 (deployment) and noted in CLAUDE.md decisions table.
- **Deployment platform: Vercel + Supabase + Cloudflare R2.** ADR 0003.
- **Demo audience: video walkthrough + live deployed app.** User confirmed.

**Citation verification pass completed.**

Six of the eight foundational pedagogy citations have been verified
against primary literature via web search:

- Bloom (1984) — VERIFIED with nuance captured (mastery learning alone
  ≈ 1 sigma; full 2-sigma requires tutoring + mastery).
- Hattie & Timperley (2007) — VERIFIED; 2019 replication finding that
  motor skill outcomes show larger feedback effects is added (directly
  load-bearing for our ASL application).
- Karpicke & Roediger (2008) — VERIFIED; the foreign vocabulary
  context of the original study makes it especially relevant to us.
- Cepeda et al. (2006) — VERIFIED; specific finding that optimal
  inter-study interval depends on retention interval is now in the
  doc and informs the scheduler design.
- Shea & Morgan (1979) — VERIFIED; load-bearing for the within-session
  interleaving choice.
- Sweller (1988, 2010) — VERIFIED; the 1988/2010 split is now correct
  in the doc.

Two remain unverified:

- Ebbinghaus + Murre & Dros (2015) — marked NOTED, treated as
  historical reference rather than load-bearing.
- Anderson Corbett Koedinger Pelletier (1995) — marked PENDING, treated
  as supporting context not load-bearing. Verify if used in README.

**Files created or updated this session:**

- `docs/PEDAGOGY.md` — citations updated to VERIFIED with nuances captured
- `docs/decisions/0003-deployment-platform.md` — new ADR
- `claude/CLAUDE.md` — decisions table updated, open questions trimmed
- `claude/SESSION_LOG.md` — this entry

**Open threads at end of session:**

- ASL instructor / Deaf community contact — recruit locally in Austin
  unless partner has a preference.
- Accessibility expectations — default to WCAG AA + captions.
- Vocabulary list sign-off — pending instructor onboarding.
- Two minor citation verifications remaining (above).

**Where to start next session:**

Phase 0 still has work: recruit the ASL instructor; draft the
candidate 75–100 sign list for instructor review. Then Phase 2
(repo scaffolding) can begin. Phase 1 (docs) is essentially complete.

## Session 3 — no-instructor decision, repo setup, vocabulary draft (2026-05-19)

**Decisions locked:**

- **No Deaf ASL instructor for the pilot.** Vocabulary, linguistic
  metadata, hint copy, and reference video are sourced from public
  corpora only (ASL-LEX 2.0, Lifeprint, WLASL, MS-ASL). Instructor
  engagement is a slice-2 production-deployment requirement, not
  pilot work. Captured in `docs/decisions/0004-public-sources-only.md`
  with mitigation, what-we-lose, and the slice-2 plan.
- **Repo layout normalized** into the `claude/` + `docs/` structure
  the documents themselves prescribe. Files had been flat at the
  repo root; moved before initial commit.
- **GitHub repo created:** https://github.com/Hirom0112/asl-mastery
  (private). Per the auto-memory rule, also mirrored to the
  self-hosted GitLab at https://labs.gauntletai.com/hiromalarcon/asl-mastery
  with default `main` branch protection dropped.
- **TODO.md scratchpad** at repo root, gitignored. The ROADMAP is
  the strategic map; TODO.md is the in-flight session-level
  working state; SESSION_LOG.md is the committed decision log.
- **Vocabulary scope confirmed** as Lifeprint ASL 1 only (Units
  1–3 / Lessons 1–15). The user's brief said "Units 1–6," which
  spans into Lifeprint's ASL 2; resolved 2026-05-19 to align with
  `claude/CLAUDE.md` §2's "college ASL 1" framing.
- **`docs/VOCABULARY.md` drafted with 96 signs.** Sourced and
  cross-referenced as follows:
  - Lifeprint vocabulary lists fetched directly from
    `lifeprint.com/asl101/lessons/lessonNN.htm` for Lessons 1–15
    on 2026-05-19 (≈338 unique candidate signs).
  - ASL-LEX 2.0 `signdata.csv` downloaded from the OSF supplementary
    materials (https://osf.io/zpha4/, file https://osf.io/download/9nygd/);
    277 of 338 Lifeprint candidates had direct ASL-LEX coverage,
    317 after alias-mapping (e.g. MOM → `mother`, MARRIAGE → `marry`,
    THIS → `this/it`). 96 final picks all have ASL-LEX codes.
  - WLASL `WLASL_v0.3.json` downloaded from
    https://github.com/dxli94/WLASL; per-gloss `instances[]` counts
    used as `WLASL Clips`. All 96 picks have ≥7 raw WLASL instances.
  - MS-ASL marked `PENDING` for every row; Microsoft's class
    manifests sit behind a license-acceptance download flow that
    was not completed this session.
- `static_or_movement` derived from ASL-LEX `Movement.2.0` and
  `RepeatedMovement.2.0`; only 8 of 96 signs are truly `static`
  by ASL-LEX phonology, which is consistent with ASL linguistics.
- `flippable` derived from ASL-LEX `SignType.2.0` (asymmetric
  signs and dominance/symmetry-violations are not flippable) plus
  a deny list for pronouns and directional verbs whose meaning is
  encoded by handedness.

**Files created or updated this session:**

- `docs/decisions/0004-public-sources-only.md` — new ADR.
- `docs/VOCABULARY.md` — new, 96 signs with full cross-reference.
- `claude/CLAUDE.md` — §4 decisions table extended; §5 open
  question on instructor closed and pointed at ADR 0004; §5 #3
  reworded.
- `docs/DATASET.md` — §1a "pending written confirmation" removed
  (decision locked in CLAUDE.md §4); §1b rewritten as
  "Canonical reference videos sourced from WLASL/MS-ASL with
  attribution"; §7 vocabulary-metadata field descriptions
  updated for slice-1 sourcing.
- `docs/ROADMAP.md` — Phase 0 work items 2 and 4 reworked; Phase 3
  work item 3 reworked as WLASL/MS-ASL selection with slice-2
  candidate.
- `docs/ARCHITECTURE.md` — three instructor references (R2
  `references/` bucket, Layer A hint authoring, hint-review line)
  reworded for slice-1 sourcing.
- `docs/MODEL.md` — confusion-pair hint authoring line reworded.
- `docs/PRIVACY.md` — `references/` bucket description reworded.
- `claude/CLAUDE.md` — moved from repo root into `claude/`.
- `claude/SESSION_LOG.md` — moved from repo root into `claude/`;
  this entry appended.
- `claude/CLAUDE_CODE_HANDOFF.md` — moved from repo root into
  `claude/`.
- `docs/{ROADMAP,ARCHITECTURE,PEDAGOGY,MODEL,DATASET,EVAL_GATE,PRIVACY}.md`
  — moved from repo root into `docs/`.
- `docs/decisions/{0001,0002,0003}-*.md` — moved from repo root
  into `docs/decisions/`.
- `docs/research/README.md` — moved from being the misnamed root
  `README.md` into `docs/research/`.
- `.gitignore` — created (TODO.md, OS junk, future Python/Node
  build artifacts).
- `TODO.md` — created at root, gitignored, working scratchpad.

**Open threads at end of session:**

- MS-ASL clip counts are `PENDING` across every row of
  `docs/VOCABULARY.md`. Resolving these requires accepting the
  Microsoft Research license and pulling the class manifest.
  Tracked as follow-up #1 in `docs/VOCABULARY.md`.
- WLASL `instances` counts are raw — the per-gloss
  actually-downloadable subset will be smaller (YouTube links rot).
  Will be reconciled in Phase 3 cleaning pipeline.
- Accessibility expectations from the partner remain open;
  default WCAG AA + captions stands.
- Demo audience confirmation (Patrick/Frank only vs. real
  learners) remains open.

**Where to start next session:**

Phase 2 — repo scaffolding. Initialize the Next.js 14 + TypeScript
+ Tailwind + shadcn/ui project under `app/` in the same repo;
wire ESLint, Prettier, Husky, GitHub Actions for typecheck/lint
on PR; create the Supabase project; create the Cloudflare R2
bucket; ship a minimal "hello" page to a Vercel deployment so the
whole pipeline is proven end-to-end. Exit per `docs/ROADMAP.md`
Phase 2 criteria.

## Session 3.5 — constraint clarification pivot (2026-05-19)

**Material project event:** brief Requirement 7 ("No Pretrained
Models") scope was clarified by Gauntlet staff on 2026-05-19. The
clarification establishes that Requirement 7 restricts pretrained
ASL pipelines and pretrained sign classifiers but **does not**
restrict general-purpose pretrained landmark detectors (MediaPipe
Hands / Holistic, OpenPose, BlazePose). Treated as authoritative
because it comes from the brief's authoring organization. The
strict reading of Requirement 7 that ADR 0001 was written against
is now a stricter interpretation than the brief author intended.

**Decisions locked:**

- **Recognition architecture pivots from Path B (end-to-end small
  3D CNN trained on raw pixels) to a landmark-based architecture**:
  MediaPipe Holistic for hand and pose keypoint extraction +
  small temporal classifier (2-layer BiLSTM ≈ 200K params baseline;
  small Transformer ≈ 500K params alternative) trained entirely
  from scratch with Kaiming init. ONNX-exported, ONNX Runtime Web
  in the browser. Combined client bundle target ≤ 5 MB.
- **ADR 0001 superseded by ADR 0006.** ADR 0001's body is preserved
  as the historical record of the decision made under the strict
  reading of Requirement 7. A `Superseded by` line was added below
  ADR 0001's `Status` header pointing forward to ADR 0006.
- **What stays the same** (per ADR 0006 "What stays the same"
  section): the entire pedagogical theory in `docs/PEDAGOGY.md`,
  the system-level architecture in `docs/ARCHITECTURE.md` §1, the
  mastery state machine and scheduler, the three-layer hint
  system, the eval gate hard and soft criteria, signer-disjoint
  splits, fairness reporting, the privacy architecture, the
  deployment platform (Vercel + Supabase + R2), the
  public-sources-only sourcing decision (ADR 0004), and the
  allowance of classical CV (ADR 0005, with role updated from
  load-bearing to slice-2 candidate).
- **What's smaller / faster as a consequence:** per-sign clip
  target drops from ~200 (Path B figure) to **50–80**; training
  time per run drops from hours to **minutes**; GPU rental cost
  drops to near-zero; deployed bundle drops from the Path B
  ~10 MB target to **≤ 5 MB combined**; per-clip classifier
  inference target tightens from 300 ms to **100 ms** and the
  end-to-end attempt latency target tightens from 1 second to
  **600 ms**.
- **What's promoted from slice-2 aspiration to slice-2 concrete
  target:** the parameter-aware hint system. Under the landmark-
  based architecture the classifier's input — keypoint sequences —
  already encodes handshape, location, palm orientation, and
  movement, so a second classifier head predicting the five sign
  parameters can be trained on the same data as the gloss
  classifier with effectively no additional data collection.
- **New hard eval-gate criterion (`docs/EVAL_GATE.md` §1
  criterion 10):** MediaPipe landmark detection must succeed on
  ≥ 95% of test clips; clips where MediaPipe fails are excluded
  from accuracy denominators but their failure rate is reported.
- **New honest disclosure (`docs/DATASET.md` §5):** the landmark-
  based architecture removes one major fairness axis (the
  classifier cannot learn skin tone as a spurious feature since
  it does not see pixels), but MediaPipe's own landmark-detection
  accuracy can vary across demographics, so the validation report
  reports MediaPipe per-demographic detection-success rate
  alongside per-demographic classifier accuracy.

**Files changed in this pivot pass:**

- `docs/decisions/0006-recognition-architecture-revised.md` — new ADR.
- `docs/decisions/0001-recognition-architecture.md` — `Superseded by`
  line added below `Status`; body unchanged.
- `docs/decisions/0005-classical-cv-allowed.md` —
  `Architectural assumption updated 2026-05-19 by ADR 0006` line
  added below `Status`; body unchanged. The classical-CV decision
  itself stands; the role changes from slice-1-essential to
  slice-2 candidate.
- `claude/CLAUDE.md` §4 — recognition-path row marked CHANGED with
  strikethrough; new row added pointing to ADR 0006; classical-CV
  row reworded to note slice-2-not-slice-1; file-map row for
  MODEL.md reworded to "no-pretrained-pipeline evidence."
- `docs/MODEL.md` — §1 (architecture) replaced with two-stage
  landmark + BiLSTM/Transformer spec; §2 (training) batch size,
  epochs, hardware adjusted (minutes not hours); §3 (augmentation)
  replaced with keypoint-level augmentations; §6 (export) keeps
  classifier-only ONNX export, quantization made optional;
  §7 retitled "No-pretrained-pipeline evidence" and rewritten;
  §8 performance targets tightened (5 MB bundle, 100 ms
  classifier, 600 ms end-to-end).
- `docs/DATASET.md` — header docstring updated; §1c per-sign
  target reduced to 50–80; recording-protocol storage rationale
  rewritten to drop stale 112×112 reference; cleaning pipeline
  gains a MediaPipe Holistic extraction stage with version
  recording; augmentation surface note added; §5 fairness honest-
  disclosure addendum added.
- `docs/EVAL_GATE.md` — §1 criterion 1 reframed as conservative
  floor with 90–95% as the realistic expectation; latency target
  tightened to 600 ms; criterion 9 reworded; new criterion 10 on
  MediaPipe detection success.
- `docs/ROADMAP.md` Phase 3 cleaning pipeline gains MediaPipe
  extraction step; exit criterion updated to 50–80 clips/sign
  and keypoint-tensor accompaniment. Phase 4 training step
  rewritten for the landmark-based classifier; validation harness
  gains MediaPipe detection-success reporting; ONNX export keeps
  classifier; quantization made optional; iteration cadence note
  added (training in minutes).
- `docs/ARCHITECTURE.md` — §2.3 inference runtime pipeline gains
  the MediaPipe Holistic extraction step between frame capture
  and classification, with a `detection_failed` outcome surfaced
  to the learner when MediaPipe cannot extract the required
  keypoints; performance targets refreshed; §2.5 training pipeline
  cleaning gains MediaPipe extraction, augmentation gains
  keypoint-level, training gains BiLSTM/Transformer baseline,
  export and artifact bundling updated; §5 hint-system gains the
  slice-2 parameter-aware-hints note.

**No code was written. No MediaPipe dependency added anywhere.**
This was a documentation-only pass.

**Honesty about the pivot:** the original Path B reading of
Requirement 7 was a defensible interpretation of the brief's text.
The 2026-05-19 clarification is a more permissive reading from the
authoring side and therefore governs. ADR 0001 stays in the repo
as honest historical record; the new architecture is in ADR 0006.
The validation report will name this pivot explicitly so a reader
of the final project sees both the original architectural
commitment and the clarified one.

**Where to start next session (revised after pivot):**

Phase 2 — repo scaffolding, exactly as queued before the pivot.
Phase 2 does not change at all under ADR 0006 (Next.js + Supabase
+ R2 + Vercel, plus ESLint/Prettier/Husky/GitHub Actions). The
MediaPipe and landmark-classifier work lives in Phase 4 and does
not enter `package.json` during Phase 2.

## Session 4 — Phase 2 scaffolding (2026-05-19)

**Phase 2 exit criterion met:** A reviewer can clone the repo, run
`pnpm install && pnpm dev`, and hit a working page; the same code
is live on a Vercel URL. ✓

**Live URLs:**

- Production: https://asl-mastery.vercel.app
- Aliases: https://asl-mastery-hirom0112-hirom0112s-projects.vercel.app,
  https://asl-mastery-hirom0112s-projects.vercel.app
- Both HTTP 200 with title "ASL Mastery".

**Decisions locked:**

- **Repo layout:** Next.js scaffolded at the repo root (not in a
  subdirectory). `claude/`, `docs/`, `app/`, `components/`,
  `public/` all live at root. Future `training/` Python subdir will
  sit alongside. Rationale: `CLAUDE_CODE_HANDOFF.md` hints at root-
  level layout and the `.gitignore` was already seeded for it.
- **Tooling versions locked at install time:** Next.js 16.2.6
  (latest stable, has API breaking changes per the scaffold's
  `AGENTS.md`), React 19.2.4, Tailwind v4, shadcn/ui with the
  base-nova preset (Base UI under the hood), pnpm 10.33.1,
  node@22.
- **Pre-commit:** Husky + lint-staged running Prettier and
  ESLint --fix on staged TS/TSX/JS/JSX; Prettier on staged JSON/CSS.
- **CI (GitHub Actions):** Prettier check + eslint + tsc + next
  build on every PR and push to main. No tests yet; Vitest is a
  Phase 5 candidate when the first testable code lands.
- **Prettier excludes** `claude/`, `docs/`, `TODO.md`, and all
  `*.md` files. Reason: those are prose the user controls; we
  don't want Prettier reflowing markdown tables.

**External services provisioned (cleanest CLI path):**

- **Supabase project `asl-mastery`** created in org `Hirom Org`
  (`ulldzlgomsdlkglmtayu`), region `us-west-1` (West US, North
  California), compute size `nano`. Project ref:
  `ehrqwtvrmejozwlybndl`. Dashboard:
  https://supabase.com/dashboard/project/ehrqwtvrmejozwlybndl.
  Pause behavior: org is on the Pro plan ($25/mo); Pro orgs
  do not pause projects regardless of compute size.
- **Cloudflare R2 buckets** created on account
  `06078a3d282287e09b90ab145bce9cf3`:
  - `asl-mastery-models` (public, r2.dev URL
    `https://pub-1d2c66f9b93e4a00a257a3dc0675d73b.r2.dev`).
  - `asl-mastery-references` (public, r2.dev URL
    `https://pub-58faaa60e7794818b22f1158e2e9d235.r2.dev`).
  - `asl-mastery-raw-training-data` (private; access via signed
    URLs only).
  Activation required clicking through R2 pricing in the Cloudflare
  dashboard before the API would accept bucket creates.
- **Vercel project** `asl-mastery` (ID `prj_ztw60B3KHXpWctaY0RaDykxdQX1b`)
  in the `hirom0112s-projects` team
  (`team_TScmf4mgrPSuaAjogMxcPNHF`). Plan: Pro ($20/mo).
  Deployment Protection was on by default (Vercel Pro feature, gives
  HTTP 401 on production) and was disabled via API
  (`ssoProtection: null`) so the production URL is publicly accessible.
  Production env vars set:
  `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`,
  `SUPABASE_SERVICE_ROLE_KEY`,
  `NEXT_PUBLIC_R2_PUBLIC_BASE_URL_MODELS`,
  `NEXT_PUBLIC_R2_PUBLIC_BASE_URL_REFERENCES`.

**Secrets handling:**

- All API keys / tokens / DB password live in `.env.local` at the
  repo root, which is gitignored via `.gitignore` (verified
  `git check-ignore .env.local`). The `.env.example` in the
  committed repo carries only placeholder names, no values.

**Files committed this session:**

- `commit 0da7dfd` — local scaffolding (Next.js + Tailwind + shadcn
  + Prettier + Husky + GitHub Actions + .env.example + hello page +
  .gitignore extensions).
- Pushed to both GitHub and GitLab per the auto-memory rule.
- External-service provisioning produces no committed code; all
  resource IDs and URLs are in this session log entry and the
  gitignored `.env.local`.

**Open follow-ups at end of session:**

- **Vercel ↔ GitHub auto-deploy hookup is not yet wired.** The
  Vercel CLI's `git connect` fails because the Vercel GitHub App
  is not installed on the `Hirom0112` GitHub account with access
  to the `asl-mastery` repo. User action required:
  https://github.com/apps/vercel → install / configure on
  `Hirom0112` → select `asl-mastery`. After that, the next push
  to `main` triggers a production deploy automatically. Until
  then, deploys can be triggered manually via `vercel --prod` from
  a local clone (as was done this session).
- **R2 server-side credentials (training-pipeline write access)
  are not yet created.** Phase 3 will need them. `wrangler` does
  not expose an R2-API-token-creation command; will create via
  Cloudflare dashboard when Phase 3 begins.
- **Sentry and PostHog not provisioned.** Placeholder env-var
  names exist in `.env.example`. Real provisioning deferred to
  Phase 6/7 when error tracking and analytics start being read.

**Where to start next session:**

Phase 3 prep — Postgres schema and migrations per
`docs/ARCHITECTURE.md` §2.4. Tables: `users`, `vocabulary_items`,
`confusion_pair_hints`, `model_versions`, `attempts`,
`mastery_state`. Approach: declarative SQL migrations committed to
`supabase/migrations/`, applied via `supabase db push` against the
linked project. Define row-level security policies for the
user-scoped tables (`attempts`, `mastery_state`, `users`) at the
same time.
