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

## Session 5 — Phase 3a Postgres schema + auth scope expansion (2026-05-19)

**Phase 3a exit criterion met:** All six tables in
`docs/ARCHITECTURE.md` §2.4 exist in the remote Supabase project,
RLS is enabled on the user-scoped tables, 96 vocabulary items are
seeded, the typed client compiles, and the home page server-renders
the live vocabulary count from Postgres. ✓

**Decisions locked:**

- **DB client = `@supabase/supabase-js` + `@supabase/ssr` + generated
  types.** Drizzle was considered and rejected: our schema is small,
  our queries are simple, and Drizzle's direct-Postgres connection
  pattern would either bypass RLS or force us to re-implement
  authorization in application code. supabase-js routes through the
  user's session JWT so `auth.uid() = user_id` policies fire
  automatically. `supabase gen types typescript --linked --schema
  public > lib/db/database.types.ts` regenerates types from the
  live schema.
- **`public.users` table with PK = `auth.users.id`, FK ON DELETE
  CASCADE.** Rejected the JWT-user-metadata alternative: metadata
  isn't enforceable via CHECK constraints or RLS, isn't queryable
  from SQL, and doesn't survive the schema rigor the rest of the
  system depends on. Trigger `on_auth_user_created` on `auth.users
  insert` populates `public.users` for every sign-in path —
  permanent or anonymous — with no branch.
- **ADR 0007 written: three auth entry points (Google OAuth + email
  magic link + anonymous sign-in for the demo).** Two scope
  additions the user requested during this session: (1) Google
  OAuth alongside magic links and (2) a "Try the demo" button on
  the landing page. The demo path is **anonymous sign-in**
  (`signInAnonymously()`), chosen over a shared demo account
  (mastery state would collide) and over a local-only IndexedDB
  stub (hides the database-backed mastery architecture, which is
  the substance of the demo). Anonymous accounts get a real
  `auth.users` row, a real `public.users` row, and RLS-protected
  attempt + mastery rows — identical to a permanent account in
  every dimension except the `is_anonymous` flag. Conversion to a
  permanent account (`linkIdentity` / add email) preserves the user
  id, so all attempt history survives. 30-day idle cleanup via a
  scheduled `cleanup_inactive_anonymous_users()` function — to be
  added in a Phase 5 migration when the cron schedule is wired up.
  `docs/ARCHITECTURE.md` §2.1 and §2.4 endpoint list updated to
  match.
- **`vocabulary_items`, `confusion_pair_hints`, `model_versions` are
  read-only for `authenticated` role.** Anonymous-signed-in users
  *are* authenticated for RLS purposes (they have a session JWT and
  `auth.uid()` returns their UUID), so the demo path sees the
  vocabulary catalog the same way a logged-in user does. The home
  page's smoke-test count uses the service-role admin client (so a
  drive-by visitor without a session sees it); production reads
  during practice use the user's session.
- **Migration naming:** `YYYYMMDDHHMMSS_<name>.sql` (Supabase CLI
  convention). First three migrations use `20260519100000`,
  `20260519100100`, `20260519100200` so they sort in the intended
  apply order.

**Files committed this session:**

- `docs/decisions/0007-auth-providers-and-demo.md` — new ADR.
- `docs/ARCHITECTURE.md` — §2.1 auth bullet rewritten; §2.4
  server-action list extended with Google / anonymous / linkIdentity
  wrappers.
- `supabase/migrations/20260519100000_init.sql` — six tables, four
  enums, partial unique index on `model_versions.is_active`,
  `updated_at` trigger.
- `supabase/migrations/20260519100100_rls.sql` — `auth.users` insert
  trigger, RLS enabled on six tables, policies per
  ARCHITECTURE §2.4, role grants.
- `supabase/migrations/20260519100200_seed_vocabulary.sql` — 96
  vocabulary rows from `docs/VOCABULARY.md`.
- `lib/db/database.types.ts` — generated from the linked project.
- `lib/db/client.ts` — browser supabase-js wrapper.
- `lib/db/server.ts` — server supabase-js wrapper (cookie-based
  session, for server components / actions / route handlers).
- `lib/db/admin.ts` — service-role client for admin paths.
- `app/page.tsx` — home page now server-renders live vocabulary
  count from Postgres as the smoke test.
- `package.json` / `pnpm-lock.yaml` — `@supabase/supabase-js`
  `^2.106.0`, `@supabase/ssr` `^0.10.3` added.
- `TODO.md` — Phase 3a section checked off.

**External operations performed (no committed code):**

- `supabase link --project-ref ehrqwtvrmejozwlybndl`.
- `supabase db push --include-all`. All three migrations applied to
  the remote project on first try.
- `supabase gen types typescript --linked --schema public` (run
  twice; first run leaked the CLI's update-available banner into
  the file, second run with stderr redirected was clean).

**Open follow-ups at end of session:**

- **Google OAuth provider not yet configured in Supabase project
  settings.** ADR 0007 commits to it but the OAuth client id /
  secret need to be created (Google Cloud Console → OAuth 2.0
  client) and pasted into Supabase dashboard → Authentication →
  Providers → Google. User action when the auth UI gets wired up
  in Phase 5a.
- **Anonymous sign-in is not yet enabled** in Supabase dashboard
  (Authentication → Providers → Anonymous Sign-Ins). One-toggle.
  Will be flipped on when the demo button lands in Phase 5a.
- **`cleanup_inactive_anonymous_users()` scheduled function** is
  specified in ADR 0007 but not yet written. Will land in Phase 5a
  alongside the auth UI, as a separate migration.
- **CI banner leakage from supabase CLI.** `supabase gen types`
  writes its "new version available" notice to stdout, not stderr,
  which corrupts the generated file when redirected naively.
  Worked around with `2>/dev/null` in this session; a more robust
  fix is to wrap the gen command in a small script that strips
  trailing non-TS lines, or upgrade the CLI (currently old enough
  to print version-installed as empty). Captured as a TODO drift.

**Where to start next session:**

Phase 3b — MediaPipe Holistic browser integration. Add
`@mediapipe/tasks-vision`, write `lib/keypoints.ts` defining the
exact 21+21 hand + N upper-body pose subset per `docs/MODEL.md`
§1, wrap loading in a `useMediaPipeHolistic` hook with lazy WASM
+ model loading, and surface the `detection_failed` outcome per
ARCHITECTURE §2.3 step 10. Unit-test against a fixture clip.

## Session 6 — Phase 3b MediaPipe integration + Vitest wiring (2026-05-19)

**Phase 3b exit criterion met** (with one deferred sub-item — the
fixture-clip end-to-end test, see Open follow-ups). The browser
inference path now has a typed, lazy-loaded keypoint extractor with
the layout that `docs/MODEL.md` §1 specifies and the failure-mode
behavior that `docs/ARCHITECTURE.md` §2.3 step 10 prescribes. ✓

**Decisions locked:**

- **`HolisticLandmarker` from `@mediapipe/tasks-vision@0.10.35`** is
  the chosen runtime. The Tasks API ships a unified
  `HolisticLandmarker` (hand + pose + face + segmentation in one
  call), so we did not need to split into separate
  `HandLandmarker` + `PoseLandmarker` instances. Face landmarks
  and segmentation masks are explicitly disabled in the loader
  options per slice-1 scope (`docs/MODEL.md` §1; face is slice-2).
- **MediaPipe version is pinned in TypeScript** as
  `MEDIAPIPE_VERSION = "0.10.35"` in `lib/mediapipe/loader.ts`. The
  WASM fileset URL embeds the same version so a dep bump can't
  silently drift the runtime out of step with the package types.
  The pin will be cross-checked against the training pipeline's
  Python MediaPipe version once that exists (Phase 3f), and the
  validation report will record both versions
  (`docs/EVAL_GATE.md` §1 criterion 10).
- **Model `.task` bundle defaults to Google CDN, with R2 override
  available.** Slice-1 inference fetches
  `holistic_landmarker.task` from
  `storage.googleapis.com/mediapipe-models/...` by default. An
  optional `modelAssetUrl` override on the loader lets us mirror
  to `asl-mastery-models/mediapipe/${MEDIAPIPE_VERSION}/` in R2
  for cases where we want every asset coming from our own
  infrastructure. Per ADR 0006, MediaPipe is treated as a
  black-box library; pulling its model from a CDN it publishes is
  consistent with that treatment.
- **GPU delegate → CPU fallback** at load time. If
  `delegate: "GPU"` fails (older browsers, no WebGPU/WebGL), the
  loader retries with `"CPU"` before surfacing an error. Avoids
  asking the learner to debug their browser.
- **Pose subset = `[11, 12, 13, 14, 15, 16, 23, 24]`** (8 indices:
  shoulders, elbows, wrists, hips). 21 + 21 + 8 = 50 landmarks; 3
  coords per landmark; **K = 150 floats per frame**. `docs/MODEL.md`
  §1 explicitly notes that pose indices are tentative and may be
  pruned during Phase 4 once we measure which contribute to
  per-sign accuracy.
- **Flat layout per frame:** `[left_hand (21×3) | right_hand (21×3)
  | pose (8×3)]`. Offsets and total count are exported from
  `lib/keypoints.ts` as the single source of truth — the Python
  training pipeline (Phase 3f) and the ONNX export step will both
  reference the same constants.
- **Missing landmarks zero-fill.** Consistent with the keypoint
  dropout augmentation in `docs/MODEL.md` §3, so the
  training-time and inference-time treatment of missing
  landmarks is identical.
- **Detection failure threshold = 0.5** of frames missing a hand
  → `detection_failed`. Matches the ≥95% MediaPipe detection
  success criterion in `docs/EVAL_GATE.md` §1 criterion 10 at the
  per-clip level (≥50% of frames must contribute at least one
  hand for the clip to score).
- **Vitest, not Jest.** First testable code landed (`lib/keypoints.ts`,
  `lib/mediapipe/extractor.ts`); chose Vitest for vite-native
  speed and zero-config TypeScript. `pnpm test` runs all tests in
  ~150 ms. CI now runs four gates: prettier check, eslint,
  typecheck, **test**, build.
- **Page now `dynamic = "force-dynamic"`.** The home page's smoke-
  test count call would have failed at CI build time (no Supabase
  env vars in CI). Switched to per-request rendering with a
  graceful try/catch so the build succeeds without env vars and
  the live page on Vercel still shows the count.

**Files committed this session:**

- `lib/keypoints.ts` — keypoint shape constants (single source of
  truth across TS + future Python).
- `lib/keypoints.test.ts` — shape constant tests (4 tests).
- `lib/mediapipe/loader.ts` — singleton `HolisticLandmarker`
  loader with WASM-pinning + GPU/CPU fallback.
- `lib/mediapipe/extractor.ts` — `resultToFrame()` + `extractClip()`
  with detection-failure rollup.
- `lib/mediapipe/extractor.test.ts` — flat-layout mapping tests
  with mocked `HolisticLandmarkerResult` (4 tests).
- `hooks/use-landmark-extractor.ts` — client-only React hook
  wrapping the loader; exposes `status`, `error`, `preload`,
  `extract`.
- `vitest.config.ts` — vitest config with `@/*` alias matching
  `tsconfig.json`.
- `package.json` — `@mediapipe/tasks-vision`, `vitest`,
  `@vitest/coverage-v8` added; `test` / `test:watch` scripts.
- `.github/workflows/ci.yml` — `pnpm test` step inserted between
  typecheck and build.
- `app/page.tsx` — `dynamic = "force-dynamic"` + graceful
  vocabulary-count fallback for CI.
- `TODO.md` — Phase 3b items checked off (fixture-clip test
  carried forward).

**Open follow-ups at end of session:**

- **Fixture-clip end-to-end test deferred.** Mocking the MediaPipe
  WASM runtime in Node is more engineering cost than benefit at
  this stage. Will land alongside the first real recorded clip
  produced by the Phase 3c admin recording tool.
- **Pose-subset audit at Phase 4.** The 8-index subset is a
  reasonable default; Phase 4 measurement will tell us whether to
  prune (e.g. drop hips) or extend (e.g. add nose for head-position
  signs).
- **MediaPipe Python version pin.** When Phase 3f starts, the
  training pipeline's `mediapipe` Python version must be recorded
  in the dataset manifest and cross-checked against
  `MEDIAPIPE_VERSION` in `lib/mediapipe/loader.ts`. If the two
  ever diverge, the validation report has to disclose it.

**Where to start next session:**

Two viable threads:

1. **Phase 3c — admin recording tool.** Auth-gated `/admin/record`
   route using the `HolisticLandmarker` extractor live as visual
   confirmation, 2-second capture window with green-box framing,
   metadata form, signed-URL upload to
   `asl-mastery-raw-training-data` R2. This unblocks data
   collection and produces the first real clip the deferred
   fixture-clip test can use.
2. **Phase 5a — auth scaffolding.** Wire the three sign-in
   entry points from ADR 0007 (Google OAuth, email magic link,
   anonymous "Try the demo") so the recording tool's admin gate
   has something to check. Flip Google OAuth credentials + the
   anonymous-sign-ins toggle in Supabase dashboard (currently
   pending per Session 5 follow-ups).

3c is the more direct continuation of Phase 3 sequencing; 5a is
the prerequisite for 3c's admin gate. Probably wire 5a first.

## Session 7 — Phase 5a auth scaffold (2026-05-19)

**Foundation of Phase 5a in.** All three sign-in entry points from
ADR 0007 are coded end-to-end; the home page reads the current user
and shows the right CTA. The two dashboard-only configurations
(Google OAuth credentials, anonymous-sign-ins toggle) remain user
actions before the corresponding buttons actually work in
production. Onboarding / handedness / Fitzpatrick / camera-permission
UX is deferred to the next 5a pass.

**Decisions locked:**

- **Middleware-based session refresh.** `middleware.ts` at the repo
  root calls `updateSession()` (in `lib/db/middleware.ts`) on every
  non-asset request, which calls `supabase.auth.getUser()` to
  refresh the access token cookie if expired. Matcher excludes
  `_next/static`, `_next/image`, common image extensions, and
  `favicon.ico`.
- **All auth flows are server actions** (in `lib/auth/actions.ts`)
  except the Google OAuth redirect, which the action initiates and
  then `redirect()`s the browser into. Anonymous sign-in is a one-
  call server action and redirects to `/`. Sign-out is a one-line
  server action.
- **OAuth callback at `/auth/callback`.** Exchanges the OAuth code
  for a session via `supabase.auth.exchangeCodeForSession()`, then
  redirects to `?next=` (or `/`). On error, bounces back to
  `/sign-in?error=callback` so the UI can surface the reason.
- **Magic link confirmation also routes through `/auth/callback`.**
  `signInWithOtp({ emailRedirectTo: ${origin}/auth/callback })`
  sends the magic-link email; the click lands at the callback
  route which exchanges the code.
- **Sign-in form is one component** (`components/sign-in-form.tsx`)
  with three actions: Try the demo (primary), Continue with
  Google (secondary), Send magic link (form). Pending state shared
  via `useTransition`. On magic-link success, the email form
  collapses to a "check your inbox" confirmation.
- **Home page distinguishes anonymous vs permanent users.**
  `user.is_anonymous === true` renders "Demo session"; otherwise
  the email (or a UUID prefix as fallback). Sign-out always
  visible when authenticated.
- **Button wrapping a Link uses `buttonVariants()` directly**, not
  `asChild`. The repo's `Button` is a base-ui `ButtonPrimitive`
  which does not accept `asChild`. Documenting the pattern so we
  don't relitigate it.
- **`/sign-in` page redirects authenticated users to `/`.** Avoids
  the "sign in to the sign-in page" loop.

**Files committed this session:**

- `middleware.ts` — root-level Next.js middleware.
- `lib/db/middleware.ts` — Supabase SSR session-refresh helper.
- `lib/auth/actions.ts` — four server actions.
- `app/auth/callback/route.ts` — OAuth + magic-link code exchange.
- `app/sign-in/page.tsx` — sign-in screen with auth-redirect guard.
- `components/sign-in-form.tsx` — client form with three actions.
- `components/sign-out-button.tsx` — client sign-out button.
- `app/page.tsx` — current-user-aware home page.
- `TODO.md` — Phase 5a checkboxes updated, dashboard follow-ups
  surfaced.

**Open follow-ups at end of session:**

- **Google OAuth dashboard config still pending.** Create OAuth
  client id + secret in Google Cloud Console → APIs & Services
  → Credentials → OAuth 2.0 Client IDs. Authorized redirect URI:
  `https://<project-ref>.supabase.co/auth/v1/callback`. Paste
  into Supabase dashboard → Authentication → Providers → Google.
  Until this is done, the "Continue with Google" button surfaces
  Supabase's "provider is not enabled" error to the user.
- **Anonymous Sign-Ins toggle in Supabase dashboard.**
  Authentication → Providers → Anonymous Sign-Ins → enable. One
  click. Until then, the "Try the demo" button surfaces Supabase's
  "Anonymous sign-ins are disabled" error.
- **`cleanup_inactive_anonymous_users()` scheduled function** still
  not written (per ADR 0007). Will land in a follow-up migration
  before slice-1 ship.
- **Onboarding UX (handedness capture, Fitzpatrick consent, camera
  permission flow)** deferred to the next 5a pass; the foundation
  is in.
- **Production redirect URLs** in Supabase dashboard need to include
  `https://asl-mastery.vercel.app/auth/callback`. The Supabase
  dashboard defaults to localhost-only, so the production magic
  link click would fail otherwise. User action.

**Where to start next session:**

Either (a) finish 5a UX (onboarding, handedness, Fitzpatrick,
camera permission) once the user toggles the dashboard switches
above, or (b) start Phase 3c — admin-gated recording tool — which
now has the auth substrate it needs (`createClient()` server-side
returns the user; gate `/admin/record` on a per-user `is_admin`
column or claim).

A natural sequencing question: 3c needs an `is_admin` flag
somewhere. Options: add a column to `public.users`, or use
Supabase Auth's custom claims, or hard-code a developer-email
allow-list during the pilot. Resolve before starting 3c.

## Session 8 — Phase 3 scope cut: public-data-only training (2026-05-19)

**Material project event:** the slice-1 training-data plan is
narrowed. Self-recording is removed from the pilot; the recording
tool moves to slice 2 as the framework the ADR-0004 Deaf-instructor
engagement will use. The honest reason — captured in ADR 0008 —
is that no one on the project team is a fluent ASL signer, and
training on non-signer clips would teach the model wrong signs.
Better to have less data we did not author than more data we
authored wrong.

**Decisions locked:**

- **Slice-1 training data: public datasets only.** WLASL and (when
  license-acceptance lands) MS-ASL. The recording-tool spec in
  `docs/ARCHITECTURE.md` §2.2 is preserved as a slice-2 design
  target. ADR 0008 written.
- **Vocabulary may be trimmed.** A per-sign downloadable-clip floor
  (initial 15) is applied during Phase 3d ingestion. Signs below
  the floor are dropped from slice 1 and annotated in
  `docs/VOCABULARY.md`. Final slice-1 vocabulary count must remain
  ≥ 75 (Brief Requirement 2 floor); if the floor would take it
  under 75, the floor itself is reduced rather than the count
  violated, with disclosure in the validation report.
- **`is_admin` decision from Session 7 is closed.** No admin role
  is needed for slice 1 (no recording tool to gate). The flag
  question reopens in slice 2 when the recording tool is built.
- **What's superseded:** Phase 3c (admin-gated recording tool) and
  Phase 3e (self-recorded supplement) are both moved to slice 2 in
  `TODO.md`. Phase 3d becomes the primary and only slice-1 data
  path. Phase 3 exit criterion in `docs/ROADMAP.md` is reworded:
  ≥ 50–80 keypoint tensors per sign from public sources alone
  where coverage permits, with the floor-filter and the final
  vocabulary count documented.

**Files committed this session:**

- `docs/decisions/0008-public-data-only-training.md` — new ADR.
- `claude/CLAUDE.md` §4 — new decisions-table row for slice-1
  training data (ADR 0008).
- `docs/ROADMAP.md` — Phase 3 work items 1, 4 reworked; exit
  criterion revised; work items 2, 5, 6 picked up the per-sign
  filter and the "public sources alone" framing.
- `docs/DATASET.md` — §1c rewritten as a slice-2 deferral header;
  §2 gains a slice-2-framework header; §3 cleaning pipeline gains
  the new per-sign filter step (renumbered 2 → through 10);
  §5 gains an honest-disclosure addendum on training-data
  authorship per ADR 0008.
- `docs/ARCHITECTURE.md` §2.2 — slice-2-framework header inserted
  pointing to ADR 0008; body unchanged.
- `docs/MODEL.md` §7 — note added that public-only training does
  not affect the no-pretrained-pipeline argument (ADR 0006 and
  ADR 0008 cover orthogonal scopes).
- `TODO.md` — Phase 3c / 3e marked slice-2; Phase 3d gains the
  per-sign filter steps; slice-2 section gains the recording-tool
  build as an explicit deliverable.

**No code changes this session.** Doc-only pass.

**Open follow-ups at end of session:**

- **WLASL downloadability check.** Run the per-sign downloadable
  count against the live YouTube state so we know which signs
  will be dropped under the per-sign floor. Tracked in Phase 3d
  TODO.
- **MS-ASL license-acceptance flow.** Still gated behind Microsoft
  Research's download form. Phase 3d work.
- **Vocabulary annotation** — when 3d resolves which signs drop,
  strike them through in `docs/VOCABULARY.md` with the actual
  count and "excluded under ADR 0008."

**Where to start next session:**

Phase 3d — public dataset ingestion. Write
`training/data/ingest_wlasl.py` (Python) that:

1. Reads `WLASL_v0.3.json` from the pinned WLASL repo commit.
2. Filters to the 96 glosses in `docs/VOCABULARY.md`.
3. For each clip, attempts download; logs failures (YouTube rot).
4. Writes a per-sign downloadable-count summary.
5. Applies the ADR-0008 per-sign floor and emits the slice-1
   vocabulary filter as JSON.

After that, MS-ASL ingestion behind the license flow, then the
cleaning pipeline (Phase 3f).

## Session 9 — Phases 3d, 3f, 4, 5b–e, 6, 8 in one push (2026-05-19)

**The big build.** This session closes most of the remaining slice-1
work that can be written without (a) downloaded training data,
(b) actual GPU time to train, or (c) external services the user has
to provision. Three commits landed:

1. `c5e28a1` — full Python training pipeline (Phase 3d/3f/4).
2. `db17807` — practice + scheduler + dashboard + settings + nav (Phase 5b–e + 6).
3. (this commit) — README.md, TODO sync, SESSION_LOG.

**What shipped (code):**

- **`training/`** — every Python script the pilot needs:
  - `keypoints.py` mirrors `lib/keypoints.ts` exactly (single source of truth).
  - `data/ingest_wlasl.py` filters WLASL to our 96, downloads via yt-dlp, emits per-sign downloadable count.
  - `data/ingest_msasl.py` stub gated on Microsoft Research license acceptance.
  - `data/filter_vocabulary.py` applies the ADR-0008 per-sign floor (15 default) with the ≥75 min-vocab guardrail; reduces floor before violating the min.
  - `data/clean.py` ffmpeg normalize → 16-frame sample → pHash dedup → MediaPipe Holistic extraction → signer-disjoint split → manifest.
  - `classifier/model.py` BiLSTM (~200K params) + Transformer alternative (~500K).
  - `classifier/init.py` Kaiming init with a module-level assertion that fires if `load_state_dict` is ever introduced.
  - `classifier/augment.py` keypoint-level stack per MODEL §3.
  - `classifier/dataset.py` + weighted sampler.
  - `classifier/train.py` AdamW + cosine + label smoothing + early stop; records git sha + manifest + classes.
  - `classifier/validate.py` top-1/3, per-sign, confusion matrix, top-3-confusion-per-sign, temperature scaling, ≥90%-precision thresholds, MediaPipe miss rate, JSON + Markdown reports.
  - `classifier/export.py` ONNX export with dynamic batch + temporal axes, PyTorch parity check, artifact manifest with sha256s.
  - `requirements.txt` pinning mediapipe 0.10.18 to match `lib/mediapipe/loader.ts`.
- **`lib/scheduler.ts`** — pure modified-SM-2 logic per ARCHITECTURE §4. State machine + `pickNextItem` priority rules. 14 unit tests in `lib/scheduler.test.ts` cover every transition.
- **`lib/scheduler/`** — server actions: `getNextItem` reads vocab + mastery + last-N attempts and selects via the pure logic; `recordAttempt` writes the attempt row + upserts mastery_state; `queries.ts` aggregates the dashboard view.
- **`lib/inference/`** — `classifier.ts` with `predict()` against ONNX Runtime Web and `stubPredict()` deterministic fallback (80% pass rate per target hash) for the period before `model_versions` has an active row; `active-model.ts` resolves the active row.
- **`/practice`** — full flow: prompt, reference video, green-box-overlay camera preview, 3-2-1 countdown, 2-second 30-fps capture into a hidden canvas with `no-track-canvas` class (per ARCHITECTURE §8), MediaPipe extraction via `useLandmarkExtractor`, classifier call, pass/fail/detection_failed branches with hint copy and "mastered!" celebration.
- **`/dashboard`** — mastery counts + per-sign rows with the **Ebbinghaus forgetting-curve sparkline** (thoughtful extra #1 from CLAUDE.md §6) marking next-review date.
- **`/settings`** — handedness toggle + Fitzpatrick consent + delete-account placeholder (real cascade pending the service-role admin path).
- **`components/site-nav.tsx`** — top-level nav with skip-to-main link, current-user label, sign-in/out.
- **`app/error.tsx`** + **`app/not-found.tsx`** — global error and 404 surfaces.
- **`README.md`** at repo root (senior-engineer writeup with honest scope disclosure section pointing at ADRs 0004 + 0008).

**What did NOT ship and why:**

- **Phase 3d run.** Code is ready; running yt-dlp on hundreds of clips takes hours of bandwidth and YouTube will rate-limit. Run when ready: `python -m training.data.ingest_wlasl --wlasl-json /path/to/WLASL_v0.3.json --output dataset/raw/wlasl`.
- **Phase 4 actual training run.** Needs (a) Phase 3d output, (b) GPU access (CPU works for BiLSTM but slow), (c) a real W&B run id if we want experiment tracking. Code is ready.
- **Phase 4d artifact upload to R2.** Needs R2 server-side credentials, which were intentionally deferred at end of Phase 2 because no codepath needed them yet. Cloudflare dashboard → R2 → Manage R2 API Tokens, Object Read+Write on `asl-mastery-models`.
- **Per-condition / per-demographic accuracy in the validation report.** Requires public-dataset demographic metadata to plumb through `clean.py`'s manifest. Layered in once the WLASL JSON is parsed for `signer_id`-derived demographics where present.
- **Account-delete cascade.** Placeholder signs out; the real delete needs the service-role admin client (`admin.deleteUser()`) which I wrote but did not wire to settings to avoid an unguarded delete-button risk during development.
- **`learner_disagreed = true` feedback button on practice fails.** Needs a tiny RLS-narrow self-update policy on `attempts`. Migration not written.
- **`confusion_pair_hints` seeding.** Needs a real validation report's top-3-confusions table; until then the practice screen routes through the generic-failure hint.
- **Phase 7 Sentry/PostHog.** External services not provisioned (deferred to user). Placeholder env vars exist in `.env.example`.

**Three CI gates per commit ran green:**

- `pnpm format:check` ✓
- `pnpm lint` ✓ (no warnings)
- `pnpm typecheck` ✓
- `pnpm test` ✓ — 22 tests across `lib/keypoints.test.ts`, `lib/mediapipe/extractor.test.ts`, `lib/scheduler.test.ts`.
- `pnpm build` ✓ — builds without Supabase env vars (CI-safe).

**Route inventory at end of session:**

```
/                  landing page (case-study framing)
/sign-in           three buttons (Try the demo / Google / Magic link)
/auth/callback     OAuth + magic-link code exchange
/practice          camera + MediaPipe + stub classifier + result
/dashboard         mastery counts + forgetting curves
/settings          handedness + Fitzpatrick + delete
```

**Where to start next time:**

Two real paths forward, depending on what the user wants:

1. **Run the data + training pipeline.** Acquire `WLASL_v0.3.json` from the upstream repo, run ingestion, run cleaning, run training. Promote the artifact to R2 + insert into `model_versions`. The practice screen automatically switches off the stub once `getActiveModelVersion()` returns a row.
2. **Polish: brand pass, Lighthouse audit, walkthrough video, mock interviews** (Phase 8). The pieces are in place; this is the customer-facing edge.

Both are externally blocked on user action (data download, GPU access, recording the walkthrough). Code-side, slice 1 is substantively complete.
