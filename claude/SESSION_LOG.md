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
