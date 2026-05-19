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
