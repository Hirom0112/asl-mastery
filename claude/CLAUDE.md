# CLAUDE.md — Project Context for Claude Code

> **Read this file first, every session, before doing anything else.**
> This is the persistent context for the ASL Mastery project. It encodes
> what we are building, who we are building it for, what we have already
> decided, and the rules of engagement.

---

## 1. What this project is

A **mastery-based skill acquisition system, instrumented for measurable
learning outcomes, that proves a pedagogical theory — using ASL
vocabulary as the controlled testbed.**

The brief calls it "an ASL learning app for college ASL 1 learners."
That is the *surface* of the project. The *substance* is that we are
demonstrating a pedagogical architecture that could generalize to any
skill, for any school, for any learner — measured by time-to-mastery,
not engagement. We are doing this for Superbuilders, the engineering
team behind Alpha School and other microschools, whose stated mission
is to accelerate educational outcomes for 1 billion kids.

---

## 2. Who we are building this for

**The brief evaluates this work** against three pillars, in order:
**mission alignment, agency, engineering skill.** Mission alignment
is non-negotiable and ranked above engineering skill.

**The learner in the brief:** college students in an introductory ASL 1
course. The brief says explicitly: assume the learner is new to ASL,
needs repeated practice, clear feedback, and progress tracking.

**The learner we frame the system for:** any future user of a
mastery-based skill acquisition system. ASL 1 college students are the
controlled pilot population; the architecture has to be defensible for
broader populations the existing school system has failed (Deaf
children without instructor access being the most morally weighty
example we cite explicitly).

---

## 3. Operating principles (locked, do not drift from these)

These are the user's stated rules. Honor them in every session.

1. **Production-grade or do not ship.** Pretend the timeline is endless
   unless the user explicitly states otherwise. Do not propose MVPs.
   Do not propose half-broken software. Do not propose "we'll come back
   to it later" as a permanent state. Slices are allowed; quality is not
   negotiable.

2. **Mastery, not engagement.** Every design decision must answer
   "does this help a learner reach mastery faster and then let them go
   live their life?" If the answer is no, the decision is wrong. Do not
   propose streaks, DAU mechanics, notification engagement, or anything
   else from the engagement-app playbook.

3. **Cite or do not claim.** Every pedagogical or scientific claim in
   user-facing documentation must have a real citation pointing to a
   real source. No "I recall reading" claims. No vague "research shows."
   If a citation cannot be produced, flag the claim as PENDING CITATION
   and either find the source or remove the claim.

4. **Show, do not tell.** Mission alignment is demonstrated by
   architecture, not by mission-statement language. Examples: an eval
   gate that blocks promotion of regressing models; per-demographic
   accuracy in the validation report; signs that exit the practice
   rotation when mastered; a forgetting-curve visualization that makes
   the spaced-retrieval theory visible to the learner.

5. **Two theories at once.** Every engineering decision must answer to
   both software theory and pedagogical theory. When proposing a
   technical choice, state which pedagogical principle it serves. When
   proposing a pedagogical choice, state how the architecture supports
   it.

6. **No vibes-based AI.** Every AI output must have a measurable
   correctness signal. Pass/fail must use documented thresholds. Hints
   must trace back to either the confusion matrix or pre-authored
   per-sign content. The validation report must be reproducible from
   committed code and a frozen test set.

---

## 4. Decisions already made (do not re-litigate without cause)

| Decision | Choice | Rationale |
|---|---|---|
| ~~Recognition path~~ CHANGED 2026-05-19, REINSTATED 2026-05-20 | Path B: end-to-end small 3D CNN (R(2+1)D-style), trained from scratch on raw RGB. ~5–10M params, input `(B, 16, H, W, 3)`, Kaiming init. ONNX-exported with INT8 quantization, ONNX Runtime Web. Client bundle target ≤ 10 MB. | The no-pretrained-models constraint rules out landmark-based paths under the strict reading of Requirement 7. End-to-end is the cleanest defense and the most honest reading of the constraint. **ADR 0001 (original) → ADR 0006 (landmark pivot under an earlier permissive reading, 2026-05-19) → ADR 0010 (reversal, 2026-05-20).** The permissive reading was withdrawn; the strict reading governs again. |
| ~~Recognition path (2026-05-19, landmark-based)~~ CHANGED 2026-05-20 | ~~Landmark-based: MediaPipe Holistic for hand + pose keypoint extraction in the browser, then a small temporal classifier (2-layer BiLSTM, ~200K params; small Transformer ~500K params as alternative) trained from scratch on keypoint sequences. ONNX-exported, ONNX Runtime Web. Combined client bundle target <5 MB.~~ | ~~Authorized under ADR 0006 by an earlier permissive reading of Requirement 7. v1.0.1, v2.0.0, v2.1.0 shipped on this architecture.~~ **Superseded 2026-05-20 by ADR 0010 after the permissive reading was withdrawn. All three artifacts deactivated; architecture reverts to ADR 0001 Path B (row above).** |
| Frontend stack | Next.js (App Router), TypeScript, React, Tailwind | Matches the evaluators' stack expectations (TS/React-based). |
| Inference runtime | ONNX Runtime Web, WebGPU primary, WebGL fallback | Best browser ML perf currently. Quantize model to int8 for size. |
| Auth/DB | Supabase | Boring, fast, free tier covers pilot. |
| Hosting | Vercel (app) + Supabase (DB/auth) + Cloudflare R2 (model artifacts and reference videos). Each best-in-class for its slice. | See ADR 0003. |
| Training location | Server-side, on rented GPU (Modal/RunPod/Vast.ai or similar hourly cloud GPU). Inference is browser-side per Requirement 5. | Browser cannot train this model. Brief is silent on training location. |
| Mirroring rule | Preview is mirrored (selfie view); model input is un-mirrored. Train on un-mirrored video. | Matches user expectation; ensures train/inference consistency. |
| Left-handed signers | Capture handedness at signup; horizontally flip frame at inference for lefties; model only ever sees right-handed input. | Smaller training cost, simpler model. |
| Framing | Visible green box during recording; crop to box before model input. Out-of-box content is invisible to the model. | Solves multi-signer / roommate / pet edge cases architecturally. |
| Classical CV libraries | **CONFIRMED ALLOWED** (see ADR 0005). **Load-bearing again for slice-1 augmentation under ADR 0010** — MOG2 background swap, color jitter, brightness/contrast, small affine, conditional horizontal flip. The interim downgrade to "slice-2 candidate" under ADR 0006 is rescinded along with that ADR. | Not pretrained models; hand-coded algorithms. |
| Public ASL datasets | **CONFIRMED ALLOWED.** WLASL, MS-ASL, ASL-LEX usable as training data. | Raw video, not pretrained models. Brief Requirement 6 permits engineer-curated datasets. |
| Placement / fluency test | **REMOVED from plan.** Brief says assume learner is new to ASL. Upfront assessment is anti-pedagogical for true beginners. Scheduler handles "learner already knows this" via fast mastery transitions. | User correctly pushed back on this. |
| Vocabulary count | 75–100 vocabulary items (brief is explicit and non-negotiable) | Brief Requirement 2. |
| Alphabet inclusion | **EXCLUDED.** Vocabulary must be life-usable content signs only. Alphabet/fingerspelling does not count toward the 75–100. | User confirmed. Pushes toward content vocabulary that serves real communication. |
| Demo audience | User produces a demo video plus deploys an application Superbuilders actually uses. Demo is both video walkthrough and live app. | User confirmed. |
| ASL instructor for pilot | **NOT RECRUITED for slice 1** (resource constraint). Vocabulary, linguistic metadata, hint copy, and reference video sourced from public sources only: ASL-LEX 2.0 (phonological annotation), Lifeprint.com (ASL 1 curriculum by deaf educator Bill Vicars), WLASL and MS-ASL (reference clips with attribution). Instructor review of vocabulary/hints and instructor-recorded canonical references are framed as slice-2 production-deployment requirements, not pilot work. | See `docs/decisions/0004-public-sources-only.md`. Engaging a Deaf instructor without time and budget to do it fairly is worse than self-limitation with honest disclosure (CLAUDE.md §3 rule 3, PEDAGOGY.md §8). |
| Slice-1 training data (2026-05-19) | **Public datasets only (WLASL, MS-ASL).** Self-recorded supplement removed from slice 1 because no one on the project team is a fluent ASL signer; training on clips authored by a non-signer would teach wrong handshape / location / movement (worse than less data — misleading data). Recording tool spec in `docs/ARCHITECTURE.md` §2.2 is preserved as slice-2 framework for the ADR-0004 instructor engagement. Vocabulary may be trimmed if per-sign clip coverage is below the 15-clip floor; ≥75 signs remain (Brief Requirement 2 floor). | See `docs/decisions/0008-public-data-only-training.md`. Natural extension of ADR 0004 to training-data authorship. |

---

## 5. Open questions

Most blocking questions have now been resolved (see Section 4 for the
locked decisions). What remains:

1. ~~**ASL instructor / Deaf community contact.**~~ CLOSED. Pilot uses
   public sources only; instructor engagement is a slice-2 production-
   deployment requirement. See `docs/decisions/0004-public-sources-only.md`.
2. **Accessibility expectations.** WCAG level? Screen reader support?
   We default to WCAG AA across the app and captions on all reference
   videos (non-negotiable given the population). Confirm if the
   partner expects more.
3. **Vocabulary list sign-off.** Drafted by Claude against Lifeprint
   Units 1–6 and ASL-LEX 2.0 phonological annotation; committed to
   `docs/VOCABULARY.md`. Deaf instructor sign-off is a slice-2
   production-deployment requirement (ADR 0004), not pilot work.

---

## 6. What "the thoughtful extras" are (per hiring partner guide)

Two extras committed to slice 1, both architectural rather than
ornamental:

1. **Mastery decay visualization.** Show the learner their personal
   forgetting curve per sign — when retention is predicted to drop,
   when the next review is scheduled, why. Makes the spaced-retrieval
   theory visible to the learner. Pedagogical, not cosmetic.

2. **Self-paced exit.** When a sign reaches mastered state, the system
   actively removes it from rotation and congratulates the learner.
   Most apps fight to keep you in; this one fights to let you leave.
   This is the "kids get off the app" principle made literal.

Slice 2 candidates (talk about; build if time): parameter-aware
multi-head model for true model-derived hints; instructor analytics
dashboard surfacing aggregate failure modes.

---

## 7. What we are NOT building (and will defend in the presentation)

- No streaks, DAU/MAU metrics, push-notification engagement.
- No gamification beyond what mastery progression naturally provides.
- No teacher/admin portal (explicitly out of scope per brief, slice 2 candidate).
- No multi-language sign support.
- No sentence/phrase recognition.
- No upfront placement test (anti-pedagogical for true beginners).
- No "vibes-based" AI feedback (every output traces to thresholds and authored hints).

---

## 8. Tone and craft rules for any document Claude writes for this project

- No bullet salad. Prose-heavy in user-facing docs. Bullets only where lists are genuinely the right structure.
- No em dashes outside of natural English usage. The user is reading these.
- Every claim about learning science gets a citation or a PENDING CITATION marker.
- Use Superbuilders' language where it lands naturally: mastery, edu/acc, cognitive sovereignty, emergence engineering. Do not stuff these in artificially.
- Honesty about limitations is a feature, not a flaw. The validation report names failure modes by name.
- Code that ships should look like code we would be proud to have on a senior engineer's desk. No tutorial-copy code, no commented-out blocks left in.

---

## 9. File map for this project's documentation

| Path | Purpose |
|---|---|
| `claude/CLAUDE.md` | This file. Persistent context for Claude across sessions. |
| `claude/SESSION_LOG.md` | Append-only log of major decisions and unresolved threads, written at the end of each session. |
| `docs/ROADMAP.md` | What we are building and in what order. Slice 1 and slice 2. |
| `docs/ARCHITECTURE.md` | System architecture, data flow, component responsibilities. |
| `docs/PEDAGOGY.md` | The stated pedagogical theory, with full citations. |
| `docs/MODEL.md` | Model architecture, training procedure, dataset details, no-pretrained-pipeline evidence. |
| `docs/DATASET.md` | Data sources, collection protocol, cleaning pipeline, signer demographics, consent. |
| `docs/VALIDATION.md` | The validation report (populated after training). |
| `docs/EVAL_GATE.md` | Promotion criteria for any future model version. |
| `docs/PRIVACY.md` | Data handling, what stays local, what is logged, why. |
| `docs/decisions/NNNN-title.md` | Architecture Decision Records. One per non-trivial decision. |
| `docs/research/README.md` | Index of every paper, dataset, or source we cite or reference. |
| `docs/research/*.md` | Per-source notes; what we took from each, with full citation. |

---

## 10. How to update this file

Append, do not silently revise. When a decision changes, leave the old
row in Section 4 with a strikethrough or marked CHANGED, and add the
new decision below with the date and reason. We want the history
visible.

When a new constraint is discovered, add it to Section 5 as an open
question, then resolve it explicitly and move it to Section 4 once
answered.

---

*Last updated: initial drafting, before slice 1 implementation begins.*
