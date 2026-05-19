# ADR 0008: Slice-1 trains on public datasets only; recording tool becomes slice-2 framework

**Status:** Accepted
**Date:** 2026-05-19

---

## Context

The classifier specified in `docs/MODEL.md` §1 (2-layer BiLSTM over
MediaPipe keypoint sequences, per ADR 0006) needs training data —
clips of people signing each of the 96 signs in `docs/VOCABULARY.md`.

ADR 0006 set the per-sign clip target at **50–80 clips** (down from
~200 under the superseded Path B). Two sources were planned to meet
that target:

1. Public datasets — WLASL and MS-ASL clips that already exist.
2. Self-recorded supplement — `docs/ROADMAP.md` Phase 3 work item 4
   and `TODO.md` Phase 3c/3e, captured through a browser-based
   recording tool at `/admin/record` (specified in
   `docs/ARCHITECTURE.md` §2.2).

The recording tool implies a contributor role and an upload path to
the `asl-mastery-raw-training-data` R2 bucket, plus a consent flow
recorded in Postgres.

On **2026-05-19** we examined whether the pilot team has the inputs
the self-recording supplement requires. The honest answer:

- No one on the project team is a fluent ASL signer.
- No one on the project team has access to a fluent ASL signer for
  pilot data collection.
- Self-recorded clips produced by a non-signer would teach the
  classifier the wrong handshape, wrong location, or wrong movement
  for many signs. That is worse than less data — it is *misleading
  data*. The system's credibility (`claude/CLAUDE.md` §3 rule 3,
  "cite or do not claim") would not survive a validation report
  whose training set was authored by people who do not know the
  language.

ADR 0004 already established the parallel decision for vocabulary,
hints, and reference videos: pilot uses public sources only because
engaging a Deaf instructor without time and budget would be worse
than transparent self-limitation. The same logic applies — even
more strongly — to training-clip authorship: it would be worse to
train on bad data than to train on less data we did not author.

---

## Decision

For slice 1 (the pilot):

1. **All training data is drawn from public datasets** — WLASL and
   (when license-acceptance is resolved) MS-ASL. The cleaning
   pipeline (`docs/ARCHITECTURE.md` §2.5 stages 1–3) processes those
   clips into keypoint tensors and into the versioned training
   manifest.
2. **The recording tool is not built for the pilot.** `/admin/record`
   does not ship in slice 1. Phase 3c (admin-gated recording tool)
   and Phase 3e (self-recorded supplement) in `TODO.md` are moved
   to slice 2.
3. **The recording tool is preserved as a slice-2 framework
   commitment.** When ADR 0004's slice-2 instructor engagement
   happens, the Deaf ASL instructor uses the recording tool to
   produce the canonical reference videos and supplementary
   training clips in our green-box framing. The architectural
   specification in `docs/ARCHITECTURE.md` §2.2 stays in the
   repository as the slice-2 design target.
4. **Vocabulary may be trimmed if public-data coverage is too thin
   for any sign.** During Phase 3d (public dataset ingestion) and
   Phase 4 (training), each sign's actually-downloadable clip count
   is measured. Signs falling below a per-sign minimum clip count
   (initial floor: **15 downloadable clips per sign**, revisable
   during Phase 4) are dropped from the vocabulary. The total
   vocabulary count must stay at **≥ 75** to meet Brief
   Requirement 2; if dropping below-floor signs would take the
   count under 75, the floor itself is reduced (with the tradeoff
   disclosed in the validation report) rather than the count
   violated.

---

## What we considered and rejected

- **Self-record clips ourselves anyway.** Rejected. None of the
  project team is a fluent ASL signer. Self-recorded clips by a
  non-signer would teach the model wrong handshape, location, or
  movement for most signs. The eval-gate validation report would
  not be defensible — and the system's whole credibility argument
  rests on `docs/PEDAGOGY.md` §8's honest-disclosure contract.
  Training on knowingly-bad data violates `claude/CLAUDE.md` §3
  rule 6 (no vibes-based AI). Better to have less data we did not
  produce than more data we produced wrong.
- **Recruit a fluent ASL signer on the project team's network.**
  Rejected by the same logic that drove ADR 0004: engaging a Deaf
  ASL signer without the time and budget to do it fairly is worse
  than self-limitation with honest disclosure. ADR 0004 names this
  position explicitly; this ADR inherits it.
- **Train on self-recorded clips but with a quality-rating filter.**
  Rejected. A non-signer cannot self-assess sign correctness with
  any reliability — the whole point of expert review is that the
  non-expert cannot tell when they are wrong.
- **Pay an online signer (Fiverr-style) to record bulk clips.**
  Rejected. Same fairness argument as ADR 0004: paying a Deaf signer
  in a transactional context without the curatorial authority that
  ADR 0004 grants to slice-2 instructors is worse than transparent
  self-limitation, both ethically and as a credibility signal.

---

## What stays the same

- The system architecture (`docs/ARCHITECTURE.md` §1) is unchanged.
- The recording tool's design in `docs/ARCHITECTURE.md` §2.2 is
  unchanged; it is now framed as a slice-2 deliverable rather than
  a slice-1 deliverable.
- The classifier architecture, training procedure, and augmentation
  stack in `docs/MODEL.md` are unchanged. Keypoint tensors from
  public-dataset clips are valid inputs to the same training
  pipeline.
- The eval gate (`docs/EVAL_GATE.md`) is unchanged; its hard
  criteria still apply, including criterion 10 on MediaPipe
  detection success.
- The fairness and privacy commitments
  (`docs/DATASET.md` §5, `docs/PRIVACY.md`) are unchanged.
- ADR 0004 (no instructor for pilot) is unchanged; this ADR is the
  natural extension of it to training data.

---

## Mitigation

The pilot's training data credibility rests on three substitutes for
the missing self-recording supplement:

1. **WLASL and MS-ASL are research-grade datasets** produced by
   academic teams (Li et al. 2020; Joze & Koller 2018). Their clips
   are sourced from multiple signers per sign, often Deaf signers,
   from ASL-instructional video corpora and online sign communities.
   Using them is more credible than training on clips authored by a
   non-signer hearing engineer.
2. **The classifier consumes MediaPipe keypoint tensors, not raw
   video** (ADR 0006). The framing mismatch between public datasets'
   varied source video and our green-box capture is partly
   neutralized by MediaPipe's pose normalization. Skin tone,
   background, lighting, and clothing — the dorm-room overfitting
   axes that the superseded ADR 0001 was worried about — are
   already absorbed by the landmark extractor.
3. **The validation report names this scope explicitly.** Per
   `docs/PEDAGOGY.md` §8, the system claims only what it can defend.
   The slice-1 README and the recorded walkthrough video both
   disclose: trained on public datasets only, no self-recorded
   supplement, no Deaf-signer review of clip selection — and frame
   the architectural commitments (mastery, scheduler, hints, eval
   gate, local inference) as the substance under evaluation.

---

## Consequences

### `claude/CLAUDE.md` §4 — decisions table

A new row records this ADR: "Slice-1 training data — public datasets
only (WLASL, MS-ASL). Self-recording deferred to slice 2 per ADR
0008. Recording tool spec remains in ARCHITECTURE.md §2.2 as
slice-2 framework."

### `docs/ROADMAP.md` Phase 3

- Phase 3 work item 1 (recording tool) reworded to slice-2.
- Phase 3 work item 4 (self-recorded supplement) reworded to slice-2.
- Phase 3 exit criterion updated: ≥ 50–80 keypoint tensors per sign
  from **public sources alone**, with the per-sign clip-count
  filter applied and the resulting vocabulary count documented.

### `docs/DATASET.md`

- §1c per-sign target rewritten as "drawn from WLASL + MS-ASL only."
- §3 cleaning pipeline gains an explicit per-sign clip-count filter
  step before the keypoint-extraction stage; signs below the
  per-sign floor (15 clips default) are dropped and recorded in
  the manifest.
- §5 fairness disclosure gains a new bullet: training-set authorship
  is drawn from public corpora; this differs from the slice-2
  instructor-recorded path.

### `docs/MODEL.md`

- §7 ("No-pretrained-pipeline evidence") gains a clarifying note
  that the keypoint tensors fed to the classifier originate from
  public clips per this ADR; the classifier itself remains trained
  from scratch with Kaiming init, no `load_state_dict` calls.

### `TODO.md`

- Phase 3c (admin recording tool) moved to slice 2.
- Phase 3e (self-recorded supplement) moved to slice 2.
- Phase 3d (public dataset ingestion) becomes the primary data path
  and gains the per-sign filter step.
- The "admin flag" decision raised in Session 7's closing note is
  closed: no admin role is needed for slice 1.

### `docs/VOCABULARY.md`

- No edits to the table now (we have not yet measured downloadable
  WLASL coverage). When Phase 3d resolves per-sign downloadable
  counts and the per-sign filter is applied, the dropped signs are
  struck through in the table with the actual count and an
  "excluded under ADR 0008" annotation, and the final slice-1
  vocabulary count is recorded.

### `docs/ARCHITECTURE.md`

- §2.2 (Recording tool) gains a slice-2 header note pointing at
  this ADR; the rest of §2.2 is preserved as the slice-2 design
  target.

---

## Honesty about this pivot

This is the second pilot-scoping ADR (alongside ADR 0004) that says
"we considered doing the more credible thing, decided we cannot do
it well right now, and chose self-limitation with disclosure over
mediocrity with implied authority." The architectural commitments
of the system — mastery-based exit, spaced retrieval, the three-
layer hint system, local inference, the eval gate, fairness
reporting — are unchanged and remain the substance of what we are
demonstrating. The training data is acknowledged as a limitation
the production-deployment slice-2 work would address, not a
limitation we are papering over.

A reader of the final validation report sees the full pilot scope
honestly:

- Vocabulary curated by hearing engineers against ASL-LEX 2.0 and
  Lifeprint (ADR 0004).
- Reference videos sourced from WLASL / MS-ASL with attribution
  (ADR 0004).
- Hint copy authored from ASL-LEX 2.0 phonological data without
  Deaf-signer review (ADR 0004).
- Training clips drawn from WLASL / MS-ASL only (this ADR).
- Recording tool specified but not built (this ADR).
- Slice-2 commitments named explicitly: paid Deaf-instructor review
  of every item (ADR 0004) plus instructor-recorded canonical
  references plus instructor-recorded training supplement (this ADR
  closes the data-collection part of that loop).

---

## How this is verified

- The repository contains no code under `app/admin/record/` for
  slice 1. Adding such code is a slice-2 deliverable explicitly
  gated on the ADR 0004 instructor engagement.
- The training pipeline (Phase 3f and later) reads only from
  `asl-mastery-raw-training-data` paths populated by the public-
  dataset ingestion script. No clips authored by the project team
  reach the training set.
- The validation report records the exact WLASL + MS-ASL versions
  used and the per-sign clip count at training time.
- The README discloses public-data-only training in the same
  paragraph as ADR 0004's public-sources-only disclosure, and
  names this ADR by number.
