# ADR 0004: Pilot uses public sources only; no contracted Deaf ASL instructor

**Status:** Accepted
**Date:** 2026-05-19

---

## Context

Phase 0 of `docs/ROADMAP.md` originally called for contracting a Deaf
ASL instructor (paid, locally in Austin or via the Gallaudet alumni
network) to perform three functions:

1. Sign off on the 75–100 vocabulary list.
2. Author and validate hint copy referencing the five sign parameters
   (handshape, location, palm orientation, movement, non-manual
   markers).
3. Record canonical reference videos in our standardized green-box
   framing — these clips would serve both as the gold-standard
   reference videos shown to learners and as anchor positives in
   training.

For this pilot we do not have the resources to recruit, contract,
schedule, and fairly compensate a Deaf ASL instructor in the time
window available. Engaging an instructor without doing so on
terms that respect their expertise and time would be worse than not
engaging one at all — both ethically and as a credibility signal.

This decision is about the *pilot*. It is explicitly not about the
production deployment, which `docs/PEDAGOGY.md` Section 8 already
flags as requiring honest disclosure of where the system's claims do
and do not hold.

---

## Decision

For slice 1 (the pilot), all vocabulary, linguistic metadata, hint
content, and reference video are sourced from public, attributable
sources:

- **Vocabulary list and curriculum alignment:** drawn from the
  Lifeprint.com ASL 1 curriculum (Bill Vicars, deaf educator), Units
  1–6. Lifeprint is a widely-used free ASL 1 curriculum and is
  appropriate as the curriculum anchor for an ASL 1 testbed.
- **Phonological annotation (the five sign parameters):** taken from
  ASL-LEX 2.0 (Sehyr, Caselli, Cohen-Goldberg, Emmorey, 2021), an
  academically validated lexical database of ASL signs with
  phonological coding. Sign IDs and parameter codes are cited per
  sign in `docs/VOCABULARY.md`.
- **Hint copy:** authored against ASL-LEX 2.0 parameter codes,
  cross-checked against Lifeprint per-sign instructional notes.
  Hints reference the five sign parameters explicitly but are not
  validated by a fluent signer for slice 1.
- **Canonical reference video:** sourced from WLASL (Li et al., 2020)
  and MS-ASL (Joze & Koller, 2018) with per-clip attribution and
  license terms recorded in `docs/DATASET.md`. The "instructor-
  recorded canonical references" path described earlier in
  `docs/DATASET.md` Section 1b is deferred to slice 2.

---

## What we lose

This decision has real costs and we name them explicitly rather than
papering over them.

1. **Instructor-validated regional sign correctness.** Public datasets
   capture multiple signers from multiple regions; we cannot assert
   that the canonical reference for any given sign is the one a
   particular Deaf community in a particular region would teach. We
   inherit whatever regional choices the public datasets made.
2. **Authored hint copy validated by a fluent signer.** Hints are
   grounded in ASL-LEX 2.0 phonological data and Lifeprint
   instructional notes, but no Deaf signer has read them and confirmed
   that the wording is helpful rather than misleading. A hint that
   makes phonological sense on paper can still be unhelpful or wrong
   when a learner is actually trying to produce the sign.
3. **The "validated with input from a Deaf ASL instructor" line in
   the README.** That line is not honest for the pilot and will not
   appear. The README will be explicit about this scope limit and
   will name it as a slice-2 requirement for production deployment.

---

## Mitigation

The pilot's pedagogical and linguistic claims rest on the credibility
of the public sources used, plus honest disclosure of what those
sources do and do not authorize us to claim.

1. **ASL-LEX 2.0 is academically validated phonological data.** It is
   the standard lexical database for ASL phonological research, peer-
   reviewed, and widely cited. Using ASL-LEX 2.0 sign IDs and
   parameter codes for our vocabulary metadata is defensible.
2. **Lifeprint is a widely-used free ASL 1 curriculum authored by a
   deaf educator** (Bill Vicars). Aligning our vocabulary to Lifeprint
   Units 1–6 gives the pilot a defensible curriculum anchor without
   requiring us to invent one.
3. **WLASL and MS-ASL are research datasets with multiple signers
   per sign.** Using them for canonical reference video means
   reference clips are produced by signers (not by hearing engineers),
   are sourced from research-grade data, and are auditable per-clip.
4. **Honest disclosure of limits.** The README and the validation
   report both name this scope explicitly: pilot uses public sources,
   does not include Deaf instructor review of vocabulary or hints,
   does not have Deaf community sign-off, and production deployment
   requires the slice-2 work below. Credibility comes from naming the
   limit, not from hiding it.

---

## Slice 2 next steps (what production deployment requires that the pilot defers)

These are not nice-to-haves. They are the conditions under which the
system could be ethically deployed beyond a pilot. They are framed
that way in `docs/ROADMAP.md` updates and in the README.

1. **Engage a Deaf ASL instructor or fluent signer to review the
   vocabulary list, validate every hint, and re-record canonical
   reference videos in our standardized green-box framing.** Reviewer
   is paid at a fair professional rate. The review is real review —
   the instructor has authority to remove signs, rewrite hints, and
   replace reference clips — not symbolic endorsement.
2. **Build a Deaf community advisory feedback channel into the learner
   app** so native signers can flag inaccurate signs, unhelpful hints,
   or culturally inappropriate framing directly from the practice
   screen. Feedback is reviewed; substantive issues block the next
   model promotion (`docs/EVAL_GATE.md` extension).
3. **Compensate reviewers fairly.** Budget is allocated up front, not
   negotiated after the work. Reviewers are credited by name in the
   README Acknowledgments unless they decline.
4. **Re-issue the validation report with the instructor's name
   acknowledged.** The validation report ships with the same level
   of linguistic review that the system claims to provide. The
   "validated with input from a Deaf ASL instructor" line returns to
   the README only at this point, not before.

---

## Consequences

- `docs/DATASET.md` Section 1b ("Instructor-recorded canonical
  references") is rewritten to point at WLASL/MS-ASL with the slice-2
  candidate noted.
- `docs/PEDAGOGY.md` Section 5 hint-system reference to instructor-
  authored hints is reworded to ASL-LEX-2.0-grounded hints with the
  same slice-2 candidate noted.
- `claude/CLAUDE.md` Section 4 gets a new row recording this decision
  and pointing to this ADR. Section 5's open question on instructor
  contact is closed with a pointer here.
- `docs/ROADMAP.md` Phase 0 work item 2 (recruit ASL instructor) is
  marked deferred to slice 2; Phase 3 work item 3 (canonical reference
  recording) is rewritten as WLASL/MS-ASL selection plus slice-2
  re-recording candidate.
- The README's credibility framing is rebuilt around honest disclosure
  rather than instructor endorsement.

---

## Rejected alternatives

- **Recruit an instructor anyway, on a compressed timeline and
  reduced budget.** Rejected. Engaging a Deaf instructor without the
  time and budget to do it well is worse than not engaging one.
  Underpaying a Deaf consultant to validate a product that ships
  under a hearing engineer's name would be a real ethical problem and
  a worse credibility signal than transparent self-limitation.
- **Ship without naming the limit.** Rejected. The whole point of
  this project's credibility contract (see `claude/CLAUDE.md` Section
  3, rule 3 and rule 6, and `docs/PEDAGOGY.md` Section 8) is that the
  system says exactly what it can defend and nothing more.
- **Use only ASL-LEX 2.0 and skip Lifeprint.** Rejected. ASL-LEX is a
  lexical database; it is not a curriculum. We need a curriculum
  anchor for "ASL 1 college-course" framing, and Lifeprint is the
  obvious choice — free, widely used, authored by a deaf educator.
