# ADR NNNN: <short title — the decision, not the question>

**Status:** Proposed | Accepted | Superseded by [ADR NNNN](./NNNN-…) | Withdrawn
**Date:** YYYY-MM-DD
**Supersedes:** [ADR NNNN](./NNNN-…) (optional)
**Reinstates:** [ADR NNNN](./NNNN-…) (optional)
**Relates to:** [ADR NNNN](./NNNN-…), [ADR NNNN](./NNNN-…) (optional)

---

## Context

What forces this decision now? What constraint, observation, or
upstream decision made this question arrive? Cite the brief
requirement, ADR, or incident that triggered the work. One to three
short paragraphs. No prose padding.

---

## Decision

The decision itself, stated as concretely as possible. If the
decision has parts, number them. A reader who reads only this
section should be able to act on the decision without reading
anything else in the ADR.

---

## Rejected alternatives

For each plausible alternative considered: one line naming it,
then one paragraph explaining why it was rejected. Rejected
alternatives are part of the decision record — they show the
decision was made deliberately, not by default.

---

## Consequences

What changes downstream? Touch every surface this decision moves —
data pipeline, training, inference, UI, validation, fairness,
docs, deploys, ADRs that need updating. Be specific about which
files, scripts, or operational commitments change.

If the decision has costs (timeline, accuracy, infrastructure,
contributor onboarding), name them. Decisions without costs are
suspicious.

---

## How this is verified

How does a reviewer (or future-you) confirm the decision was
actually executed? Greps that should return zero hits, files that
should exist, validation thresholds that should be met. The
verification surface should be a fixed, small set of artifacts.
