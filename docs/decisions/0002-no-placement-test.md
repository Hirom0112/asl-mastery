# ADR 0002: No placement or fluency test in the onboarding flow

**Status:** Accepted
**Date:** initial scoping

---

## Context

An earlier scoping conversation proposed a placement assessment on
first session — having the learner attempt 10 signs spanning easy /
medium / hard to seed the mastery state. The user flagged this as
anti-pedagogical and we re-examined.

The brief is explicit (Section 2: Target Users): "The application
should assume learners are new to ASL and need repeated practice,
clear feedback, and progress tracking. The app should not assume the
learner already understands sign linguistics, model confidence, or
computer vision limitations."

---

## Decision

The system does not give a placement or fluency test. The mastery
state machine treats every sign as `untouched` for every new account,
and the scheduler accelerates fast learners through the ordinary
practice loop.

---

## Rationale

- **Beginners cannot pass an upfront test on material they have not
  been taught.** Asking them to do so risks demoralizing them in their
  first 60 seconds with the product. Patrick's writing on EdTech is
  emphatic that we are not building yet another tool that punishes
  learners for not already knowing things.
- **The scheduler already handles fast learners.** Two consecutive
  passes in a session move a sign to `reviewing`. A third pass at
  ≥7-day interval moves it to `mastered`. A learner with prior
  exposure accelerates past those items within minutes of starting
  to practice, without ever experiencing an "assessment."
- **Removing the placement test removes a whole surface of UI
  complexity.** Simpler is better when simpler also serves pedagogy.

---

## Alternatives considered and rejected

- **Optional skip-ahead test.** Even optional, its presence signals to
  the learner that they ought to know something. For true beginners
  this is the wrong signal. Rejected.
- **Adaptive first session that introduces difficulty gradually.**
  This is what the scheduler does anyway; there is no need to brand
  it separately as "placement."

---

## Consequences

- Onboarding is shorter and gentler.
- A learner with substantial prior ASL exposure will spend a small
  number of minutes "demonstrating mastery" through the ordinary loop
  on items they already know. This is acceptable cost.
- The system's onboarding screens can focus on explaining how the
  system works (mastery, hints, exit-on-mastery), not on calibrating
  the learner.
