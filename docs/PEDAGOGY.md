# Pedagogical Theory

> The stated theory of learning that this system is built to test. This
> document is read by Patrick and Frank to decide whether the engineer
> who built this can think pedagogically, not just write code.
>
> **Citation discipline.** Every claim below is tagged with one of:
> - `[VERIFIED]` — citation has been checked against the original published source. Paper exists, claim is accurately attributed, key nuances captured.
> - `[NOTED]` — the source is historically referenced rather than being load-bearing for our argument; abstract has been checked but full paper has not been read.
> - `[PENDING]` — citation is from Claude's training knowledge and has *not* been verified against the original source yet. **Must be verified before this doc ships.**
> - `[NEEDS REWORD]` — the underlying point is sound but the citation does not actually support it; rewrite or drop.
>
> Status as of current revision: most major citations verified via the
> primary literature; one remains `[PENDING]` (Anderson Corbett 1995),
> treated as supporting context not load-bearing.

---

## 1. The theory in one paragraph

Mastery of a motor skill requires three things that conventional ASL 1
instruction cannot deliver at scale: immediate feedback at the moment
of attempt, targeted error correction that identifies the specific
failure mode, and spaced retrieval practice scheduled by the learner's
actual performance rather than by the calendar. This system delivers
all three for ASL beginner vocabulary, and measures whether learners
reach a defined mastery state in less time than the status quo. The
unit of progress is the sign-mastered, not the minute-spent.

---

## 2. Why "mastery, not engagement"

Conventional EdTech has converged on engagement metrics — daily active
users, time-on-app, streaks, session length — because these are easy
to measure and easy to optimize. Engagement is not learning. An app
with high engagement and zero retained learning is a failure on every
dimension that matters. Engagement is a means, never an end.

`[VERIFIED]` Bloom's "2 sigma problem" (Bloom, 1984) reported that
students taught one-to-one with mastery learning techniques performed
two standard deviations above students in conventional classroom
instruction — that is, the average tutored student outperformed 98%
of the control class. Bloom posed this as a "problem" because human
one-on-one tutoring does not scale: the question is whether group
instruction can be made as effective. Important nuance: Bloom's 1984
paper compared three conditions — conventional instruction, mastery
learning in a group setting, and one-to-one tutoring with mastery
learning. Mastery learning alone (without one-to-one tutoring)
produced about a 1-sigma effect; the full 2-sigma effect required
both. Subsequent researchers have reproduced large effect sizes for
mastery learning, though typically not the full 2 sigmas of Bloom's
original. *Citation: Bloom, B. S. (1984). "The 2 Sigma Problem: The
Search for Methods of Group Instruction as Effective as One-to-One
Tutoring." Educational Researcher, 13(6), 4–16. DOI:
10.3102/0013189X013006004.*

The promise of AI-mediated mastery learning is to recover some of
the 2-sigma effect by automating the personal correction loop that
one-on-one tutors provide. We are not claiming our system achieves
2 sigmas; we are claiming we are building on the substrate that
research identifies as where the gains live.

`[VERIFIED]` Hattie & Timperley's meta-synthesis on feedback (2007)
reported a high effect of feedback on student achievement overall
(d in the 0.70–0.79 range) but with substantial variance across
feedback types. Effect sizes vary dramatically by feedback content:
in Kluger & DeNisi's earlier review (which Hattie & Timperley cite
extensively), task-level feedback indicating "you're correct"
produced d = 0.43, while feedback explaining changes from previous
trials produced d = 0.55. Feedback designed to discourage the
student produced *negative* effects (d = -0.14). A 2019 replication
meta-analysis (Wisniewski, Zierer, Hattie) found a more conservative
overall d = 0.48, again with strong moderation by feedback type —
**and the 2019 replication explicitly found higher effects for
cognitive and motor skill outcomes than for motivational or behavioral
outcomes**, which is directly relevant to ASL as a motor skill.
*Citation: Hattie, J., & Timperley, H. (2007). "The Power of Feedback."
Review of Educational Research, 77(1), 81–112. DOI:
10.3102/003465430298487. Follow-up: Wisniewski, B., Zierer, K., &
Hattie, J. (2019). "The Power of Feedback Revisited: A Meta-Analysis
of Educational Feedback Research." Frontiers in Psychology, 10, 3087.*

These two findings underwrite the decision to optimize the system for
fast, specific, in-the-moment correction rather than for any
engagement metric.

---

## 3. Why spaced retrieval

`[NOTED]` Ebbinghaus' 19th-century work on the forgetting curve
established that retention of newly learned material decays
predictably without retrieval, and that retrieval at increasing
intervals re-stabilizes the memory trace. A modern partial
replication by Murre & Dros (2015) reproduced Ebbinghaus' original
self-experiment and found broadly consistent decay curves. We treat
Ebbinghaus as historical grounding rather than a load-bearing
citation; the modern spacing-effect literature does the substantive
work. *Citation note: Murre, J. M. J., & Dros, J. (2015).
"Replication and Analysis of Ebbinghaus' Forgetting Curve." PLOS ONE,
10(7), e0120644. Verified to exist; we have not yet read the full
paper, only its abstract and citations elsewhere.*

`[VERIFIED]` Cepeda, Pashler, Vul, Wixted, & Rohrer (2006)
meta-analyzed 839 assessments of the distributed practice effect
across 317 experiments in 184 articles. Their core finding is more
nuanced than "spaced beats massed": the optimal inter-study interval
depends on the retention interval — the longer you want the
information retained, the longer the optimal gap between study
episodes. This directly informs our scheduler design: review
intervals grow over time as a sign approaches mastery, rather than
being fixed. *Citation: Cepeda, N. J., Pashler, H., Vul, E., Wixted,
J. T., & Rohrer, D. (2006). "Distributed practice in verbal recall
tasks: A review and quantitative synthesis." Psychological Bulletin,
132(3), 354–380. DOI: 10.1037/0033-2909.132.3.354.*

`[VERIFIED]` Karpicke & Roediger (2008) tested students on foreign
language vocabulary learning under four conditions: repeated study
and testing (the standard paradigm), repeated study with testing
dropped after first correct, repeated testing with study dropped, and
both dropped. The finding: **repeated study after first correct had
no effect on delayed recall, but repeated testing produced a large
positive effect**. This is the cleanest available evidence that
retrieval practice — not re-study — is the active ingredient in
durable learning. Especially relevant to us because the study was on
foreign vocabulary acquisition, which is the closest classical
analog to beginner ASL vocabulary. *Citation: Karpicke, J. D., &
Roediger, H. L. (2008). "The critical importance of retrieval for
learning." Science, 319(5865), 966–968.*

Our system favors retrieval (the learner signs the sign) over
re-study (the learner watches the reference video) for exactly this
reason. The reference video is available on demand but is not the
primary loop.

`[VERIFIED]` Motor-skill learning has a partly different profile from
declarative recall. Shea & Morgan (1979) introduced the contextual
interference effect to motor learning: subjects who learned three
motor tasks in *random* (interleaved) order performed worse during
acquisition but showed substantially better retention and transfer
than subjects who learned in *blocked* order. The effect has been
extensively replicated and reviewed (see Brady 2004 meta-analysis;
Magill & Hall 1990 review). *Citation: Shea, J. B., & Morgan, R. L.
(1979). "Contextual interference effects on the acquisition,
retention, and transfer of a motor skill." Journal of Experimental
Psychology: Human Learning and Memory, 5(2), 179–187.*

This affects the scheduler directly: **within a session, we
interleave practice across multiple signs rather than drilling one
sign repeatedly**, even though this produces slightly worse in-session
performance than blocked practice would. The cost is paid for in
better long-term retention, which is the actual goal.

The SM-2 algorithm — popularized by Anki — operationalizes spaced
retrieval for declarative material. Our scheduler is a modification of
SM-2 with motor-learning-specific tuning: more aggressive interval
reset on failure (because reinforcing a wrong motor pattern is
specifically harmful), and a mastery exit condition requiring
multiple successful retrievals at intervals ≥ 7 days. These
modifications are research-informed but ultimately empirical; we will
tune the constants against our own observed time-to-mastery data.

---

## 4. Why targeted error correction over "incorrect"

`[VERIFIED]` Cognitive Load Theory was introduced by Sweller (1988)
and elaborated through the 1990s and 2000s. The 1988 paper
established that working memory is severely limited and that
problem-solving methods consuming working memory for non-learning
purposes (e.g., means-ends analysis) impair learning. The three-
component model — intrinsic load (inherent task difficulty),
extraneous load (load from how material is presented), and germane
load (load that builds schemas, which is what produces learning) —
was developed through subsequent work, most fully articulated in
Sweller (2010). The practical implication for instructional design:
**design materials to minimize extraneous load so that more of the
learner's working memory capacity can go to germane load, which is
what builds durable knowledge**. Generic feedback ("wrong, try
again") forces the learner to spend extraneous load identifying what
went wrong, displacing germane load. Targeted feedback ("your
handshape was correct but your starting location was too high")
reduces extraneous load and channels effort into productive
correction. *Citations: Sweller, J. (1988). "Cognitive load during
problem solving: Effects on learning." Cognitive Science, 12(2),
257–285. Sweller, J. (2010). "Element interactivity and intrinsic,
extraneous, and germane cognitive load." Educational Psychology
Review, 22(2), 123–138. DOI: 10.1007/s10648-010-9128-5.*

`[PENDING]` Anderson's ACT-R framework and the broader intelligent
tutoring systems literature (Anderson, Corbett, Koedinger, Pelletier,
1995) demonstrated that ITS systems delivering targeted, immediate
correction can produce learning gains comparable to one-on-one human
tutoring. — *Source to verify: Anderson, J. R., Corbett, A. T.,
Koedinger, K. R., & Pelletier, R. (1995). "Cognitive tutors: Lessons
learned." The Journal of the Learning Sciences, 4(2), 167–207.
Treated as supporting context rather than load-bearing; verify
before shipping if used in the README or presentation.*

These motivate the system's three-layer hint architecture:

- **Layer A: pre-attempt priming.** Reduce extraneous load *before*
  the attempt by directing attention to the five sign parameters.
- **Layer B: confusion-pair-aware feedback on fail.** When the model
  predicts a different sign with reasonable confidence, the system
  knows likely *which* part of the sign was wrong and can correct
  specifically.
- **Layer C: generic per-sign failure hint.** Fallback when the model
  cannot localize the failure. Authored to direct attention toward
  the most common failure mode for that sign.

---

## 5. Why mastery-based exit and self-paced removal

The two-hour-learning model that Superbuilders builds toward depends
on learners being able to *finish* — not just plateau. A learning
system that holds you longer than necessary is broken on the same
axis as a system that releases you before mastery.

Our mastery state machine has an explicit exit. When a sign reaches
mastered (three successful retrievals at intervals ≥ 7 days), it is
*actively removed* from active rotation. The learner is told. The
sign returns only on a long review schedule designed to confirm
retention, not to consume attention. This is the "kids get off the
app" principle made literal in the data model.

The architectural complement: there is no streak counter, no daily-
target nudge, no notification asking the learner to come back. The
only thing that pulls the learner back is the natural review schedule
the SM-2 modification produces, and that review can be skipped
without penalty.

---

## 6. Why this system does not give a placement test

A placement or fluency assessment at the start of a beginner program
fails for two compounding reasons. First, the brief specifies that the
learner is new to ASL — assessing them on material they have not yet
been taught is at best uninformative and at worst demoralizing.
Second, the mastery state machine already handles the case of a
learner who comes in with prior exposure: they pass a sign on first
attempt, the scheduler promotes the item to reviewing on the second
attempt in-session, and the sign goes to long-interval review within
minutes. The learner accelerates past known material *through the
ordinary practice loop*, not through a separate assessment surface.

This is the architecture answering to the pedagogy: the simpler,
gentler entry point achieves the same diagnostic function as a
placement test, without the confidence cost.

---

## 7. The unit of measurement

The system's primary outcome metric is **time-to-mastery per sign**,
where:

- Mastery is defined by the explicit state machine (three retrievals
  at intervals ≥ 7 days, last attempt correct).
- Time is wall-clock time from first introduction to mastery state
  achieved.

Secondary metrics:

- **Forgetting rate.** When the learner returns after N days, do they
  retain mastery? Measured directly by review-attempt outcomes.
- **Hint efficacy.** Of attempts that received a confusion-pair hint
  and were retried, what fraction passed on the next attempt? Per
  confusion pair.
- **Equity metrics.** Time-to-mastery distribution across demographic
  buckets (where consented data permits). A pedagogically credible
  system cannot have systematically different time-to-mastery for
  learners with different skin tones because the model fails them more
  often.

Engagement metrics — DAUs, session length, return rate — are
explicitly *not* outcome metrics. They are operational metrics
(useful for capacity planning) and fairness diagnostics (a sudden
drop in return rate for one demographic is a signal to investigate
the model). They are not goals.

---

## 8. What we are *not* claiming

We are not claiming this system will produce learning outcomes
equivalent to one-on-one human tutoring. We are claiming that the
*architectural commitments* — local inference, mastery-based exit,
spaced retrieval scheduling, targeted feedback, eval-gated AI quality
— are the right substrate on which to build toward that outcome.

We are not claiming the underlying recognition model is research-
grade. Per the brief, it is pilot-grade under documented controlled
conditions. The pedagogical theory does not require research-grade
recognition; it requires recognition reliable enough that the feedback
loop is trustworthy under documented conditions, with honest reporting
of failure modes.

We are not claiming a placement test is universally wrong. We are
claiming it is wrong for *true beginners* in a system whose scheduler
already accelerates fast learners through the ordinary loop.

---

## 9. Open questions we will study, not answer

These are positions a slice-2 follow-up would investigate; we name
them explicitly so the limits of the slice-1 theory are visible:

- Does the confusion-pair hint actually improve next-attempt pass rate
  relative to a generic hint? (Answerable with the A/B framework
  described in `EVAL_GATE.md`.)
- Does the long-interval mastery exit retain better than a fixed
  schedule? (Answerable with forgetting-rate data over time.)
- Does priming with the five sign parameters before attempt improve
  first-attempt accuracy versus no priming? (Answerable with the same
  A/B framework, applied to Layer A.)
- Do learners with different prior exposure to signed languages (e.g.
  Black ASL, signed exact English, family-only home signs) accelerate
  through the system differently?

These are real research questions. Surfacing them as questions rather
than pretending to have answered them is part of the credibility
contract.

---

## 10. Citation status summary

| Source | Status | Notes |
|---|---|---|
| Bloom (1984) — 2-sigma problem | VERIFIED | DOI 10.3102/0013189X013006004. Important nuance: full 2-sigma effect requires tutoring + mastery learning; mastery learning alone produced ~1 sigma. Captured in the doc. |
| Hattie & Timperley (2007) — feedback | VERIFIED | DOI 10.3102/003465430298487. Effect size 0.70-0.79 overall in original; 2019 replication d = 0.48 with strong moderation by type. Replication finding that feedback effect is larger for motor skills is directly load-bearing for our ASL application. |
| Karpicke & Roediger (2008) — testing effect | VERIFIED | Science 319, 966-968. Study used foreign vocabulary learning, which is the closest analog to beginner ASL vocabulary. |
| Cepeda et al. (2006) — distributed practice | VERIFIED | Psych Bulletin 132(3), 354-380. 839 assessments, 317 experiments. Key finding: optimal interval depends on retention interval; informs growing-interval scheduler design. |
| Shea & Morgan (1979) — contextual interference | VERIFIED | J Exp Psych: Human Learning and Memory 5(2), 179-187. Load-bearing for our scheduler: interleave signs within session, do not block. |
| Sweller (1988, 2010) — cognitive load theory | VERIFIED | Both citations confirmed. Three-component model fully articulated in 2010 paper, not 1988. |
| Ebbinghaus + Murre & Dros (2015) | NOTED | Historical reference; modern spacing-effect work does the substantive citation work. Murre & Dros abstract checked, full paper not read. |
| Anderson Corbett Koedinger Pelletier (1995) | PENDING | Supporting context, not load-bearing. Verify if used in README or presentation. |

The verification pass was conducted during initial drafting via web
search against the primary literature, with cross-checks against
multiple secondary sources for the key effect-size claims.
