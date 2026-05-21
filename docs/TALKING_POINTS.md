# Talking points

> Anticipated questions from the project's technical evaluators.
> One grounded paragraph per question, with the pointer to where
> the claim is backed.
>
> Use this as a personal cue card before the demo. Do not read it
> aloud. The point is that the answers are *grounded in the
> repository*, not improvised.

---

## 1. Why ASL?

ASL is the highest-leverage vocabulary domain we could pick for a
mastery-based-system testbed. The brief asked for it directly, but
the substantive reasons are: (a) it has well-cataloged phonological
parameters (ASL-LEX 2.0) that map cleanly to targeted feedback, (b)
it is a *motor* skill with measurable performance, where retrieval
and spaced practice have research-backed effects (Shea & Morgan 1979
contextual interference), (c) Deaf children's access to instruction
is the morally heaviest case we cite, and (d) it lets us prove
mastery measurement on something that isn't already saturated by
existing edtech. The architecture generalizes to any skill with a
recognizable performance signal; ASL is just our defensible pilot.
Citations: `docs/PEDAGOGY.md` and `claude/CLAUDE.md` §1.

## 2. Why "mastery, not engagement"?

Engagement is a means, never an end. An app with daily streaks and
zero retained learning is a failure on every dimension that matters.
Bloom 1984 showed that mastery learning alone produces ~1 sigma of
improvement; tutoring + mastery produces ~2 sigma. We can't deliver
1:1 tutoring at scale, but we can deliver the *substrate it lived
on* — immediate targeted feedback at the moment of attempt, spaced
retrieval scheduled by performance, and a defined exit state called
*mastered* that actively removes content from rotation. The opposite
of streaks. Citation: `docs/PEDAGOGY.md` §2 (VERIFIED, DOI
10.3102/0013189X013006004).

## 3. How do you know your feedback actually helps?

Two answers. **At the architecture level**: every hint we serve
traces to either a measurable model signal (confusion-pair lookup
from the validation confusion matrix) or pre-authored per-sign
content (ASL-LEX 2.0 phonological parameters). No vibes. **At the
empirical level**: the validation report tracks hint efficacy —
of attempts that received a confusion-pair hint and retried, what
fraction passed on the next attempt? Per confusion pair, per sign.
This is a slice-1 instrument; an A/B comparison between Layer B
(confusion-pair) and Layer C (generic) hints is in the "open
research questions" section of `docs/PEDAGOGY.md` §9.

## 4. Why no streaks, no notifications, no DAU?

Because we built the opposite of engagement infrastructure on
purpose. The mastery state machine has an explicit exit. When a sign
hits mastered, the system congratulates the learner, removes the
sign from active rotation, and schedules long-interval reviews to
confirm retention. There is no daily target, no notification
calling the learner back, no streak that punishes a missed day.
This is the "kids get off the app" principle made literal in the
data model. `docs/ARCHITECTURE.md` §4 + `claude/CLAUDE.md` §7.

## 5. How did you avoid pretrained models?

Two ADRs cover this. ADR 0001 was written under the strict reading
of brief Requirement 7 and chose an end-to-end 3D CNN trained from
scratch. ADR 0006 supersedes it after Gauntlet staff clarified on
2026-05-19 that Requirement 7 restricts pretrained ASL pipelines and
sign classifiers, not general-purpose landmark detectors. The
slice-1 architecture is: MediaPipe Holistic (pretrained, permitted)
→ small BiLSTM classifier trained from scratch with Kaiming init.
The audit surface is `training/classifier/init.py`: a module-level
assertion fires if anyone introduces a `load_state_dict` call. No
external classifier weight URLs anywhere in the training entry
point. `docs/MODEL.md` §7.

## 6. Why landmarks instead of pixels?

Three reasons. (a) **Data efficiency**: keypoint sequences are
dramatically lower-dimensional than raw video tensors. Per-sign
target dropped from ~200 clips (Path B) to ~50-80 (ADR 0006), and
in practice we ship the pilot with even fewer per sign without the
model collapsing because keypoint-level augmentation effectively
multiplies each clip by 5-10×. (b) **Fairness**: the classifier
literally cannot learn skin tone as a spurious feature because skin
tone is not in its input. MediaPipe's own landmark-detection accuracy
varies by demographic, so the validation report breaks out MediaPipe
detection-success rate per demographic alongside classifier accuracy
— honest disclosure. (c) **Browser footprint**: combined client
bundle is under 5 MB. `docs/MODEL.md` §1 + ADR 0006.

## 7. What's the eval gate, concretely?

`docs/EVAL_GATE.md` §1 names ten hard criteria that a model artifact
must pass before promotion. Examples: top-1 ≥ 85%, no sign below
60% accuracy, no per-demographic gap > 10 percentage points, ≥ 90%
precision at the per-sign confidence threshold, latency p95 ≤ 600ms,
no regression > 3pp vs current production, MediaPipe detection
success ≥ 95%. The criteria are enforced in code by
`scripts/check_eval_gate.py`, which exits non-zero on any miss. A
GitHub Actions workflow at `.github/workflows/eval-gate.yml` runs
the enforcer on every PR that touches `docs/validation/`. So
"no vibes-based AI" is a property of the deployment infrastructure,
not a wish.

## 8. What is the failure mode of this system?

Three real ones. **(a) Public training data is uneven.** Our 96
signs draw from WLASL + MS-ASL only (ADR 0008); some signs have far
fewer clips than others, and the per-sign floor flexes downward
before we drop below 75 total signs. The validation report names
the per-sign clip counts and any floor reduction explicitly. **(b)
Slice-1 hints are not Deaf-reviewed.** They are authored from
ASL-LEX 2.0 phonological data and Lifeprint instructional notes
(ADR 0004). The reviewer will see hint copy that is plausible and
parameter-grounded but not validated by a fluent signer. Slice-2 is
the paid Deaf-instructor engagement that addresses this. **(c)
MediaPipe per-demographic detection-success varies.** We delegated
landmark extraction to a third-party model whose own fairness is
not under our control. We report it per demographic in the
validation report and treat it as a real failure mode, not a footnote.

## 9. Why no instructor for slice 1?

ADR 0004. Engaging a Deaf instructor without time and budget to do
it fairly would be worse than transparent self-limitation. We do
not have the resources to recruit, contract, schedule, and fairly
compensate one for the pilot. We chose to ship with public-sources-
only sourcing and explicit disclosure rather than ship with a
Deaf consultant whose review we couldn't honor. The slice-2
production-deployment plan in ADR 0004 names the exact engagement
the system would require: paid review of every vocabulary item and
every hint, plus re-recording all 96 canonical references in
green-box framing, plus an in-app community feedback channel.

## 10. Why not record clips yourself?

ADR 0008. Nobody on the project team is a fluent ASL signer.
Training the classifier on clips authored by non-signers would
teach wrong handshape / location / movement — worse than less data,
it would be *misleading data*. Self-recording is removed from slice
1; the recording tool's specification in `ARCHITECTURE.md` §2.2 is
preserved as the slice-2 framework target for the ADR-0004
instructor engagement.

## 11. What would you do with three more months?

Slice 2 in priority order: (1) Deaf instructor engagement —
vocabulary review, hint validation, canonical-reference re-recording
in our standardized framing. (2) Parameter-aware hint system —
second classifier head predicting the five sign parameters
(handshape, location, palm orientation, movement, non-manual
markers), so hints can name *which parameter was wrong*, not just
*which sign was predicted*. ADR 0006 promoted this from vague
aspiration to concrete slice-2 target. (3) In-app feedback channel
for Deaf community to flag inaccurate signs or unhelpful hints
directly from practice. (4) Automated model promotion in CI (slice
1 has human eyes on the eval gate). (5) Custom domain replacing
`asl-mastery.vercel.app`.

## 12. How does this scale to a billion kids?

By being substrate, not product. The pieces that generalize: the
mastery state machine, the spaced-retrieval scheduler, the
three-layer hint architecture, the eval gate, local inference,
self-paced exit. The pieces that don't: the specific recognition
model, the specific vocabulary, the specific hint copy. The
generalization story is "if your skill has a recognizable
performance signal and a defined parameter set, this architecture
plugs in." ASL is a particularly good showcase because it is one of
the few skills where the architectural commitments — mastery, local
inference, instructor-validated content, demographic-fairness
reporting — are simultaneously *necessary* (the brief) and
*verifiable* (the eval gate, the validation report, the ADR trail).

## 13. What if the model is wrong about a learner's sign?

Three answers. **The user-facing path**: every fail panel has a
"I think I did this right" feedback button. Clicking it flips
`attempts.learner_disagreed = true` via an RLS-narrow update
policy. The flagged attempts feed a manual-review queue that
identifies false-negative-heavy signs. `docs/ARCHITECTURE.md`
§2.3 + the trigger in `supabase/migrations/...learner_disagreed.sql`.
**The model path**: per-sign confidence thresholds are tuned for
≥ 90% precision on the "pass" decision — we tolerate false
negatives more than false positives because telling a learner
they're wrong when they're right is worse than the reverse, and
because the next-attempt loop is cheap. **The fairness path**:
per-user pass-rate alerting (described in `EVAL_GATE.md`) surfaces
users whose pass rate stays below 30% over 50+ attempts, the most
likely cause being demographic mismatch in training data. The
response is targeted data collection, not threshold gymnastics.

## 14. How is privacy handled?

Architectural, not policy. (a) Inference runs in the browser; the
model is downloaded once and runs on the learner's device. (b) The
practice canvas carries the `no-track-canvas` class so PostHog's
autocapture (Phase 7) cannot pick up frames. (c) Sentry session
replay is configured with `maskAllInputs: true, blockAllMedia: true`.
(d) The `raw-training-data/` R2 bucket is private; only the training
pipeline (server-side, with a credential not present in the client)
can read from it. (e) Contributors sign a consent form before any
recording; consent records are stored in Postgres. (f) Account
deletion cascades through users → attempts → mastery_state via FK
constraints, and the Supabase Admin API is invoked from a server
action that requires the caller's session. (g) Anonymous accounts
inactive for 30 days are deleted by the
`cleanup_inactive_anonymous_users()` scheduled function (ADR 0007).
`docs/PRIVACY.md` + `supabase/migrations/*.sql`.

## 15. What's the honest scope of this pilot?

Three explicit limits, named in the README and in the validation
report: (1) **No Deaf instructor for slice 1** (ADR 0004) —
vocabulary, hints, and reference videos are sourced from public
corpora. (2) **No self-recorded training data** (ADR 0008) — slice-1
training set is WLASL + MS-ASL only. (3) **MediaPipe per-demographic
detection variance** — we report it but do not control it.
Everything else — the mastery model, the scheduler, the hint
layering, the eval gate, local inference, the privacy architecture
— is the substance we claim to demonstrate. The credibility argument
is: we built the pieces of the system that are load-bearing for
proving the pedagogical theory at scale. We did not build the
pieces that require expert authorship at pilot scope.

---

## Appendix: short citations to have ready

- Bloom 1984, 2-sigma problem: DOI 10.3102/0013189X013006004. VERIFIED.
- Hattie & Timperley 2007, feedback meta-analysis: DOI 10.3102/003465430298487. VERIFIED. Wisniewski Zierer Hattie 2019 replication: motor-skill outcomes show larger effect sizes.
- Karpicke & Roediger 2008, testing effect on foreign vocabulary: Science 319, 966-968. VERIFIED.
- Cepeda Pashler Vul Wixted Rohrer 2006, distributed practice meta-analysis: DOI 10.1037/0033-2909.132.3.354. VERIFIED.
- Shea & Morgan 1979, contextual interference for motor learning: J Exp Psych: Human Learning and Memory 5(2), 179-187. VERIFIED.
- Sweller 1988, Cognitive Load Theory: Cog Sci 12(2), 257-285. VERIFIED.
- Sevcikova Sehyr Caselli Cohen-Goldberg Emmorey 2021, ASL-LEX 2.0: J Deaf Studies and Deaf Education 26(2), 263-277.
- Li Rodriguez Opazo Yu Li 2020, WLASL: WACV 2020.
- Joze & Koller 2018, MS-ASL: arXiv 1812.01053 / BMVC 2019.
