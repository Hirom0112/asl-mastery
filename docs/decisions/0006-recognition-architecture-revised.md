# ADR 0006: Recognition architecture revised — landmark-based, pretrained landmark extractor permitted

**Status:** **Superseded by [ADR 0010](./0010-reversal-of-adr-0006.md) on 2026-05-20.** The earlier permissive reading of brief Requirement 7 that authorized this ADR was withdrawn; architecture reverts to ADR 0001 Path B (end-to-end small 3D CNN trained from scratch on raw RGB). The body below is preserved as the historical record of the architecture under which v1.0.1, v2.0.0, and v2.1.0 shipped. Those three artifacts remain in `model_versions` with `is_active = false` per the migration applied in T1 of the ADR 0010 triage.
**Date:** 2026-05-19
**Supersedes:** ADR 0001 (recognition architecture, end-to-end small 3D CNN) — supersession itself superseded by ADR 0010, which reinstates ADR 0001.

---

## Context

ADR 0001 was written under a strict reading of brief Requirement 7
("No Pretrained Models"). That reading treated *every* pretrained
component — including general-purpose pretrained hand or pose
landmark detectors — as out of scope, and chose Path B (end-to-end
small 3D CNN trained from raw pixels) on that basis.

On **2026-05-19** an earlier clarification suggested pretrained
landmark detectors (MediaPipe Hands, MediaPipe Holistic, OpenPose,
BlazePose) were permitted under Requirement 7, while pretrained
ASL classifiers, sign-recognition pipelines, and ASL-specific
feature extractors remained forbidden. ADR 0001 was written under
the strict reading; this ADR captured the more permissive reading
and the architecture pivoted to landmark-based recognition on
that basis.

This clarification was withdrawn on 2026-05-20. The strict reading
of Requirement 7 now governs again. See ADR 0010 for the reversal.

---

## Decision

We pivot from Path B to a **landmark-based recognition architecture**:

1. **Landmark extraction.** MediaPipe Holistic in the browser. We
   consume a subset of its output:
   - 21 left-hand keypoints (x, y, z).
   - 21 right-hand keypoints (x, y, z).
   - A small subset of upper-body pose landmarks (shoulders, elbows,
     wrists, torso anchors). Exact subset confirmed during Phase 4.
   - Face landmarks deferred to slice 2 (non-manual markers).
2. **Temporal classifier, trained from scratch.** The actual ASL
   recognition logic is *our* classifier, trained entirely from
   scratch with Kaiming initialization. No pretrained sign classifier
   or ASL-specific weights are loaded at any point.
   - **Baseline:** 2-layer bidirectional LSTM over the keypoint
     sequence, target ≈ 200K parameters.
   - **Alternative:** small Transformer encoder (≈ 500K parameters),
     used if the BiLSTM underperforms during Phase 4 iteration.
3. **Output.** Softmax over the 75–100 classes from
   `docs/VOCABULARY.md`. Temperature-scaled for calibration. Per-sign
   confidence thresholds tuned for ≥90% precision on the "pass"
   decision, same as before.
4. **Inference deployment.** MediaPipe Holistic runs in the browser
   via MediaPipe Tasks Web / Web Solutions API. Our classifier is
   exported to ONNX and runs via ONNX Runtime Web. Total client
   model size target: **under 5 MB combined** (MediaPipe runtime +
   classifier).

---

## Rationale

- **Data efficiency.** Keypoint sequences are dramatically lower-
  dimensional than raw video tensors. The 7-clip WLASL minimum
  visible in `docs/VOCABULARY.md` becomes comfortable under landmark-
  based training; under Path B it was tight. Phase 3's per-sign clip
  target drops from ~200 to **~50–80**.
- **Fairness improvement.** The classifier sees keypoint coordinates,
  not pixels. It cannot learn skin tone as a spurious feature because
  skin tone is not present in its input. This does not eliminate
  fairness risk — MediaPipe's landmark detection itself can vary
  across demographics — but it removes a major axis of risk from the
  classifier itself.
- **Smaller browser footprint.** Classifier weights are well under
  1 MB. Combined with MediaPipe Tasks Web, the deployed bundle is
  under 5 MB.
- **Faster inference.** Per-clip inference target tightens from
  300 ms (Path B target) to **100 ms** end-to-end on a mid-range
  laptop.
- **Better substrate for parameter-aware hints.** The slice-2
  parameter-aware hint system from ADR 0001 (multi-head model
  predicting handshape, location, palm orientation, movement,
  non-manual markers) becomes substantially more achievable. The
  landmark sequences already encode handshape geometry, location
  in 3D space, palm orientation (via finger joint vectors), and
  movement (via the temporal axis). What was a vague slice-2
  aspiration is now a concrete slice-2 design target.
- **No loss of pedagogical theory.** Every claim in `docs/PEDAGOGY.md`
  remains correct. Mastery-based exit, spaced retrieval, targeted
  hints, eval-gated AI quality — all unchanged.

---

## What stays the same

The system-level architecture and every pedagogical and operational
commitment is unchanged:

- The five logical surfaces in `docs/ARCHITECTURE.md` §1.
- The mastery state machine and scheduler (`docs/ARCHITECTURE.md` §4).
- The three-layer hint system (`docs/ARCHITECTURE.md` §5).
- The eval gate hard and soft criteria (`docs/EVAL_GATE.md`).
- Signer-disjoint splits (`docs/DATASET.md` §4).
- Fairness commitments and per-demographic accuracy reporting
  (`docs/DATASET.md` §5, `docs/EVAL_GATE.md` §1).
- Privacy architecture: local inference, no frames leaving the
  device (`docs/PRIVACY.md` §1, §2).
- Deployment platform: Vercel + Supabase + Cloudflare R2 (ADR 0003).
- Public-sources-only sourcing for the pilot (ADR 0004).
- Classical CV libraries remain allowed (ADR 0005) for slice-2
  augmentation if useful, but are no longer load-bearing for
  slice-1 augmentation — landmarks normalize most of what
  classical CV was previously fighting.
- The full pedagogical theory in `docs/PEDAGOGY.md`.

---

## What's superseded

ADR 0001 in its entirety. The end-to-end small 3D CNN approach is
no longer the chosen recognition path. ADR 0001 is *not* deleted; it
remains in the repository as the historical record of the decision
we made under the strict reading of Requirement 7. ADR 0001 gains a
`Superseded by ADR 0006 on 2026-05-19` line below its `Status`
header so a reader who lands on the old ADR is routed forward.

---

## Consequences

### Data pipeline (`docs/DATASET.md`, `docs/ROADMAP.md` Phase 3)

- Per-sign clip target drops from ~200 to **~50–80**.
- Cleaning pipeline gains a stage: every cleaned clip is run through
  MediaPipe Holistic to extract keypoint sequences, which are saved
  alongside the source clip and become the actual training inputs.
- Raw clips are retained in R2 so reprocessing is possible when
  MediaPipe versions change.

### Training (`docs/MODEL.md`, `docs/ROADMAP.md` Phase 4)

- Training time per run drops from hours (Path B target) to minutes
  on a single small GPU; many runs feasible on CPU.
- GPU rental cost drops to near-zero.
- Iteration cadence increases substantially; the train-fail-collect-
  retrain loop tightens.

### Augmentation

- Pixel-level augmentations (color jitter, brightness, gamma,
  background swap via MOG2) are deprecated. They served Path B and
  are not meaningful on keypoint inputs.
- Landmark-level augmentations replace them: per-keypoint coordinate
  jitter (small Gaussian), temporal stretching (resample to vary
  apparent signing speed), occasional keypoint dropout (simulates
  partial occlusion), small in-plane rotation. Horizontal flip via
  x-coordinate negation remains, conditional on the sign's
  `flippable` flag in `docs/VOCABULARY.md`.

### Inference (`docs/ARCHITECTURE.md` §2.3)

- The per-attempt pipeline gains a MediaPipe extraction step between
  frame capture and classification. Keypoint sequences are what the
  classifier sees; raw frames never reach the classifier.
- End-to-end latency budget tightens from 1 second (Path B target)
  to roughly 600 ms; per-clip classifier inference under 100 ms.

### Hint system (`docs/ARCHITECTURE.md` §5)

- Slice-1 hint behavior is unchanged. The three layers and the
  confusion-pair lookup all remain.
- Slice-2 parameter-aware hints are promoted from "vague candidate"
  to "concrete design target." The landmark sequences are already
  in our possession at inference time, so a parameter-prediction
  head can be trained on the same data as the gloss classifier.

### Fairness reporting (`docs/EVAL_GATE.md`)

- A new hard criterion is added: MediaPipe landmark detection must
  succeed on ≥95% of test clips. Failed-detection clips do not
  count toward accuracy, but their failure rate is reported in
  every validation report. If MediaPipe fails to detect a
  learner's hands, that is a real failure mode and we will not
  paper over it.

### No-pretrained-pipeline evidence (`docs/MODEL.md` §7)

- The repository's audit surface for the no-pretrained-pipeline
  claim is the `training/classifier/` directory and `training/init.py`:
  no `load_state_dict` calls, no external classifier weight URLs.
  MediaPipe is imported as a library and used as a black-box
  landmark extractor, which the 2026-05-19 clarification permits.

---

## Rejected alternatives

- **Continue with Path B (end-to-end 3D CNN) despite the
  clarification.** Rejected. The clarification is from the brief
  author's organization; the costs of ignoring it (worse fairness,
  more data needed, larger model, slower iteration) are real and
  the project would be poorer for them.
- **Train our own landmark detector from scratch (the old Path A
  from ADR 0001).** Rejected. That was rejected in ADR 0001 as
  research-scale work and remains so. The clarification removes the
  need.
- **Use a pretrained image classifier (e.g., ImageNet ResNet) as a
  feature extractor.** Rejected. The clarification permits
  pretrained landmark detectors, not pretrained general-purpose
  feature extractors. Using a pretrained image backbone would
  exceed the clarification.

---

## Honesty about the pivot

This is a project-level course correction. ADR 0001 was a defensible
reading of the brief's text and remains in the repository as the
record of what we decided under that reading. The 2026-05-19
clarification is a more permissive reading from the brief's
authoring side. We align with the author's stated intent and pivot
cleanly. The validation report will name this pivot explicitly so a
reader of the final project sees both the original architectural
commitment and the clarified one.

---

## How this is verified

- The training entry point in the future repository will use only
  `torch.nn.init.kaiming_normal_` for classifier weights and will
  contain no `load_state_dict` calls and no external classifier
  weight URLs. A reviewer auditing the no-pretrained-pipeline claim
  reads `training/classifier/` and `training/init.py`.
- MediaPipe is loaded as a library at inference time; its presence
  in `package.json` and in the runtime bundle is the visible
  evidence that the clarification was applied. The clarification
  itself (this ADR) is the written authorization for that
  dependency.
- A note appears in the validation report identifying MediaPipe's
  exact version and the date the clarification was received.
