# Eval Gate

> ⚠️ **PARTIALLY SUPERSEDED — read [`STATUS.md`](../STATUS.md) first.**
> Most hard criteria below still apply (top-1 accuracy floor,
> per-Fitzpatrick gap, confidence calibration, latency, no-regression).
> What changed under [ADR 0011](decisions/0011-landmarks-and-templates-pivot.md):
> - The former per-frame landmark-detection-success criterion (10)
>   was removed and stays removed.
> - The "model" being gated is no longer a single classifier — it is
>   the combined four-detector + sign-matcher pipeline. Promotion is
>   gated on the **pipeline's end-to-end** metrics, not per-component.
> - Per-component sub-criteria (hand-detector AP@IoU=0.5 ≥ 0.85,
>   hand-landmark mean per-keypoint pixel error < 8 px @ 224, etc.)
>   are tracked in `docs/VOCABULARY_TRAINER_ROADMAP.md` per phase and
>   are not promotion gates themselves.
> A v2 of this gate (the "eval gate ADR" referenced from STATUS.md)
> will codify the pipeline-level criteria and supersede this file.

> The criteria a model version must meet before it can replace the
> currently-active one. Enforced by humans for slice 1, by CI for
> slice 2.

This document is the contract between the AI quality of the system and
its users. A model that does not meet these criteria does not ship.

---

## 1. Hard criteria (no merge / no promotion if violated)

A candidate model version must satisfy *all* of the following on the
held-out test set:

1. **Overall top-1 accuracy ≥ 85%.** This is the **conservative floor**, not the expected achievement. Under a landmark-based architecture, classifiers at this vocabulary scale typically achieve 90–95% accuracy. The 85% gate is held as the honest minimum we will not ship below; the validation report surfaces the actual achieved accuracy.
2. **No sign with test accuracy below 60%.** If a sign cannot clear
   60%, either collect more data for it or remove it from the
   vocabulary; do not ship a sign the model cannot recognize.
3. **No Fitzpatrick bucket with a per-demographic accuracy gap > 10
   percentage points** versus the best-performing bucket. Where consent
   data is sufficient to compute the buckets.
4. **Per-sign confidence threshold yields ≥ 90% precision** on the
   "pass" decision on validation set.
5. **Latency p95 on a Chromebook-class device ≤ 600 ms** end-to-end
   per attempt, measured on the deployed bundle. (Tightened from an earlier 1-second target.)
6. **No regression > 3 percentage points** on overall test accuracy
   versus the currently-active model.
7. **Confidence calibration**: reliability diagram shows expected
   calibration error ≤ 0.05 after temperature scaling.
8. **Hint coverage**: the confusion-pair hint catalog covers every
   confusion pair occurring more than once in the validation confusion
   matrix; remaining failures fall back to authored generic hints.
9. **No-pretrained-pipeline evidence intact:** every CV model is
   Kaiming-initialized and trained from scratch, with no `load_state_dict` reading foreign weights and no external weight URL. **No pretrained vision components are imported anywhere** in the inference path or the training pipeline, and no pretrained-model packages are declared in `package.json` or `training/requirements.txt`. (The lockfile carries one unused `@mediapipe/tasks-vision` from `@react-three/drei`'s face-tracking helpers — never imported or installed.) Classical (non-learned) CV remains permitted for training-time augmentation under [ADR 0005](./decisions/0005-classical-cv-allowed.md).

(A former criterion 10 — a per-frame landmark-detection-success target tied to a pretrained landmark detector — was dropped, since the pipeline uses no such detector. It is renumbered out.)

---

## 2. Soft criteria (review and discuss, not automatic block)

These warrant human attention but do not automatically block
promotion. The reviewer either justifies the regression in writing or
sends the candidate back.

1. **Latency p95 regression > 200 ms** versus current production.
2. **Hint efficacy drop > 5 percentage points** on the top-10
   confusion pairs (next-attempt pass rate after hint shown).
3. **Per-sign accuracy regression > 5 percentage points** on any sign.

---

## 3. The validation report

Generated automatically by the training pipeline at the end of every
training run. Stored alongside the model artifact in R2 and committed
as `docs/validation/v<N>.md`.

Required sections:

- Dataset version hash; total clip count; clip distribution per sign.
- Signer demographic breakdown.
- Split methodology (signer-disjoint) and the signer-to-split mapping.
- Overall accuracy (top-1, top-3, top-5).
- Per-sign accuracy with confidence intervals.
- Per-condition accuracy: lighting × background × sleeve length.
- Per-demographic accuracy: Fitzpatrick bucket, handedness.
- Full confusion matrix.
- Reliability diagram for calibration.
- Per-sign confidence threshold table.
- Hint coverage table with confusion pairs.
- Latency benchmarks: float32 vs int8, WebGPU vs WebGL vs WASM, on
  representative hardware.
- Known failure modes with example clips that demonstrate them.
- Diff against previous version: which signs improved, which
  regressed, by how much.

---

## 4. The promotion procedure

Slice 1 (manual):

1. Engineer runs training. Validation report is produced.
2. Engineer reviews report against this document's hard and soft
   criteria.
3. If all hard criteria pass, engineer opens a PR that updates the
   `model_versions` table seed with the new version row and sets
   `is_active = true`.
4. PR description embeds the validation report diff against the
   currently-active version.
5. A second engineer reviews the PR. Promotion happens on merge.
6. Client cache invalidation is triggered post-merge so learners
   download the new model on their next session.

Slice 2 (CI-enforced):

The PR pipeline runs the validation harness, parses the JSON output,
and refuses to merge unless hard criteria pass. Soft criteria warnings
are surfaced as PR comments.

---

## 5. Rollback

If a promoted model exhibits production issues (sudden drop in per-sign
pass rate, spike in `learner_disagreed` flags, latency regression),
rollback is a single config change: flip `is_active` back to the
previous version. The previous artifact remains in R2 indefinitely.

This is a single-command operation; we will not be in a position to
need a multi-hour incident response to a bad model.

---

## 6. Why this document exists

A well-documented EdTech failure pattern: brilliant engineers, slick
demos, no eval discipline. Models drift, accuracy degrades, nobody
notices until a school cancels the contract. This document is the
structural defense against that pattern.

The eval gate is the place where mission alignment meets engineering
discipline. We do not promote a model whose fairness is worse than
the one it replaces. We do not promote a model whose accuracy is
worse on any sign. We do not promote a model whose latency would
frustrate learners on cheap laptops. The criteria are written, the
criteria are public, the criteria are enforced.
