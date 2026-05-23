# RELEARN.md — get fluent again on the current training architecture

> Read this *after* `STATUS.md`. Then come back here.
> Today's truth: **four from-scratch CV detectors → distribution-of-templates
> sign matcher**, governed by ADRs 0011 + 0012 + 0015. Everything else is
> superseded — see STATUS.md:56–69 for the supersession table.

---

## ⚠️ CONSTRAINT RISKS (read first)

I audited the ADR-0012 compliance greps. **No live violations of the
No-Pretrained-Models rule.** Three `mediapipe` hits exist but each is benign:

1. `ml/archive/v3.0/code_v1_bilstm/clean.py` — archived pre-pivot pipeline
   under `ml/archive/`. Not on any code path. Kept as historical record per
   ADR 0010 "no silent revisions."
2. `training/detectors/external_loaders/hagrid.py:_check_no_landmark_drift()` —
   a **tripwire** that scans HaGRID schemas for the literal string
   `"mediapipe"` so a future source switch can't smuggle MediaPipe-derived
   landmarks back in. The mention is defensive, not consumptive.
3. `components/practice/runner.tsx:116` — writes `mediapipeDetectionFailed:
   false` to the attempts table. This is a DB column kept for the v1/v2/v2.1
   historical schema (comment at lines 113–115). New attempts always write
   `false`; nothing reads MediaPipe.

`torchvision.models` grep: zero real imports. All hits are documentation
strings in `training/detectors/*.py` declaring "No torchvision.models."

`load_state_dict`: only project-internal checkpoint loads
(`training/classifier/validate.py:126`, `training/classifier/export.py:62`,
both inside the archived classifier path). The new detector training loops
use their own resume logic; verify before reactivating the
`training/classifier/` directory for anything new.

**One thing to verify before the interview:** the `training/classifier/`
tree (cnn.py, validate.py, export.py, dataset_video.py) is from the ADR-0010
end-to-end CNN era. ADR 0011 says it is "not built" (line 232). It still
exists on disk and pollutes the compliance grep. If asked, the honest answer
is: "Pre-pivot scaffolding, deactivated, not loaded by anything in the new
pipeline. Leaving it under `training/classifier/` was an oversight — should
move under `ml/archive/v3.0/` alongside `code_v1_bilstm`."

No other smells found. The four detector files (`hand_detector.py`,
`hand_landmarks.py`, `pose_detector.py`, `face_detector.py`) each begin with
explicit "No pretrained init. No torchvision.models." docstrings and use
Kaiming-normal initialization only.

---

## 30-minute fastest path

Read these five files in this order. They are arranged so each one sets up
the question the next one answers.

| # | File | What to look for |
|---|---|---|
| 1 | `STATUS.md` | The diagram at lines 14–31 — this *is* the architecture. Note "Session 15" and the pipeline-state table (lines 84–109) for what's trained vs. pending. |
| 2 | `docs/decisions/0011-landmarks-and-templates-pivot.md` | §"Decision" (lines 49–91) is the contract. §"Why this pivot now" (94–127) is the data-ceiling argument you'll repeat in the interview. |
| 3 | `docs/decisions/0012-strict-from-scratch-cv-constraint.md` | §"The CV perimeter" (38–55) and §"Anti-drift rules" (96–131). This is the rulebook you defend the architecture *with*, not against. |
| 4 | `docs/decisions/0015-external-cv-datasets-provenance.md` lines 1–80 | Why we can use FreiHAND / CMU HandDB / MPII / COCO / WIDER without breaking ADR 0012: those datasets are **human / sensor / multi-view labeled**, not pretrained-model labeled. The six acceptance criteria are the audit surface. |
| 5 | `training/detectors/hand_detector.py` (skim, ~150 lines) | A concrete instance of "from-scratch": Kaiming init, no torchvision.models, CenterNet head built by hand. Mentally substitute for the other three detectors — they follow the same shape. |

After this you can answer: *what runs, what trains it, what proves it's from-scratch.*

---

## 2-hour deep dive

Add these, in order. Stop when you understand each one — don't speed-read.

### A. Training loop walkthrough (~25 min)
1. `training/detectors/train.py` — the hand-detector training loop. Pay
   attention to the dual-collate path (`_collate` vs `_collate_gpu_aug`) at
   lines 31–46 — GPU augmentation was added because CPU aug was the
   bottleneck on Modal.
2. `training/detectors/losses.py` — `HandDetectorLoss`: CenterNet focal
   heatmap loss + L1 size regression. From-scratch implementation, no
   external loss libraries.
3. `training/detectors/augment.py` and `gpu_augment.py` — see why both
   exist (CPU path = readable, GPU path = throughput). Same augmentations,
   two implementations.
4. `training/detectors/train_landmarks.py` — the keypoint-agnostic refactor
   (Session 15, see STATUS.md:94). Used to train hand_landmarks and pose
   from the same loop with different keypoint counts.
5. `training/modal_app.py` lines 379–880 — the Modal entrypoints:
   `train_hand_detector`, `train_hand_landmarks`, `train_pose`, `train_face`,
   `extract_trajectories_v2`, `fit_templates`. Each `@app.function` is one
   GPU job. The "from-scratch budget" is the wall-clock of these jobs.

### B. Dataset + ingestion (~25 min)
1. `training/detectors/dataset.py` — `HandBboxDataset`, `load_manifest()`.
   Manifests are JSON lists pointing at frames + label dicts. **The
   manifest format is the contract** between ingestion and training.
2. `training/detectors/external_loaders/{freihand,cmu_handdb,mpii_pose,wider_face,hagrid}.py` —
   each loader normalizes one external dataset into the manifest format.
   Open `hagrid.py` and find `_check_no_landmark_drift()` — that's the
   ADR-0015 tripwire mentioned in the constraint-risks section.
3. `docs/data/external_datasets_audit.md` — the provenance audit. Skim §
   "Combined-corpus summary" for total labeled-instance counts per detector
   (~150K–600K hand instances, ~290K person instances, ~393K face boxes).
4. `data/templates_v3/_thresholds.json` — actual artifact for ADR 0011
   Requirement 9. Per-sign pass thresholds calibrated against
   positive/negative score distributions. STATUS.md:102 notes accuracy is
   currently poor (~24% prec / ~26% rec) — pipeline OK, models
   undertrained. That's the next iteration.

### C. Eval gate logic + matcher (~30 min)
1. `training/detectors/fit_templates.py` lines 1–60 — template structure:
   per-sign (T, F) mean + diagonal variance, plus raw clips for nearest-
   clip fallback, plus `dominant_hand_only` flag for one-handed signs,
   plus mirror-aware hand slot assignment.
2. `training/detectors/sign_matcher.py` lines 1–60 — `mahalanobis_score`,
   `predict`, `predict_in_slice`, `calibrate_thresholds`. Note the DTW
   alignment (Sakoe-Chiba radius 8) at lines 38–42 — handles users signing
   faster/slower than templates. Note `NEAREST_CLIP_FALLBACK_THRESHOLD = 8`
   — signs with <8 clips can't fit a reliable Gaussian, so fall back to
   1-NN.
3. `lib/inference/sign_matcher.ts` + `sign_matcher.parity.test.ts` — TS
   port of the matcher for browser inference. The parity test is the
   canary: if py and TS diverge by more than ε, deploy is blocked.
4. `scripts/check_eval_gate.py` — the criterion checker. The eval-gate
   doc itself (`docs/EVAL_GATE.md`) is **stale** per STATUS.md:67; an
   eval-gate v2 ADR is pending. Read the script, not the doc, to know what
   currently gates.

### D. Dataset versioning (~10 min)
- Templates live under `data/templates_v3/` (v3 = post-ADR-0011). The `v3`
  suffix on `extract_trajectories_v2.py` and `templates_v3/` is your version
  marker. There is no semver on detector weights yet — each Modal run
  produces a `runs/<run_id>/best.pt`. Run IDs encode date + hardware (e.g.
  `pose_v0_h100_20260521_164906Z`).
- Model cards under `docs/model_cards/` are the per-model spec:
  `hand_detector_v0.md`, `hand_landmarks_v0.md`, `face_detector_v0.md`
  exist; pose card is pending per STATUS.md:105.

### E. Final compliance sweep (~10 min)
Run the three ADR-0012 audit greps yourself so the muscle memory is fresh:
```bash
grep -r 'mediapipe' --include='*.ts' --include='*.tsx' --include='*.py' . | grep -v node_modules | grep -v ml/archive
grep -r 'torchvision.models' --include='*.py' training/
grep -rn 'load_state_dict' --include='*.py' training/detectors/
```
You should see the three benign hits I cataloged above and **nothing in
`training/detectors/`** for the third grep.

---

## Be ready to defend this in the Superbuilders interview

Three likely questions. One paragraph each. All defenses hold under ADR 0012.

### Q1 — "You switched from end-to-end RGB to a four-stage landmark pipeline. Why is that not just MediaPipe in disguise?"

The architecture *shape* echoes MediaPipe's Hands+Pose+Face decomposition,
but every weight on every stage is trained by us on documented data — see
`docs/model_cards/{hand_detector_v0,hand_landmarks_v0,face_detector_v0}.md`,
each declaring "Pretrained components: none" per ADR 0012 §"Anti-drift
rule 5." The data motivating the pivot is the 13–20 clips/sign ceiling
(ADR 0011 §Context): an end-to-end small 3D CNN on that data ceilings at
30–50% top-1 by our own ADR 0010 estimate, which is below the 85% eval-gate
floor. Landmarks are vastly more sample-efficient because each detector
trains on *frames* (~150K labeled hand instances, 290K person instances,
393K face boxes — `docs/data/external_datasets_audit.md`), not on per-sign
clips, and the matcher consumes a representation that has already absorbed
skin-tone / lighting / background variance. The from-scratch story is
intact: four small models we trained, plus a statistical template matcher.

### Q2 — "You're training on FreiHAND, CMU HandDB, MPII, COCO, WIDER. Aren't those just somebody else's labels you're inheriting?"

Inheriting *labels* is not the same as inheriting *weights*. ADR 0012's
constraint is "no pretrained CV weights in the recognition pipeline." Our
loss is computed against external labels, not against external model
outputs. ADR 0015's acceptance criteria explicitly disqualify any dataset
whose labels were produced by MediaPipe, OpenPose, or any pretrained
keypoint model — so labels of the form "we ran MediaPipe and saved its
output" are categorically rejected. The accepted datasets use human
annotation (WIDER FACE crowdworkers, MPII, COCO keypoints, HaGRID via
Yandex.Toloka), physical sensors (magnetic mocap), or multi-view geometric
fitting on author-collected human seeds (FreiHAND, CMU HandDB Panoptic).
That is the same provenance class the project already accepts for the
12K-clip ASL video corpus under ADR 0008/0009. Every dataset has a
`PROVENANCE.md` and an entry in `docs/data/external_datasets_audit.md`.

### Q3 — "The current pipeline accuracy is ~24% precision / ~26% recall (per STATUS.md). Isn't that worse than Path B's projected 30–50%?"

Two things to separate. First, what's measured: ADR 0010's 30–50% was an
*estimate* for a model we never trained; the current 24/26 is a real
end-to-end pipeline output. Second, the cause: STATUS.md:102 records
"pipeline OK, models undertrained" — the hand detector trained for 30
epochs on a single L4/A100, the pose model is still training as of Session
15, and threshold calibration ran on those undertrained outputs. The pivot
buys us a path to improvement that didn't exist for Path B: each of the
four detectors can be iterated independently against per-component
validation metrics (val=1.74 on hand_det, 0.036 on hand_landmarks, 1.16 on
face_det per STATUS.md:90–92), without retraining a 60M-parameter video
model, and the template matcher decomposes failure along the four Stokoe
parameters (handshape / location / movement / palm orientation) which is
both pedagogically useful (Phase 5 hint targeting) and diagnostically
useful for where to spend the next training cycle. The accuracy will move;
the architecture choice is what made the moves cheap.

---

## Smoke test — confirm the pipeline is healthy

One command, runs locally on CPU in <60s, catches the failure modes from
session 16:

```bash
python -m scripts.preflight
```

This validates: manifest schema integrity, no out-of-bounds bboxes, loss
numerics finite on a synthetic batch, and that every Modal entrypoint
referenced in `training/modal_app.py` resolves to an importable function.
If it passes, the training pipeline is healthy enough to launch a Modal
job. If it fails, read the failure message — preflight reports the exact
manifest path or symbol that broke.

For an end-to-end check that exercises the full code path (not just
imports), use the smoke-manifest builder:

```bash
python scripts/make_smoke_manifest.py \
    --input  dataset/clean/v3/dataset_v3_manifest.json \
    --output dataset/clean/v3/dataset_v3_smoke_manifest.json \
    --signs  2
```

Then the smoke train invocation documented at the top of
`training/detectors/train.py` (2 epochs, batch 8, on the 2-sign subset).
That one needs a GPU or patience — preflight is the daily health check.
