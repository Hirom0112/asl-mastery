# Dataset

> Where the training video comes from, how it gets cleaned, and how
> splits are assigned. This document is part of the no-pretrained-pipeline
> defense (per ADR 0006) and is also where the project's fairness
> commitments live.

---

## 1. Sources

Three sources, layered:

### 1a. Public ASL video datasets (confirmed allowed; see CLAUDE.md §4)

| Dataset | Scale | Notes |
|---|---|---|
| WLASL (Word-Level ASL) | ~2,000 signs, ~21,000 clips | Multiple signers. Research-use license; check terms before commercial use. |
| MS-ASL | ~1,000 signs, ~25,000 clips | Microsoft Research dataset. Similar license. |
| ASL-LEX 2.0 | 2,723 signs, phonological metadata | Smaller per-sign sample but linguistically annotated. Useful for the five-parameter sign metadata in our vocabulary table. |

These provide the bulk of training data per sign. We filter each dataset to our 75–100 vocabulary, take only clips of those signs, and store them in R2 with provenance metadata (source dataset, original clip id, original signer id, license terms).

These are not pretrained models. They are raw video data, which the brief Requirement 6 permits and Requirement 7 does not exclude. Confirmed allowed per CLAUDE.md §4.

### 1b. Canonical reference videos sourced from WLASL / MS-ASL with attribution

For slice 1 (per ADR 0004) we do not contract a Deaf instructor to record canonical references. Canonical reference clips are selected from WLASL and MS-ASL, per-clip, with attribution and license terms recorded. Selection criteria per sign:

- Signer is documented by the source dataset as a fluent ASL user.
- Framing and lighting are clean enough that the clip can be re-cropped into our green-box framing without cutting off the sign.
- Multiple candidate clips per sign are preserved; the highest-quality clip is the learner-facing reference, the rest serve as anchor positives in training.

Slice-2 candidate (production deployment requirement, not pilot work): engage a Deaf ASL instructor to re-record all 75–100 signs with 3–5 takes each in our standardized green-box framing, replacing the WLASL/MS-ASL learner-facing references. Budget estimate when undertaken: $50–75/hour, ~4 hours = $200–300, plus separate compensation for vocabulary and hint review. See `docs/decisions/0004-public-sources-only.md`.

### 1c. Self-recorded learner-condition supplement

Team members, friends, and willing cohort-mates. ~30 minute sessions, distributed across signs.

- Captures variation in laptop webcams, dorm/apartment lighting, casual clothing, varied skin tones, varied signing skill.
- This is the data that brings the training distribution closer to the deployment distribution.

Target (revised under ADR 0006): **50–80 clips per sign minimum** across all three sources combined, down from the ~200 target that served the superseded ADR 0001's pixel-trained architecture. Landmark-based classifiers train comfortably on dramatically less data because their inputs are low-dimensional and MediaPipe absorbs most of the visual variance. Diversity targets across skin tone, lighting, background, and clothing are tracked in a coverage spreadsheet (`docs/dataset_coverage.csv`) and remain non-negotiable.

---

## 2. Recording protocol

Implemented by the admin-gated recording tool (`/admin/record`).

- Resolution: source-camera native, downsampled to 480×480 for storage. The stored resolution is intentionally higher than what any current pipeline step requires, so future reprocessing (e.g. a new MediaPipe version, or slice-2 work that revisits pixels) can be done without re-collecting clips.
- Frame rate: 30 fps.
- Clip length: 2 seconds per take.
- Empty-frame capture: 1 second of background, with the signer stepped out of frame, captured at session start. Used for MOG2 background-subtraction augmentation.
- Green-box framing visible on screen during recording; clips that fall outside the box are still saved (so we can crop differently later) but the visible box trains the contributor to position consistently.
- Metadata captured per clip:
  - `signer_id` (uuid)
  - `sign_id`
  - `take_number`
  - `captured_at`
  - `lighting` (enum: bright / normal / dim, self-reported)
  - `background` (enum: plain / cluttered, self-reported)
  - `handedness` (right / left / ambidextrous)
  - `sleeve_length` (short / long / none — sleeves change visible arm contour)
  - `fitzpatrick` (1–6 integer or null, optional, with consent)
  - `self_rated_correctness` (1–5 integer, signer's own judgment)

Consent form signed before any contributor records.

---

## 3. Cleaning pipeline

Reproducible from a single command: `python -m training.data.clean --version <vN>`.

Stages:

1. **Ingest.** Pull raw clips from `raw-training-data/` R2 bucket along with their metadata rows.
2. **Sign-window trimming.** Trim each clip to the actual sign window. For instructor-recorded clips, trim by manual annotation. For self-recorded clips, the recording tool's countdown means the sign occupies a known 2-second window. For public dataset clips, use whatever window the source dataset annotates.
3. **Framing normalization.** Crop to the green-box region. For public dataset clips, this is a center-square crop scaled to match our box's aspect ratio.
4. **Frame rate normalization.** Resample to 30 fps where source differs.
5. **Length normalization.** Sample 16 frames evenly across the 2-second window (every 3.75 frames at 30 fps).
6. **Resize.** Bilinear resize to 256×256. (Under ADR 0006 the classifier no longer sees pixels, but a clean square crop is preserved for reproducibility and for slice-2 work that may re-process the source video.)
7. **Dedup.** Compute perceptual hash per clip; remove duplicates within a single signer's contributions.
8. **MediaPipe Holistic extraction (per ADR 0006).** Run each cleaned clip through MediaPipe Holistic and extract the keypoint subset specified in `docs/MODEL.md` §1 (21 left-hand + 21 right-hand + upper-body pose subset, each frame). Save the resulting `(T, K)` keypoint tensor per clip alongside the source video. Record the exact MediaPipe version in the manifest. Raw clips are retained in R2 so reprocessing is possible when MediaPipe versions change.
9. **Manifest write.** Output `dataset_v<N>_manifest.json` listing every clip with its source, signer id, sign id, conditions, MediaPipe version, keypoint-tensor storage path, and source-video storage path.

All processed clips and their keypoint tensors are written to a versioned directory in R2: `cleaned-training-data/v<N>/`. The keypoint tensors are the actual training inputs; source videos are retained for reproducibility.

**Note on augmentation surface.** Under ADR 0006 augmentation happens at the keypoint level inside the training pipeline (see `docs/MODEL.md` §3), not at the pixel level in the cleaning pipeline. The cleaning pipeline produces clean, MediaPipe-extracted keypoint tensors; augmentation is layered on top during training only.

---

## 4. Splits

**Signer-disjoint.** This is the single most important property of our validation methodology.

- A given `signer_id` appears in exactly one of train, validation, or test for the lifetime of that dataset version.
- The assignment is committed to the repo as `training/splits/v<N>.json`.
- Target proportions: 70% train, 15% validation, 15% test, measured by signer count and by clip count after assignment.

**Stratification within the split.** All three demographic axes we care about (skin tone, handedness, recording conditions) appear in test, not just train. If a Fitzpatrick bucket has only one signer, that signer goes in test (the validation report needs to be honest about per-demographic performance).

The test set is touched once, for the final validation report per model version. It is never used for hyperparameter tuning or threshold calibration. Validation set is used for tuning and calibration. Train set drives gradient updates.

---

## 5. Fairness commitments

The validation report (per `EVAL_GATE.md`) breaks accuracy out by:

- Skin tone (Fitzpatrick bucket) where consent permits.
- Handedness (right / left).
- Lighting condition.
- Background condition (plain / cluttered).
- Sleeve length (short / long).

If the gap between best and worst Fitzpatrick bucket exceeds 10 percentage points on overall accuracy, the model fails the eval gate. The response is data collection for the under-represented buckets, not threshold gymnastics.

Per-user monitoring (`PRIVACY.md`-permitting) tracks pass-rate distribution; users whose pass rate stays below 30% over 50+ attempts are surfaced for review. The most likely cause is demographic mismatch in training data, and the response is targeted data collection.

**Honest disclosure about landmark-based fairness (ADR 0006).** The landmark-based architecture removes one major axis of fairness risk: the classifier itself sees keypoint coordinates, not pixels, so it cannot learn skin tone as a spurious feature. However, **MediaPipe's own landmark-detection accuracy can vary across demographics**, and that residual risk does not disappear when we delegate landmark extraction to a third-party model. The validation report (`docs/EVAL_GATE.md`) therefore reports MediaPipe's per-demographic detection-success rate alongside per-demographic classifier accuracy. If MediaPipe is failing more often on some demographic, we surface that, not paper over it.

---

## 6. Provenance and licensing

Every clip in the final dataset has:

- Source attribution (which of 1a/1b/1c, and within 1a, which dataset and original clip id).
- License terms recorded.
- Signer consent record (for 1b and 1c).

Public datasets — if we use them — are credited in `README.md` per their license terms. ASL instructor and self-recorded contributors are credited in `README.md` Acknowledgments unless they request otherwise.

---

## 7. Vocabulary metadata

A companion table records, per sign:

- `gloss` (display form, e.g. "THANK-YOU").
- `parameters`: handshape, location, palm orientation, movement, non-manual markers. Pulled from ASL-LEX 2.0 (per ADR 0004); instructor validation deferred to slice 2.
- `flippable` (bool): whether horizontal flip is a safe augmentation.
- `static_or_movement`: whether the sign is single-frame-identifiable or requires temporal context. Drives slice-1 prioritization toward more static signs initially.
- `pre_attempt_hint` (text): authored against ASL-LEX 2.0 parameter codes and Lifeprint per-sign instructional notes. Fluent-signer validation deferred to slice 2 (ADR 0004).
- `generic_failure_hint` (text): same authoring and slice-2 validation path as `pre_attempt_hint`.
- `reference_video_url`: pointer to the WLASL/MS-ASL clip selected for that sign, mirrored to R2 with attribution metadata; replaced by instructor-recorded reference in slice 2.

The vocabulary metadata file is committed to the repo. It is the single source of truth for what the system can recognize and how it teaches each item.
