# Dataset

> ⚠️ **PARTIALLY SUPERSEDED — read [`STATUS.md`](../STATUS.md) first.**
> This document covers the ASL clip corpus only (WLASL + ASL Citizen +
> Sem-Lex + ingestion + cleaning + splits). That content is still
> current as the substrate for **trajectory extraction and template
> fitting** under ADR 0011 Phase 4. **What's new and not covered here:**
> the four from-scratch CV detectors (hand bbox, hand keypoints, pose,
> face) also need labeled training data, which now comes from public
> human-labeled CV datasets per
> [ADR 0015](decisions/0015-external-cv-datasets-provenance.md). The
> external-dataset corpus is audited in
> [`docs/data/external_datasets_audit.md`](data/external_datasets_audit.md);
> live PROVENANCE.md records per dataset live under
> `data/external/<dataset>/`. Self-training pseudo-labels on the ASL
> corpus close the signing-context coverage gap (see
> `training/detectors/self_train.py`).

> Where the training video comes from, how it gets cleaned, and how
> splits are assigned. This document is part of the no-pretrained-
> pipeline defense (per [ADR 0010](./decisions/0010-reversal-of-adr-0006.md),
> which reinstated [ADR 0001](./decisions/0001-recognition-architecture.md)
> Path B on 2026-05-20) and is also where the project's fairness
> commitments live.

---

## 1. Sources

Three sources, layered:

### 1a. Public ASL video datasets (confirmed allowed; see CLAUDE.md §4)

| Dataset | Scale | License | Notes |
|---|---|---|---|
| WLASL (Li et al. 2020) | ~2,000 signs, ~21,000 clips | Research-use (mixed YouTube licenses on individual clips) | Multiple signers. Used in v1.0.1 and v2.x. |
| MS-ASL | ~1,000 signs, ~25,000 clips | MSR-LA (non-commercial research) | Skipped for slice 1 per [ADR 0008](./decisions/0008-public-data-only-training.md) sufficiency check. |
| **ASL Citizen** (Desai et al. 2023, `arXiv:2304.05934`) | **83,912 clips, 2,731 signs, 52 signers** | **MSR-LA (non-commercial research; no redistribution)** | **Added at v2.x per [ADR 0009](./decisions/0009-asl-citizen-v2.md). 100% vocabulary overlap with our 80 signs (verified Phase 9b.2 against ASL-LEX 2.0). Slice-2 commercial deployment requires a separate license.** |
| ASL-LEX 2.0 (Sehyr et al. 2021) | 2,723 signs, phonological metadata | CC-BY-NC 4.0 | Source of canonical glosses and five-parameter metadata. Used for ASL Citizen overlap verification. |

These provide the bulk of training data per sign. We filter each dataset to our 75–100 vocabulary, take only clips of those signs, and store them in R2 with provenance metadata (source dataset, original clip id, original signer id, license terms).

These are not pretrained models. They are raw video data, which the brief Requirement 6 permits and Requirement 7 does not exclude. Confirmed allowed per CLAUDE.md §4.

### 1b. Canonical reference videos sourced from WLASL / MS-ASL with attribution

For slice 1 (per ADR 0004) we do not contract a Deaf instructor to record canonical references. Canonical reference clips are selected from WLASL and MS-ASL, per-clip, with attribution and license terms recorded. Selection criteria per sign:

- Signer is documented by the source dataset as a fluent ASL user.
- Framing and lighting are clean enough that the clip can be re-cropped into our green-box framing without cutting off the sign.
- Multiple candidate clips per sign are preserved; the highest-quality clip is the learner-facing reference, the rest serve as anchor positives in training.

Slice-2 candidate (production deployment requirement, not pilot work): engage a Deaf ASL instructor to re-record all 75–100 signs with 3–5 takes each in our standardized green-box framing, replacing the WLASL/MS-ASL learner-facing references. Budget estimate when undertaken: $50–75/hour, ~4 hours = $200–300, plus separate compensation for vocabulary and hint review. See `docs/decisions/0004-public-sources-only.md`.

### 1c. Self-recorded learner-condition supplement — DEFERRED TO SLICE 2 per ADR 0008

Originally planned: team members, friends, and willing cohort-mates recording in 30-minute sessions distributed across signs. Removed from slice-1 scope on 2026-05-19 because no one on the project team is a fluent ASL signer; training on non-signer-authored clips would teach the model wrong handshape / location / movement — worse than less data, it would be misleading data. The recording tool's specification in `docs/ARCHITECTURE.md` §2.2 is preserved as the slice-2 framework for the ADR-0004 instructor engagement. See `docs/decisions/0008-public-data-only-training.md`.

**Slice-1 training-data target (revised under ADR 0010, reverted from the ADR 0006 landmark sizing):** ~200 video clips per sign per [ADR 0001](./decisions/0001-recognition-architecture.md) Path B sizing, drawn from public sources alone (1a) where coverage permits. The actual public-source corpus available (WLASL + ASL Citizen + Sem-Lex) provides ~30–90 clips/sign for our 75-sign vocabulary — materially under the Path B target and the dominant accuracy ceiling for v3.0 (honest expected outcome: 30–50% top-1, named in [`docs/validation/v3.md`](./validation/v3.md) when v3.0 ships). Per-sign downloadable-clip count is measured in Phase 3d; signs falling below a per-sign floor (default: 15 downloadable clips, revisable during Phase 4) are dropped from the slice-1 vocabulary and annotated in `docs/VOCABULARY.md`. The final slice-1 vocabulary count must remain ≥ 75 (Brief Requirement 2); if applying the floor would take it under 75, the floor is reduced rather than the count violated, with the tradeoff disclosed in the validation report.

Diversity over skin tone, lighting, background, and clothing is inherited from WLASL's and MS-ASL's source-signer distributions rather than engineered by us; per-source-signer demographics are recorded in the manifest where the public datasets supply them.

---

## 2. Recording protocol — SLICE-2 FRAMEWORK (not built in pilot per ADR 0008)

The specification below describes the recording tool that the slice-2 Deaf-instructor engagement will use to produce canonical references and supplementary training clips. The tool is not implemented in slice 1.

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

Under [ADR 0010](./decisions/0010-reversal-of-adr-0006.md), the
cleaning pipeline drops the MediaPipe Holistic keypoint-extraction
stage that ran under the now-superseded ADR 0006 landmark
architecture. The pipeline outputs cleaned per-clip MP4 + per-clip
metadata. The classifier consumes raw video tensors directly per
[`docs/MODEL.md`](./MODEL.md) §1; there is no intermediate keypoint
tensor file.

Stages:

1. **Ingest.** Pull clips from `raw-training-data/` R2 bucket along with their metadata rows. For slice 1 under ADR 0008, all rows are sourced from public-dataset ingestion (§1a — WLASL, ASL Citizen per ADR 0009, Sem-Lex); self-recorded clips do not appear in slice-1 manifests.
2. **Per-sign clip-count filter (under ADR 0008).** Count downloadable clips per sign after license / availability filtering. Drop signs below the per-sign floor (default 15) from the slice-1 vocabulary; record the dropped signs and counts in the manifest. The remaining count must be ≥ 75; if not, reduce the floor (with disclosure in the validation report) rather than violate the count.
3. **Sign-window trimming.** Trim each clip to the actual sign window. For public-dataset clips, use whatever window the source dataset annotates (WLASL `frame_start`/`frame_end`, ASL Citizen split CSVs, Sem-Lex metadata). Slice-2 instructor-recorded clips will use the recording tool's countdown for a known 2-second window.
4. **Framing normalization.** Crop to the green-box region. For public-dataset clips, this is a center-square crop scaled to match our box's aspect ratio; motion-weighted sampling is applied within the window to bias toward frames where the signer is mid-sign rather than at rest.
5. **Frame rate normalization.** Resample to 30 fps where source differs.
6. **Length normalization.** Sample 16 frames evenly across the 2-second window (every 3.75 frames at 30 fps).
7. **Resize.** Bilinear resize to 256 × 256. The training-time augmentation stack then crops down to the model's `H × W` (initial target 96 × 96, decided in T4 of the ADR 0010 triage) with offset jitter.
8. **Dedup.** Compute perceptual hash per clip; remove duplicates within a single signer's contributions.
9. **Manifest write.** Output `dataset_v<N>_manifest.json` listing every clip with its source, signer id, sign id, conditions, normalized-MP4 storage path, plus the slice-1 dropped-sign list and the per-sign clip-count floor used. The training loop loads MP4s directly via `torchvision.io` / decord; no intermediate tensor files.

All cleaned MP4 clips are written to a versioned directory:
`dataset/clean/v<N>/normalized_videos/`. The MP4s are the actual
training inputs.

**Note on augmentation surface.** Under [ADR 0005](./decisions/0005-classical-cv-allowed.md)
+ [ADR 0010](./decisions/0010-reversal-of-adr-0006.md), augmentation
happens at the **pixel level** inside the training pipeline (see
`docs/MODEL.md` §3): random spatial crop, color jitter,
brightness/contrast, MOG2 background swap (classical CV under
ADR 0005), small affine, conditional horizontal flip. The
cleaning pipeline produces clean, framing-normalized MP4 clips;
pixel-level augmentation is layered on top during training only.

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

**Honest disclosure about raw-RGB fairness ([ADR 0010](./decisions/0010-reversal-of-adr-0006.md)).** Under the reverted ADR 0001 Path B architecture the classifier *does* see pixels, and it *can* in principle learn skin tone, background, lighting, and clothing as spurious features. The interim downgrade of these axes under ADR 0006 — "the classifier cannot learn skin tone because skin tone is not in its input" — is rescinded along with that ADR. The defenses under Path B are architectural rather than absent:

- The per-Fitzpatrick ≤ 10 pp gap criterion (eval-gate hard criterion 3) becomes *more* important, not less, and is the structural blocker for skin-tone overfitting.
- MOG2 background swap (classical CV per [ADR 0005](./decisions/0005-classical-cv-allowed.md)) randomizes background per training example, which is the canonical pixel-level defense against the dorm-room overfitting axis the superseded ADR 0001 was worried about.
- Color jitter and brightness augmentation cover the lighting axis.

These defenses are not as crisp as "the classifier literally cannot see this feature." The validation report names this honestly.

**Honest disclosure about training-data authorship (ADR 0008).** The slice-1 training set is drawn from WLASL and MS-ASL only. No clips were recorded by the project team. Per-demographic accuracy reporting depends on whatever signer demographics those datasets publish; where they are silent, the validation report says so rather than imputing values. The slice-2 ADR-0004 instructor engagement is the path under which our own signer-disjoint, demographic-tagged, green-box-framed training supplement enters the dataset; that path is named explicitly in the README and is not papered over as if slice 1 produced it.

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
