# ADR 0015: External labeled CV datasets — provenance and license constraints

**Status:** Proposed (awaiting user approval as part of the Phase 1 data-plan review)
**Date:** 2026-05-21
**Relates to:** [ADR 0011](./0011-landmarks-and-templates-pivot.md) (recognition architecture), [ADR 0012](./0012-strict-from-scratch-cv-constraint.md) (strict from-scratch CV constraint), [ADR 0004](./0004-public-sources-only.md) (public sources only for the pilot)

---

## Context

ADR 0011 commits the project to a four-stage from-scratch landmark pipeline:
hand detector → hand landmark regressor → pose detector → face detector. Each
stage needs labeled training data. The roadmap's original estimate
(`docs/VOCABULARY_TRAINER_ROADMAP.md` Phase 2 Slice 2.1) budgets ~120 hours of
hand-keypoint labeling on the assumption that the project labels everything
from scratch.

Recent investigation (`docs/data/external_datasets_audit.md`) confirms that
several **public hand / pose / face keypoint datasets** exist where the labels
were produced **by humans, by physical sensors, or by the dataset authors' own
multi-view fitting pipelines** — i.e., not by a third-party pretrained CV
model. These are the same kind of provenance the project already accepts for
the raw video corpus (WLASL, Sem-Lex, ASL Citizen per ADR 0009): public,
human-or-physical-sensor-derived, license-permitted for research.

Using them would dramatically reduce the labeling cliff and would not
contradict ADR 0012's strict from-scratch CV rule, provided the labels were
not produced by a pretrained CV model. **The question ADR 0015 answers is:
what acceptance criteria govern external labeled CV datasets in this
project?**

---

## Decision

External labeled CV datasets are **permitted as training inputs** for any
detector in the four-stage landmark pipeline, subject to all six of the
following criteria. A dataset failing any criterion is rejected.

### Acceptance criteria

1. **Verifiable provenance from a primary source.** The dataset's labeling
   methodology must be documented in the originating paper, the official
   dataset README, or the authoritative project page. Third-party blog posts
   and HuggingFace card excerpts do not satisfy this criterion on their own.
2. **Permitted label-production mechanism.** Labels must be produced by **one
   or more of** the following:
   - **Human annotation.** Manual labeling, including via crowdsourcing
     platforms (Mechanical Turk, Scale AI). Documented quality-control steps
     strengthen but are not required for acceptance.
   - **Physical sensor measurement.** Magnetic motion-capture sensors
     attached to the body, Leap Motion or similar depth sensors,
     optical-marker mocap.
   - **Multi-view geometric fitting.** Triangulation across calibrated
     cameras with no learned model in the loop, OR with a learned model
     trained by the dataset authors on their own human-annotated seed data
     (the Simon et al. / FreiHAND pattern).
   - **Synthetic rendering with synthetic ground truth.** Annotations
     derived geometrically from the rendered scene, not predicted by a
     learned model.
3. **Disqualifying label-production mechanisms.** A dataset is rejected if
   its labels were produced in whole or in part by **any of** the following:
   - **Pretrained landmark detectors** (hand / pose / holistic trackers), in any version, with any
     post-processing.
   - **OpenPose** or any other pretrained pose / keypoint detector from
     CMU, Facebook AI, Google Research, Microsoft Research, or any third
     party.
   - **Any unspecified "AI tool" or "ML model"** without traceable provenance
     to one of the four permitted mechanisms above.
   - **Any pretrained backbone** repurposed as a feature extractor whose
     outputs become labels (e.g., ImageNet ResNet features used to seed
     pseudo-labels).
4. **License compatibility.** The dataset license must permit research use
   in this pilot. Restrictions in order of acceptability:
   - Permissive (BSD, MIT, CC BY, Apache 2.0) — accept unconditionally.
   - Research-only / non-commercial (CC BY-NC, CC BY-NC-ND, MSR-LA-style
     research licenses, custom "research purposes only" terms) — accept for
     the pilot; flag in the dataset's `PROVENANCE.md` so the slice-2
     commercial-cliff review catches the restriction.
   - More restrictive than CC BY-NC-ND or requiring per-use approval — flag
     for explicit user approval before download.
5. **Modality fit.** Datasets must be **RGB single-frame or RGB video** to
   match the browser inference path. Depth-only or thermal-only datasets are
   out of scope for the landmark pipeline (they may be in scope for future
   work and that's a separate ADR).
6. **Default-conservative provenance ruling.** When a primary source does
   not clearly state which of the permitted mechanisms produced the labels,
   **the dataset is rejected with reason "unverified provenance."** Better
   to leave out a usable dataset than to silently include a contaminated one.
   A subsequent ADR amendment can promote a rejected-unverified dataset to
   accepted once the provenance is independently confirmed.

### Why pretrained-model-labeled (and analogously labeled) datasets are excluded

The brief's Requirement 7 forbids pretrained CV models. ADR 0012 codifies the
CV perimeter and the anti-drift rules implementing that prohibition. A naive
reading of ADR 0012 might permit training on pretrained-model-labeled data —
the pretrained model itself is not in our pipeline, only its predictions are. The strict
reading rejects this for two reasons:

1. **Substantive equivalence.** Training a network on a pretrained model's predictions
   minimizes a loss against those predictions. The network learns to imitate
   that model within the noise floor of the distillation loss. Our weights
   become a compressed approximation of its weights. We have not
   built a from-scratch detector; we have built a from-scratch wrapper
   around someone else's model. The brief's intent — that every learned component be
   the project's own work — is violated even if the literal text is
   satisfied.
2. **Audit credibility.** When a reviewer reads ADR 0012
   and asks "what is from-scratch about this?" the honest answer must be
   "the architecture and the weights." If the labels came from a pretrained model,
   the honest answer is "the architecture, but the weights are a
   distillation of that model's." That is a different project than the one
   ADR 0012 declares.

By extension, any dataset whose labels were generated by any pretrained CV
model is disqualified for the same reasons.

### Acceptance procedure

For each candidate external dataset, the process is:

1. Add an entry to `docs/data/external_datasets_audit.md` under "Per-dataset
   findings" filling in: image count, subject count, annotation methodology
   (with primary-source citation), originating paper, license, signing
   specificity, download URL, approximate size, format.
2. Apply the six criteria above. Declare PASS or REJECT with reason.
3. If PASS, add a download script under `scripts/datasets/` that downloads
   to `data/external/<dataset_name>/`, verifies checksums when available,
   and writes `data/external/<dataset_name>/PROVENANCE.md` recording: source
   URL, download date, license, SHA256 of each downloaded archive, and the
   audit-memo entry by name.
4. Do not execute the download until the user has read the audit memo and
   approved the dataset.

### Self-recorded supplementary data

External datasets are *additive* to, not a *replacement* for, self-recorded
ASL-specific frames. The roadmap's coverage gaps — signing context,
body-relative locations, the user's webcam setup, per-Fitzpatrick balance —
are not closed by any of the candidate public datasets. The supplemental
labeling estimate is documented in
`docs/data/external_datasets_audit.md` § "Supplemental ASL-specific
labeling estimate."

---

## Rejected alternatives

- **Label everything from scratch ourselves.** Rejected. The original
  ~120-hour estimate for hand keypoints alone in
  `docs/VOCABULARY_TRAINER_ROADMAP.md` Phase 2 Slice 2.1 was a self-imposed
  cliff. Public human-labeled hand-keypoint corpora exist and were never
  excluded by ADR 0012's text. Using them recovers months of labeling
  time without breaking the from-scratch constraint.
- **Permit pretrained-model-labeled datasets as long as we don't import the pretrained model.**
  Rejected. Argued above — substantive equivalence + audit credibility.
- **Permit datasets with unverified provenance pending a "good faith"
  judgement call.** Rejected. The audit's default-conservative rule
  eliminates a class of subtle contamination that no later test can detect.
  The cost of erroneously rejecting a clean dataset is small (we still
  have many recommended sources); the cost of erroneously accepting a
  contaminated one is project-credibility-destroying.
- **Promote a single dataset (e.g., FreiHAND) to canonical and skip the
  others.** Rejected. Single-source training under-represents the variance
  in real-world hand appearance and is documented to under-perform on
  diverse subjects (per published fairness literature). The intent of using
  multiple recommended sources is exactly the diversity ADR 0011's
  fairness commitments require.

---

## Consequences

### Roadmap

- **Phase 2 Slice 2.1** (hand-keypoint labeling) collapses from
  ~120 hours of from-scratch human labeling to ~5–15 hours of
  *supplemental* self-recording focused on coverage gaps. The labeling
  cliff named in the roadmap is no longer the dominant Phase 2 cost.
- **Phase 1 Slice 1.1** (hand-bbox labeling) is also reduced: most of the
  RECOMMENDED hand keypoint datasets ship usable bounding boxes derived
  from their keypoint annotations.
- **Phase 3 Slices 3.1 and 3.2** (pose and face) similarly collapse to
  small supplemental labeling efforts.

### Disk and bandwidth

The combined RECOMMENDED corpus is in the hundreds of GB if InterHand2.6M's
30fps split is included, ~80–120 GB if only the 5fps split is taken. The
user's available disk and download bandwidth should be confirmed before
the larger downloads execute.

### Model cards

Every model card under `docs/model_cards/` for a detector trained on
external data must declare under "Training data":
- The full list of external datasets used, each with a link to its
  `data/external/<dataset_name>/PROVENANCE.md`.
- The supplemental self-recorded set, with the same provenance record.
- The "Pretrained components: none" line remains as before, plus a new
  line: "External labeled datasets used: yes — see PROVENANCE.md
  references; all sources audited under ADR 0015."

### Fairness reporting

The per-Fitzpatrick eval-gate criterion 3 (`docs/EVAL_GATE.md`) becomes
more important under this ADR, not less. Public datasets carry documented
demographic imbalances; we inherit them. The held-out self-recorded set
must be deliberately Fitzpatrick-diverse to act as a fair eval surface.

### Slice-2 commercial cliff (per ADR 0009 precedent)

Datasets accepted under research-only or CC BY-NC licenses (FreiHAND, CMU
HandDB, InterHand2.6M, MPII, COCO via its image rights, WIDER FACE) make
the eventual commercial deployment of the trained weights
license-incompatible without retraining. The slice-2 commercial-cliff
risk is identical in structure to the one ADR 0009 already names for ASL
Citizen, and is mitigated the same way: ship the pilot, scope commercial
release to a future retraining on license-clean substitute data.

---

## How this is verified

- `docs/data/external_datasets_audit.md` exists and is up to date.
- Every `data/external/<dataset_name>/` directory contains a
  `PROVENANCE.md` whose content matches a corresponding RECOMMENDED entry
  in the audit memo.
- Each `data/external/` annotation and provenance record is checked to
  confirm no labels were produced by a pretrained model — zero
  pretrained-model-labeled sources (a hit would indicate either a wrongly
  accepted dataset or a contaminated annotation pipeline).
- Every model card under `docs/model_cards/` for a from-scratch detector
  references its training data's PROVENANCE entries by path.
- Every download script under `scripts/datasets/` writes `PROVENANCE.md`
  with SHA256 of the downloaded archives. A reviewer can re-checksum the
  on-disk archives and confirm they match what the script declared.
