# ADR 0009: Include ASL Citizen as a slice-1 v2.x training source (MSR-LA, non-commercial)

**Status:** Accepted. **Architectural assumption updated by [ADR 0010](./0010-reversal-of-adr-0006.md) on 2026-05-20:** the +25–30 pp accuracy projection in this ADR was computed under the landmark architecture (ADR 0006), where ASL Citizen was projected to lift v2.x from 17.86% to ~50–60% top-1. Under the reverted raw-RGB Path B architecture (ADR 0010) that projection no longer applies — v3.0's expected outcome is 30–50% top-1 across the slice-1 vocabulary, governed by data-quality limits unrelated to which dataset supplied the raw video. The *inclusion decision* (ASL Citizen raw video is license-permitted for slice-1 research use under MSR-LA, and dropping it would worsen v3.0's data ceiling further) stands. The slice-2 commercial-cliff (cannot ship the trained weights commercially without retraining on license-clean data) is unchanged.
**Date:** 2026-05-20
**Supersedes:** Not a supersession of [ADR 0008](./0008-public-data-only-training.md) — extends its named source list.

---

## Context

The v1.0.1 classifier (`docs/validation/v1.md`) hit **17.86% top-1**
on 80 signs against the eval-gate's 85% floor. Root-cause analysis
in that report named two dominant failure modes: ~11 clips/sign of
training data (1/5 of ADR 0006's 50–80 target) and a 40% MediaPipe
per-frame miss rate. Phase 9 of `docs/ROADMAP.md` and the working
`TODO.md` lay out a four-workstream lift plan to bring the next
artifact (`v2.x`) into striking distance of the gate.

The single largest projected accuracy contribution in that plan
(+25–30pp) is **ASL Citizen** — a Microsoft Research isolated-sign
dataset of 83,912 clips across 2,731 signs from 52 signers, built
specifically for ML training of sign-language recognition models
(Desai et al. 2023, `arXiv:2304.05934`). Our 80 slice-1 signs have
**100% overlap** with ASL Citizen's vocabulary (verified at Phase
9b.2 against ASL-LEX 2.0, which is ASL Citizen's gloss source).
Expected per-sign yield is ~50 clips/sign — for the first time
reaching ADR 0006's 50–80 floor.

ADR 0008 named WLASL and MS-ASL as the public sources for slice-1
and explicitly skipped MS-ASL as "license-gated; skipped for slice
1 per ADR 0008 sufficiency check." That sufficiency check was the
1,107-clip → 17.86% outcome. The check has now been re-run with
the v1 result in hand, and ADR 0008's named sources are
demonstrably insufficient.

ASL Citizen is not a CC-BY public dataset. It is distributed under
the **Microsoft Research License Agreement (MSR-LA)**, verified
against Microsoft's published terms at
`https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/`
during Phase 9b.1 reconnaissance. The relevant provisions:

- Use is permitted **solely for non-commercial, non-revenue
  generating, research purposes.**
- **No redistribution** of the data or modifications to the data.
- **No sharing, publishing, distributing, or lending** the materials.
- Per-signer consent is via the IRB framework documented in the
  original paper; no individual opt-out mechanism is published.

A misreading of this license appeared in `TODO.md` Phase 9 — the
plan claimed CC-BY-4.0 and "better governance posture than ADR
0008's named sources." That claim is incorrect. The actual
governance posture is **comparable to** MS-ASL (which ADR 0008
skipped): a research-only license with no redistribution rights.

This ADR records the deliberate decision to include ASL Citizen
anyway for the slice-1 v2.x model, with the slice-2 cliff named
explicitly.

## Decision

**Include ASL Citizen as a training source for the slice-1 v2.x
classifier**, subject to the following constraints recorded in
the manifest, the validation report, and the README:

1. **Slice-1 pilot only.** The slice-1 pilot is a non-commercial,
   non-revenue-generating demo to a hiring partner. Training and
   demonstrating a model on ASL Citizen for that purpose fits
   within MSR-LA's "research purposes" clause. The slice-1
   evaluation is research evaluation, not commercial deployment.

2. **No clip redistribution.** The trained classifier is shipped
   as ONNX weights, not as a derivative of the source clips. No
   ASL Citizen video bytes are uploaded to R2, mirrored on a CDN,
   or otherwise redistributed by this project. The cleaning
   pipeline's intermediate `dataset/clean/v<N>/normalized_videos/`
   directory stays local to the training machine.

3. **No publication of a derivative dataset.** We do not publish
   our cleaned `dataset_v<N>_manifest.json` outside the project
   repository in a form that would constitute redistribution of
   ASL Citizen's contents. The manifest stores `source =
   "asl_citizen"` and clip identifiers; it does **not** mirror
   keypoint tensors derived from clips outside the training
   machine.

4. **Slice-2 cliff named.** If the project moves toward production
   deployment, paid users, or any monetized form (the slice-2
   transition), the v2.x model in its current form **cannot be
   used commercially under MSR-LA.** Slice 2 must do one of:
   - retrain the classifier with ASL Citizen excluded;
   - negotiate a commercial license with Microsoft Research; or
   - retrain on the ADR-0004 / ADR-0008 instructor-recorded
     training supplement (which the Deaf-instructor engagement
     produces and which has clean commercial-use terms by
     contract).

5. **Honest disclosure.** `docs/validation/v2.md` names this ADR
   by number, names the MSR-LA license verbatim, and discloses
   that the model cannot ship commercially in its current form.
   `README.md`'s "Honest scope disclosure" section is updated to
   name this ADR alongside ADRs 0004 and 0008.

6. **Citation.** Every artifact that quotes ASL Citizen numbers
   cites Desai et al. 2023 (`arXiv:2304.05934`) per the dataset
   page's standard request.

## Why not the other paths

**Drop ASL Citizen entirely.** The plan's pessimistic accuracy
projection without ASL Citizen is 50–60% top-1 on 75 signs — well
short of the 85% gate, even with all of Workstream 9a's pipeline
fixes. Shipping the slice-1 pilot under that ceiling repeats the
v1.0.1 honest-disclosure-of-failure pattern. The next slice-2
re-training cycle would then need to bring the model *both* up to
production quality *and* off the v2.x license posture — two
separate lifts. Including ASL Citizen now and naming the slice-2
re-train explicitly is the cleaner sequencing.

**Wait for the slice-2 Deaf-instructor recordings.** That work is
gated on the ADR-0004 instructor engagement, which is itself
gated on the pilot demonstrating enough promise to justify the
engagement. Bootstrapping requires a slice-1 model good enough
to be worth funding the engagement for. The chicken-and-egg
problem dissolves only if slice-1 includes ASL Citizen.

**Use a CC-BY alternative (How2Sign, ChicagoFSWild, BSL Corpus).**
Surveyed during 9b.1 reconnaissance:
- How2Sign is continuous sign-language translation, not isolated-
  sign recognition; the temporal segmentation doesn't match our
  pipeline.
- ChicagoFSWild is American fingerspelling, not isolated word
  signs.
- BSL Corpus is British Sign Language and would not transfer.
- Sem-Lex and PHOENIX are sentence-level or domain-specific.

No CC-BY isolated-sign dataset of comparable scale and signer
diversity exists at the time of this ADR. ASL Citizen is the
realistic option for v2.x.

## Consequences

**Slice-1 acceptance:**
- The slice-1 v2.x model is targetable at the 85% gate for the
  first time.
- The validation report has an additional named ADR (this one).
- The README discloses three license-shaped constraints (ADR
  0004 no-instructor, ADR 0008 no self-recording, ADR 0009
  research-only training data) instead of two.

**Slice-2 constraint:**
- The current v2.x weights cannot be shipped as a commercial
  product without a re-training cycle.
- The recording-tool work in ADR 0008's slice-2 framework is
  also the data-license cleanup path: instructor-recorded clips
  have clean contractual terms.

**Operational:**
- The 42.8 GB ASL Citizen ZIP stays on the training machine; not
  in any cloud bucket this project controls.
- The ingestion script (`training/data/ingest_asl_citizen.py`)
  records `source = "asl_citizen"` and the per-clip basename so
  every keypoint tensor's origin is traceable for license audit.

## How this is verified

- `docs/DATASET.md` §1a lists ASL Citizen with the MSR-LA terms
  named in the table.
- `dataset/raw/asl_citizen_manifest.json` (generated by the
  ingestion script) records the license string and citation in
  the manifest body.
- `docs/validation/v2.md` cites this ADR by number when reporting
  v2 numbers.
- The cleaning pipeline's manifest at `dataset/clean/v2/` records
  `source = "asl_citizen"` on every clip that originates from
  ASL Citizen, so a future license audit can enumerate which
  training inputs are MSR-LA-bound.
- `README.md`'s "Honest scope disclosure" section names this ADR
  in the same paragraph as ADR 0008's public-data-only
  disclosure.
