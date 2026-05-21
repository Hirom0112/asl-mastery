# Labeling rubric — hand bounding boxes (Phase 1, Slice 1.1)

> Used by the labeler (you, initially) to produce the dataset that
> trains the from-scratch hand detector. Consistency beats perfection.
> Read once, then label.

## Scope

- **Task:** draw an axis-aligned bounding box around each visible hand.
- **Target dataset size:** 3,000 frames from the corpus + 500–1,000
  self-recorded frames covering varied lighting, distance, background.
- **Per-frame target:** every visible hand boxed, including partial
  hands at frame edges. Two-hand signs produce two boxes per frame.

## What counts as a hand

- A "hand" is the wrist plus everything distal: palm, dorsum, fingers,
  thumb. The bounding box wraps the visible extent of these parts.
- The wrist crease is the proximal boundary. Do not include forearm
  pixels beyond the wrist.

## What to box (positive examples)

- **Fully visible hand**, any pose, palm or dorsum facing camera.
- **Partial hand at frame edge** — box the visible portion only.
- **Occluded hand** where ≥30% of the hand surface is visible — box
  the visible portion. If a finger is hidden behind another, the box
  still wraps the visible silhouette only.
- **Two hands touching or overlapping** — produce two separate boxes,
  one per hand, even when they share pixels. The boxes may overlap.
- **Motion-blurred hand** — box the blur envelope.

## What NOT to box (negative examples)

- **Fully occluded hand** (e.g. behind the signer's head) — no box.
- **Hand of a person in the background** other than the signer — no
  box. The detector is intended for the foreground signer; background
  hands are noise.
- **Forearms, elbows, shoulders** — out of scope for this dataset.
- **Hand silhouettes in posters, logos, prints on clothing** — no
  box. Only real flesh-and-bone hands.

## Box tightness

- **Tight:** the box should hug the visible hand within ~5 pixels at
  the source resolution. Inconsistent tightness is the most common
  labeling pathology — pick "snug" and stay there.
- **Do NOT pad** to "make sure you got it all." A consistently tight
  box trains a more precise detector than an inconsistently loose one.

## Edge cases

- **Sign with the hand in front of the face:** box the hand, ignore
  the face.
- **Sign with the hand at the body (chest, chin):** box the hand,
  do not extend the box to include body pixels.
- **Hands held together as one shape** (e.g. classifier signs):
  produce two boxes, one per hand, even if the boxes are nearly
  identical. The detector is hand-instance based.
- **Fingerspelling:** out of scope for the vocabulary
  (`docs/VOCABULARY.md` — alphabet excluded), but if a fingerspelled
  hand appears in a sampled frame, box it anyway. The detector is
  vocabulary-agnostic.

## Calibration set

Before labeling at scale, label the same **50 reference frames** at
the start of every labeling session. Compare with your prior pass.
If your box positions drift > 5 pixels on average from the prior
pass, recalibrate by re-reading this rubric before continuing.

The calibration set lives at
`data/labeled_frames/hand_bbox/_calibration/` with the reference
labels in `_calibration/reference_labels.json`. Drift report goes
into the per-session log under `labeling/runs/<date>_hand_bbox.md`.

## Tooling

- **Labeling tool:** CVAT or LabelStudio (decision pending in Slice 1.1).
  Both export to formats that normalize to the trainer's manifest
  schema (see `training/detectors/dataset.py`).
- **Project-internal export format:** flat JSON, version 1, with
  `image_path`, `width`, `height`, `bboxes` (list of `[x0, y0, x1, y1]`
  in input-image pixel coordinates). Schema validated by
  `training/detectors/dataset.load_manifest`.

## Provenance

Every labeling batch ships with a `PROVENANCE.md` in its
`data/labeled_frames/hand_bbox/<batch_id>/` directory recording:

- Source-frame range from `source_pool/index.json`
- Labeling-tool version
- Labeler identity (initially: yourself)
- Start + end timestamps
- Number of frames + number of boxes
- Any drift-recalibration events during the session

This is the audit surface for ADR 0012 Rule 5 ("every model card
declares pretrained components: none, alongside training data
provenance").
