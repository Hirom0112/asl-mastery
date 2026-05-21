# `/data` — local-only dataset and labeled-artifact storage

Per `docs/decisions/0012-strict-from-scratch-cv-constraint.md` and
the project structure in `docs/VOCABULARY_TRAINER_ROADMAP.md`.

**Gitignored.** Nothing in here is committed except this README.

## Layout

```
data/
├── raw_clips/        # Modal-cleaned 12K-clip corpus (mirrors / supersedes dataset/clean)
├── labeled_frames/   # Hand-bbox, hand-keypoint, pose-keypoint, face-bbox labels (CVAT / LabelStudio output)
├── templates/        # Per-sign mean trajectory + per-keypoint per-timestep variance (Phase 4 output)
└── splits/           # train / val / test manifests (signer-disjoint per docs/DATASET.md §4)
```

## Equivalence with existing dirs

The pre-pivot project already has `dataset/` (raw clips + vocabulary
manifests) and `artifacts/` (deactivated model weights) at the repo
root. Those remain in place and gitignored. New from-scratch CV
labeling and template work lives under `/data` per the roadmap
structure block.

If a label set or split is referenced by a model card or ADR, the
path in the doc is authoritative — do not rely on convention alone.

## Provenance

Every label set under `labeled_frames/` must ship with a
`PROVENANCE.md` declaring:

- Source corpus (e.g. WLASL slice, self-recorded, ASL Citizen)
- Labeling tool + rubric version
- Labeler identity + date range
- Number of items + per-class breakdown

This is the audit surface for ADR 0012 Rule 5 ("every model card
declares pretrained components: none, alongside training data
provenance").
