# `/data` — local-only data storage (gitignored)

**Nothing here is committed except this README.** The raw clips, labels,
templates, and splits are large, so they live locally / on the Modal volume —
not in git. This file is the **index of what data the project uses and where it
comes from.**

Full provenance — every accepted *and* rejected dataset, with licenses and the
six acceptance criteria — is audited in
[`docs/data/external_datasets_audit.md`](../docs/data/external_datasets_audit.md),
governed by [ADR 0015](../docs/decisions/0015-external-cv-datasets-provenance.md).
The hard rule (ADR 0012): **no label is ever produced by a pretrained model** —
only humans, physical sensors, or multi-view fitting.

## 1. ASL signing clips — train the sign classifier

~12.7K clips across the 80-sign vocabulary, from public ASL corpora:

| Source | Clips (approx) | License |
|---|---|---|
| Sem-Lex | ~7,700 | CC BY-NC-SA |
| ASL Citizen | ~3,000 | MSR-LA (research) |
| WLASL | ~770 | research use |
| MS-ASL | ~450 | research use |
| Lifeprint (Dr. Bill Vicars) | ~235 | educational |
| Targeted YouTube | ~560 | per-clip |
| Self-recorded | ~100 | ours |

## 2. CV detector labels — train the from-scratch hand / pose / face models

Human-, sensor-, or multi-view-annotated public datasets (never a pretrained
model's output):

| Model | Label sources |
|---|---|
| Hand detector + 21-keypoint landmark regressor | FreiHAND, CMU HandDB, COCO-WholeBody, HaGRID (human-drawn boxes) |
| Pose detector | MPII Human Pose, COCO Keypoints |
| Face detector | WIDER FACE |

(Licenses + primary-source provenance for each → the audit doc above.)

## Folder layout

```
data/
├── raw_clips/        # the ASL clip corpus (section 1)
├── labeled_frames/   # hand-bbox, hand-keypoint, pose-keypoint, face-bbox labels
├── templates/        # per-sign trajectory stats (matcher baseline)
└── splits/           # signer-disjoint train / val / test manifests
```

## Provenance discipline

Every label set under `labeled_frames/` ships a `PROVENANCE.md` (source corpus,
labeling tool + rubric, labeler + date, per-class counts) — the audit surface
for each model card's "Pretrained components: none" declaration
([`docs/model_cards/`](../docs/model_cards)). If a model card or ADR names a
specific path, that path is authoritative.
