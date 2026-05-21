# `/scripts/datasets` — external CV dataset download scripts

Per [ADR 0015](../../docs/decisions/0015-external-cv-datasets-provenance.md) and
the provenance audit at
[`docs/data/external_datasets_audit.md`](../../docs/data/external_datasets_audit.md).

Every script in this directory:

- Is a **no-op without `--confirm`** — running it prints the plan and exits.
- Writes to `data/external/<dataset_name>/` (gitignored under `data/*`).
- Verifies SHA256 checksums when upstream supplies them; otherwise records the
  on-disk SHA256 it observed.
- Writes `data/external/<dataset_name>/PROVENANCE.md` with source URL,
  download timestamp, license, archive SHA256s, originating paper, and the
  audit-memo entry name.
- Supports `--dry-run` to print every command without touching network or disk.

## Recommended scripts

| Script | Dataset | Auto-downloadable? | Approx size |
|---|---|---|---|
| `download_freihand.py` | FreiHAND (Zimmermann et al., ICCV 2019) | Yes | ~10 GB |
| `download_cmu_panoptic_handdb.py` | CMU Panoptic HandDB (Simon et al., CVPR 2017) | Yes (URLs may rotate — verify) | ~20 GB |
| `download_multiview_hand_pose.py` | Multiview Hand Pose (Gomez-Donoso et al., 2017) | V1 yes; V2 Google Drive manual | small–medium |
| `download_interhand26m.py` | InterHand2.6M `human_annot` (Moon et al., ECCV 2020) | Manual click-through (LICENSE accept) | ~35 GB at 5fps |
| `download_coco_wholebody.py` | COCO 2017 + COCO-WholeBody annotations | COCO yes; WholeBody JSON manual | ~19 GB |
| `download_mpii_pose.py` | MPII Human Pose (Andriluka et al., CVPR 2014) | Yes | ~12 GB |
| `download_wider_face.py` | WIDER FACE (Yang et al., CVPR 2016) | Annotations yes; images manual | ~2 GB |

## Usage

```bash
# Print the plan, do nothing
python scripts/datasets/download_freihand.py

# Dry-run — print every command without touching disk or network
python scripts/datasets/download_freihand.py --dry-run

# Actually download
python scripts/datasets/download_freihand.py --confirm
```

## Order of approval (recommended)

1. Read the audit memo first: `docs/data/external_datasets_audit.md`.
2. Read ADR 0015: `docs/decisions/0015-external-cv-datasets-provenance.md`.
3. Read ADR 0011's amended "Data sourcing — landmark training" section.
4. Decide which datasets to approve. For each approved dataset, run the
   corresponding script with `--confirm`.
5. After download, inspect every `data/external/<dataset_name>/PROVENANCE.md`
   and confirm the recorded SHA256s match what you expect.

## Rejected datasets

The following do **not** have download scripts here. They are documented in
the audit memo under "REJECTED summary":

- Ultralytics 26K Hand Keypoints — MediaPipe-labeled (disqualifying)
- BigHand2.2M — depth-only modality (wrong fit)
- AIST++ — unverified 2D-keypoint provenance
- Halpe-FullBody — unverified annotation methodology
- RWTH-PHOENIX-Weather — video corpus not labeled-keypoints; not downloadable
- Voxel51/hand-keypoints — duplicate of CMU manual subset
