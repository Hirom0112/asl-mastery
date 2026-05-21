# v3.0 archive — landmark+CNN era (superseded by ADR 0011)

> Frozen 2026-05-21. Everything in this directory documents the
> pre-pivot machine-learning era of the project. Both the BiLSTM-on-
> keypoints architecture that shipped as v1.0.1 / v2.0.0 / v2.1.0
> (authorized by the since-superseded ADR 0006) and the never-trained
> 3D-CNN-on-raw-RGB architecture that ADR 0010 reverted to are
> preserved here as historical record. See [ADR 0011] for the pivot
> rationale and [ADR 0012] for the constraint perimeter that governs
> the new pipeline.

[ADR 0011]: ../../../docs/decisions/0011-pivot-to-landmarks-templates.md
[ADR 0012]: ../../../docs/decisions/0012-strict-from-scratch-cv-scoping.md

---

## Contents

```
artifacts/
  v1.0.0/  v1.0.1/  v1.0.2/  v1.0.3/   ← BiLSTM era (ADR 0006); v1.0.1 was the
                                          17.86 % top-1 baseline (docs/validation/v1.md)
  v2.0.0/  v2.1.0/                     ← BiLSTM era (ADR 0006); v2.x lifted to
                                          67.07 % top-1 via ASL Citizen + Sem-Lex
                                          (docs/validation/v2.md). v2.1.0 was the
                                          last promoted model before the 2026-05-20
                                          deactivation (ADR 0010 T1, commit 06b3004).

code_v1_bilstm/                        ← Training code as it stood at 9fbe061 (the
                                          T0 baseline). Frozen copy — the live
                                          tree's training/ has since been rewired
                                          for the 3D CNN attempt then deleted by
                                          ADR 0011 entirely.

code_v3_3dcnn_unbuilt/                 ← T4 rebuild of the training pipeline
                                          (SmallR2Plus1D 3D CNN over raw RGB)
                                          per the now-superseded ADR 0010. Never
                                          produced a trained model — every smoke
                                          run crashed during setup (PyAV → cv2
                                          loader migration was the last fix
                                          before cancellation).

manifests/
  slice1_vocabulary.json               ← 80-sign vocab filter (v1 era)
  slice1b_vocabulary.json              ← 75-sign vocab filter (v2/v3 era, post-9c trim)
  dataset_v3_manifest.json             ← 11,153 cleaned-clip records, 75 classes,
                                          7931/1770/1452 train/val/test, produced
                                          by the v3 cleaning run that completed at
                                          2026-05-20 23:33 CDT.
  dataset_v3_smoke_manifest.json       ← 2-sign smoke subset (eat + water)
  splits/                              ← Frozen signer-disjoint split JSONs

runs/
  v1-001 / v1-002 / v1-003 / v1-004    ← run.json + validation.json for each
  v2-001 / v2-002 / v2-005 / v2-007    ← BiLSTM-era training run. The .pt
  v2-008 / v2-009                        checkpoints stay outside git (gitignored
                                          per repo policy "never commit raw clips
                                          or trained weights") but are still on
                                          local disk at /Users/hirom/.../runs/.

20260520200000_deactivate_all_models.sql   ← T1 production-honesty migration
                                              (06b3004). All v1/v2/v2.1 rows
                                              flipped to is_active=false.
20260521000000_promote_v3_0_0.sql.template ← Promotion template that v3.0 never
                                              triggered.
```

## What is *not* in this archive

- **Trained model weights (.pt, .onnx) larger than text**: preserved on
  local disk under `artifacts/` and `runs/` (gitignored). The
  text-shaped artifacts (config.json, manifest.json, validation.json)
  are in this archive and git-tracked.
- **Raw training clips** (WLASL + ASL Citizen + Sem-Lex MP4s):
  preserved on the Modal volume under `/dataset/raw/` and on local
  disk under `dataset/raw/`. License-bound per ADR 0008 + ADR 0009;
  never redistributed.
- **Cleaned per-clip MP4s** for v3 (11,153 files): preserved on the
  Modal volume under `/datasets/v3/normalized_videos/`. Repurposable
  for the new landmark-pipeline's labeling and template-extraction
  workflows per ADR 0011.

## Cancellation record

- **v3.0 training cancelled 2026-05-21 ~00:05 CDT** after Modal apps
  `ap-S64dhuRHqpdgG3D6d73W45`, `ap-3mU0Sns7mc68KCO0DcovQo`, and
  `ap-fqz9wvDAHSolheD4AsGh9Z` (smoke-train attempts on `v3-smoke` /
  2-sign subset, 5 epochs, batch 16, L4 GPU). Total wall-clock
  compute: ~2 minutes across all attempts.
- **No epoch ever completed** — every smoke crashed during setup
  (PyAV import error → torchvision-PyAV mismatch → cv2 loader fix
  → modal_app.py mid-build edit detection → final cancellation).
- **No checkpoint promoted to production.** The practice screen
  continues to serve the deterministic stub + ADR 0010 offline
  banner from T1 (commit 06b3004); ADR 0011 inherits that state.
- **Cleaning run completed normally** at 2026-05-20 23:33 CDT
  (`ap-p3EDfrX8xfa9Wa1PlTWfQy`, after the None-local_path fix in
  30c43ff). Output preserved for re-use.
