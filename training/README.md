# Training pipeline

Python-side of ASL Mastery: ingestion of public datasets (WLASL,
MS-ASL), cleaning, MediaPipe keypoint extraction, classifier
training, validation, and ONNX export.

This directory exists in the same repository as the Next.js learner
app so that:

- The keypoint subset is defined exactly once (`lib/keypoints.ts`
  is the source of truth; Python re-derives the same constants in
  `training/keypoints.py`).
- The MediaPipe version is pinned in both places.
- The model artifact produced by `training/classifier/export.py`
  is the artifact the browser loads at inference.

---

## Architecture references

- `docs/ARCHITECTURE.md` §2.5 — training pipeline stages.
- `docs/MODEL.md` — classifier spec, augmentation, calibration,
  ONNX export, no-pretrained-pipeline evidence.
- `docs/DATASET.md` — sources, cleaning pipeline, splits, fairness.
- `docs/EVAL_GATE.md` — promotion criteria the validation report
  must meet.
- ADR 0006 — landmark-based recognition architecture.
- ADR 0008 — public-data-only training for slice 1.

---

## Phases and entry points

Each entry point is a single command, reproducible from a fresh
clone with `pip install -r requirements.txt`.

**Local vs Modal split:**

- **Local (your laptop):** Phase 3d ingestion (`yt-dlp` works better from a residential IP) and Phase 3f cleaning (CPU-bound, no GPU need).
- **Modal:** Phase 4 training + validation + ONNX export — reproducible GPU container, no laptop fan, identical environment every run. See `training/modal_app.py`.

```
# Phase 3d — public dataset ingestion (per ADR 0008)
python -m training.data.ingest_wlasl --output dataset/raw/
python -m training.data.ingest_msasl --output dataset/raw/

# Phase 3d — apply the slice-1 per-sign clip-count filter
python -m training.data.filter_vocabulary \
  --raw dataset/raw/ \
  --vocabulary ../docs/VOCABULARY.md \
  --floor 15 \
  --output dataset/slice1_vocabulary.json

# Phase 3f — clean + extract keypoints
python -m training.data.clean --version v1

# Phase 4a — train
python -m training.classifier.train \
  --dataset-version v1 \
  --model bilstm  # or transformer

# Phase 4b — validate
python -m training.classifier.validate \
  --dataset-version v1 \
  --checkpoint runs/<run-id>/best.pt

# Phase 4d — export ONNX
python -m training.classifier.export \
  --checkpoint runs/<run-id>/best.pt \
  --output artifacts/v1.0.0/
```

### Modal path for Phase 4 (recommended)

```sh
# One-time per machine.
modal token new

# Push the cleaned dataset to a Modal volume.
modal volume create asl-mastery-data        # idempotent
modal volume put asl-mastery-data dataset/clean/v1 /datasets/v1

# Train + validate + export on Modal in one call.
modal run training/modal_app.py \
    --manifest /datasets/v1/dataset_v1_manifest.json \
    --run-id v1-001 \
    --artifact-version v1.0.0

# Pull the artifact bundle back down.
modal volume get asl-mastery-data /artifacts/v1.0.0 ./artifacts/v1.0.0

# Upload to R2 and insert the model_versions row to flip the
# /practice screen off its stub classifier.
wrangler r2 object put asl-mastery-models/v1.0.0/classifier.onnx ...
# (full upload + INSERT documented in TODO.md Phase 4d)
```

---

## What this directory does NOT contain

- **Pretrained classifier weights.** None. Every weight is
  initialized from scratch (`torch.nn.init.kaiming_normal_` for
  linear/projection layers; PyTorch defaults for LSTM internals).
  See `training/classifier/init.py` and `docs/MODEL.md` §7.
- **Pretrained image backbones.** None. MediaPipe is the only
  pretrained component, and it is used as a black-box landmark
  extractor (ADR 0006).
- **Self-recorded clips.** Per ADR 0008, slice-1 training data is
  drawn from public datasets only. The recording tool stays
  unimplemented in slice 1.

---

## Reproducibility contract

Every training run records:

- Git commit (HEAD short SHA).
- Dataset version hash.
- MediaPipe version (must match `MEDIAPIPE_VERSION` in
  `lib/mediapipe/loader.ts`).
- Full hyperparameter config.
- Random seed.

Anyone with the repo + dataset version + run id can re-derive the
same model.
