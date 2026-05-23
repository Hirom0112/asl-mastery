# Post-encoder launch commands — ready to fire

All commands require explicit spend approval. Do not run without confirming
cost and Hiromu's go.

## Prerequisite verification (free, ~30 s)

Confirm encoder ckpt landed and v4 manifest is on volume:

```bash
/Users/hirom/Library/Python/3.9/bin/modal volume ls asl-mastery-data /runs/handshape_v0d
/Users/hirom/Library/Python/3.9/bin/modal volume ls asl-mastery-data /labeled_frames
```

Expected:
- `/runs/handshape_v0d/best.pt` exists
- `/labeled_frames/unified_clip_manifest_modal_v4.json` exists

---

## Step 1 — Linear probe diagnostic (free, ~5 min, decides spend)

Quick sanity check before the $0.80 extract. Pulls the encoder ckpt locally,
embeds a small batch of hand crops, trains a tiny linear classifier on
sign_id labels, reports top-1 accuracy. If accuracy is meaningfully above
chance (~1/79 ≈ 1.3%), the encoder learned useful representations.

```bash
/Users/hirom/Library/Python/3.9/bin/modal volume get asl-mastery-data \
  /runs/handshape_v0d/best.pt artifacts/ckpts/handshape_v0d_best.pt --force

training/.venv/bin/python -m scripts.linear_probe_encoder \
  --encoder-ckpt artifacts/ckpts/handshape_v0d_best.pt \
  --crops-root data/hand_crops \
  --n-per-sign 50 \
  --epochs 30
```

Decision gate:
- top-1 < 5%: encoder is broken; debug before spending on extract
- 5-15%: encoder works but marginally; v4 retrain may not move ceiling much
- 15%+: strong signal; proceed to extract + retrain with confidence

## Step 2 — Final v4 trajectory extract WITH encoder (~$0.80, ~50-60 min L4)

```bash
/Users/hirom/Library/Python/3.9/bin/modal run training/modal_app.py::extract_trajectories_v2_l4 \
  --manifest /labeled_frames/unified_clip_manifest_modal_v4.json \
  --out-dir /trajectories_v8 \
  --hand-detector-ckpt /runs/hand_det_v2_hagrid_fromscratch/best_ema.pt \
  --hand-landmarks-ckpt /runs/hand_landmarks_v0_20260521_105603Z/best.pt \
  --pose-ckpt /runs/pose_v0_h100_20260521_164906Z/best.pt \
  --handshape-encoder-ckpt /runs/handshape_v0d/best.pt \
  --fps 15.0 \
  --decode-workers 12
```

Estimated: 12,752 clips × ~5s each / 12 decode workers ≈ 90 min wall-clock.
Cost: L4 ($0.80/hr) × 1.5h = ~$1.20. Higher than v3 ($0.48) because:
- 60% more clips than v3's 7,397
- Encoder inference adds ~20% per-clip time

Output: `/trajectories_v8/<sign>/<clip_stem>.json` — ~12K JSONs with 128D
embeddings per detected hand. Adds ~12K inodes to volume (currently ~434K,
hard cap 500K — still safe; tar after extract if you want extra margin).

## Step 3 — (Optional) Tar v8 trajectories + reclaim inodes (~$0.02)

```bash
/Users/hirom/Library/Python/3.9/bin/modal run training/modal_app.py::tar_and_cleanup_dir \
  --src-dir /trajectories_v8 \
  --tar-out /trajectories_v8.tar \
  --delete-after False
```

Tars but keeps the dir. Pass `--delete-after True` only after confirming the
classifier retrain succeeds.

## Step 4 — Final classifier retrain on 356D features (~$0.40, ~5-10 min A100)

Reuses the existing entrypoint. Auto-detects 356D feature dim from the v8
trajectories.

```bash
/Users/hirom/Library/Python/3.9/bin/modal run training/modal_app.py::train_sign_classifier_a100 \
  --trajectories-dir /trajectories_v8 \
  --vocab-json /vocabulary/slice1b_vocabulary.json \
  --run-id sign_classifier_v8_356d_tcn \
  --epochs 80 \
  --early-stop-patience 10 \
  --arch tcn
```

For a clean architecture A/B at 356D, also run:

```bash
/Users/hirom/Library/Python/3.9/bin/modal run training/modal_app.py::train_sign_classifier_a100 \
  --trajectories-dir /trajectories_v8 \
  --vocab-json /vocabulary/slice1b_vocabulary.json \
  --run-id sign_classifier_v8_356d_xformer \
  --epochs 80 \
  --early-stop-patience 10 \
  --arch transformer \
  --transformer-d-model 256 \
  --transformer-layers 4 \
  --transformer-heads 8
```

## Step 5 — Pull the trained classifier ckpt + results

```bash
/Users/hirom/Library/Python/3.9/bin/modal volume get asl-mastery-data \
  /runs/sign_classifier_v8_356d_tcn/best.pt \
  artifacts/ckpts/sign_classifier_v8_356d_tcn.pt --force

/Users/hirom/Library/Python/3.9/bin/modal volume get asl-mastery-data \
  /runs/sign_classifier_v8_356d_tcn/per_class.json \
  artifacts/runs/v8_tcn_per_class.json --force
```

## Decision matrix after Step 4

| TCN top-1 vs prior 11.1% baseline | Read | Next |
|---|---|---|
| 20-30% | Handshape encoder is the architectural unlock | Move to live-demo + browser inference (Phase 5) |
| 14-19% | Real lift but more to do | Try transformer A/B; consider face encoder (Phase 4.5) |
| 12-14% | Marginal; data + encoder gave us ~3pp | Need bigger architectural change (multi-stream fusion 4.8) |
| 11% or worse | Encoder didn't help OR per-class collapsed | Debug: per-class confusion matrix, t-SNE viz of embeddings |

## Total Modal spend after this session

| Item | Cost |
|---|---|
| Phase 4.7 transformer A/B (Session 19) | ~$0.50 |
| Encoder training (v0a/b/c crashes + v0d success) | ~$3.50 |
| MSAsl Modal scrape (failed, killed) | ~$0.05 |
| Final v4 extract (this run) | ~$1.20 |
| Final classifier retrain (this run, both A/B) | ~$0.80 |
| **Session 19 total** | **~$6.05** |
