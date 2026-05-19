# Model

> Model architecture, training procedure, and evidence that no pretrained
> models were used.

---

## 1. Architecture

Two-stage pipeline (per ADR 0006): pretrained landmark extractor → temporal classifier trained from scratch.

**Stage 1: Landmark extraction (pretrained, used as a library).**

- **MediaPipe Holistic**, loaded via MediaPipe Tasks Web in the browser at inference and via the Python MediaPipe SDK in the training cleaning pipeline.
- Keypoint subset consumed:
  - 21 left-hand landmarks, each `(x, y, z)`.
  - 21 right-hand landmarks, each `(x, y, z)`.
  - A small upper-body pose subset (shoulders, elbows, wrists, and torso anchors). Exact pose indices confirmed during Phase 4 once we measure which contribute to per-sign accuracy.
  - Face landmarks deferred to slice 2 (non-manual markers).
- Output per frame: a flattened keypoint vector. Output per clip: a `(T, K)` tensor where `T` is the temporal length (default 16 frames, matching the 2-second capture window resampled to 30 fps then subsampled) and `K` is the total keypoint coordinate count after the subset is fixed.
- Permitted per ADR 0006. MediaPipe is treated as a black-box library; we do not load any pretrained ASL classifier or any pretrained component that is ASL-specific.

**Stage 2: Temporal classifier (trained from scratch).**

- **Baseline:** 2-layer bidirectional LSTM over the keypoint sequence.
  - Input: `(B, T, K)`.
  - LSTM hidden size 128, dropout 0.3 between layers.
  - Mean-pool over the temporal axis, then a linear layer to N classes (vocabulary size, 75–100, see `docs/VOCABULARY.md`).
  - Total parameters: target ~200K.
- **Alternative:** small Transformer encoder.
  - 4 encoder layers, 4 attention heads, d_model 128, feed-forward 256.
  - Sinusoidal positional encoding over the temporal axis.
  - CLS-token pool, linear to N classes.
  - Total parameters: target ~500K.
  - Adopted if the BiLSTM underperforms during Phase 4 iteration.

All classifier weights initialized from Kaiming-normal (`torch.nn.init.kaiming_normal_`) for linear and projection layers; LSTM weights via PyTorch's default orthogonal/xavier init; biases zero. The init code is committed to the repository as the no-pretrained-pipeline evidence — see §7.

---

## 2. Training procedure

- Optimizer: AdamW, lr 1e-3 (BiLSTM) or 3e-4 (Transformer), weight decay 1e-4.
- Scheduler: cosine annealing across the full run.
- Batch size: 128 (keypoint tensors are small; large batches fit easily).
- Epochs: 60 max, early stopping with patience 8 epochs on validation top-1.
- Loss: cross-entropy with label smoothing 0.1.
- Class balance: weighted sampling so each class appears roughly equally per epoch.
- Reproducibility: every Python random source seeded; W&B run records git commit, dataset version hash, MediaPipe version, full hyperparameter config.

Hardware: a single small GPU is sufficient, and CPU training is feasible for the BiLSTM. Expected training time per run: **minutes, not hours** — the classifier is small and the inputs are low-dimensional. GPU rental cost per run drops to near-zero compared to the Path B target in the superseded ADR 0001.

---

## 3. Augmentation stack

Training-time only. Applied at the **keypoint level**, after MediaPipe extraction, not at the pixel level. The pixel-level augmentations that served the superseded ADR 0001 architecture (color jitter, brightness, gamma, Gaussian noise on frames, background swap via MOG2) are deprecated; they are not meaningful on keypoint inputs because MediaPipe has already normalized away most of what they were fighting.

- **Per-keypoint coordinate jitter.** Small Gaussian noise added to each `(x, y, z)` coordinate (σ ≈ 0.005 in normalized space). Models the noise floor of MediaPipe detection.
- **Temporal stretch.** Resample the keypoint sequence to a slightly longer or shorter temporal length (±15%) before re-cropping back to `T = 16`. Models variation in signing speed.
- **Temporal random crop.** Slide the 16-frame window within a longer captured sequence.
- **Keypoint dropout.** With small probability, zero a random subset of keypoints in a random subset of frames. Models partial occlusion (e.g. one hand off-camera briefly).
- **In-plane rotation.** Apply a small 2D rotation (≤ ±10°) to all `(x, y)` coordinates jointly. Models camera tilt.
- **Horizontal flip via x-coordinate negation.** Negate `x` for all keypoints and swap left-hand / right-hand keypoint groups. Applied only on signs marked `flippable: true` in `docs/VOCABULARY.md`. Two-handed asymmetric signs and signs whose handedness encodes meaning are never flipped, per the rule in `docs/VOCABULARY.md`.

Classical CV augmentation (MOG2 background swap, etc.) remains permitted per ADR 0005 but is not in the slice-1 augmentation stack — under the landmark-based architecture it is unnecessary.

---

## 4. Calibration

- Temperature scaling on validation set after training. A single scalar `T` minimizes negative log-likelihood when logits are divided by `T` before softmax.
- Per-sign confidence threshold derived from validation precision-recall curves. Target: ≥90% precision per sign on the "pass" decision.
- Retry-leniency: on second and third attempts at the same prompt within a session, threshold drops slightly (decay schedule defined in app config). Prevents demoralizing repeat failures while keeping the first-attempt bar honest.

---

## 5. Confusion analysis

After training, the validation confusion matrix is dumped as `confusion_matrix.json`. The top 2–3 confusion targets per sign are extracted to drive the hint system (`confusion_pair_hints` table). Hints for each pair are authored against ASL-LEX 2.0 parameter codes for slice 1 (per ADR 0004); Deaf-signer validation is a slice-2 production-deployment requirement.

---

## 6. Export and quantization

Our classifier is small (≈ 200K params for the BiLSTM, ≈ 500K for the Transformer alternative). MediaPipe ships its own client runtime separately and is **not** quantized by us; we consume its Tasks Web build as-is.

1. Export the classifier from PyTorch to ONNX via `torch.onnx.export` with dynamic axes for the batch dimension and the temporal axis.
2. Verify ONNX output matches PyTorch float32 output within tolerance (max absolute error < 1e-4 on a 100-clip keypoint-sequence sample).
3. Quantization is optional under the new architecture — the float32 classifier is already well under 1 MB. We default to **shipping float32** and only quantize if Phase 4 measurement shows a meaningful win.
4. **Classifier artifact target: ≤ 1 MB.** Combined client bundle (MediaPipe Tasks Web runtime + classifier) target: **≤ 5 MB**.

---

## 7. No-pretrained-pipeline evidence

Brief Requirement 7 was clarified by Gauntlet staff on 2026-05-19 (see ADR 0006). The clarification states that Requirement 7 restricts pretrained ASL pipelines and pretrained sign classifiers, not pretrained general-purpose landmark detectors. This section is rewritten against that clarified scope.

- **MediaPipe Holistic is used for landmark extraction.** This is permitted per the 2026-05-19 clarification (ADR 0006). MediaPipe is a general-purpose hand and pose landmark detector; it is not an ASL-specific component.
- **The classifier — the actual ASL recognition logic — is trained entirely from scratch** with Kaiming initialization. No pretrained sign classifier, no pretrained ASL feature extractor, and no ASL-specific weights are loaded at any point.
- The classifier architecture is implemented in this repository (`training/classifier/`) from primitives in `torch.nn`. Weight initialization uses `torch.nn.init.kaiming_normal_` for linear and projection layers; biases zero; LSTM weights via PyTorch's default orthogonal/xavier init.
- No `load_state_dict` call exists in the training entry point for the classifier. No external classifier weight files are downloaded, referenced, or required.
- Programming frameworks (PyTorch, ONNX, ONNX Runtime, MediaPipe SDK) and data libraries (NumPy, OpenCV) are used as permitted by the brief.
- Classical CV algorithms remain permitted (see ADR 0005) but are not load-bearing for slice-1 augmentation under the landmark-based architecture.

A reviewer auditing the no-pretrained-pipeline claim should inspect: `training/classifier/`, `training/init.py`, and the absence of any external classifier weight URLs in `training/config/`. MediaPipe's presence in `package.json` and in the runtime bundle is the visible evidence that the 2026-05-19 clarification was applied; ADR 0006 is the written authorization for that dependency.

**Note on training-data authorship (ADR 0008).** The keypoint tensors fed to this classifier originate from WLASL and MS-ASL public clips only for slice 1 — no project-team-recorded clips reach the training set. This does not affect the no-pretrained-pipeline argument: the keypoints are extracted by MediaPipe (the only pretrained component, permitted under ADR 0006) and the classifier still trains from scratch on those keypoints. ADR 0008 documents the data-authorship scope; ADR 0006 documents the pretrained-component scope. The two are independent.

---

## 8. Performance targets

Under the landmark-based architecture (ADR 0006) the deployed bundle is dramatically smaller and inference is faster than the Path B targets in the superseded ADR 0001.

- **First-visit client bundle download** (MediaPipe Tasks Web + our classifier, combined): ≤ 3 seconds over reasonable broadband. Combined size target ≤ 5 MB.
- **Cached-visit warm-up:** ≤ 500 ms.
- **Per-clip classifier inference** (keypoint sequence in, logits out, WebGPU or WASM): ≤ 100 ms.
- **MediaPipe extraction over the 2-second capture window** (browser, WebGL/WebGPU backend, mid-range integrated GPU): ≤ 400 ms.
- **End-to-end "submit attempt" to result UI:** ≤ 600 ms.

If targets are missed, the response is to shrink the classifier or reduce the keypoint subset — not to drop quality elsewhere.

---

## 9. Versioning

Every trained model has:

- A semantic-ish version string: `v{major}.{minor}.{patch}`.
- A content hash of the artifact.
- A bundled config (per-sign thresholds, class list, normalization params).
- A bundled validation report.
- A row in the `model_versions` table with `is_active` flag.

Promotion to active is manual for the pilot; the eval gate (`EVAL_GATE.md`) is the human's checklist. Slice-2 candidate: automated promotion gated by the same checklist in CI.
