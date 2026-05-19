# Model

> Model architecture, training procedure, and evidence that no pretrained
> models were used.

---

## 1. Architecture

Small R(2+1)D-style 3D CNN.

- Input shape: `(B, 16, 112, 112, 3)` — batch, frames, height, width, channels.
- Pixel range: `[-1, 1]` after normalization. No ImageNet mean/std (we have no pretrained weights to match).
- Five R(2+1)D blocks with channel widths `[32, 64, 128, 192, 256]`.
- Each R(2+1)D block: 2D spatial conv (3×3) → BN → ReLU → 1D temporal conv (3) → BN → ReLU.
- Spatial stride 2 in every block; temporal stride 2 in blocks 2 and 4.
- Global average pool over spatial and temporal dimensions.
- Dropout 0.4.
- Linear layer → N classes (vocabulary size, 75–100).
- Total parameters: target ~3M.

All weights initialized from Kaiming-normal (`torch.nn.init.kaiming_normal_`) in the training code; biases initialized to zero. The init code is committed to the repository as the no-pretrained-weights evidence.

---

## 2. Training procedure

- Optimizer: AdamW, lr 3e-4, weight decay 1e-4.
- Scheduler: cosine annealing across the full run.
- Batch size: 32 (adjusted up/down to fit GPU).
- Epochs: 80 max, early stopping with patience 10 epochs on validation top-1.
- Loss: cross-entropy with label smoothing 0.1.
- Class balance: weighted sampling so each class appears roughly equally per epoch.
- Reproducibility: every Python random source seeded; W&B run records git commit, dataset version hash, full hyperparameter config.

Hardware: a single GPU is sufficient. RTX 4090 or A100 for training runs (rented cloud, hourly). Full training run: 4–12 hours depending on dataset size.

---

## 3. Augmentation stack

Training-time only. Applied per clip consistently across all 16 frames within a clip.

- Spatial random crop within 10% margin of the green box.
- Random brightness/contrast (±20%).
- Color jitter (small hue/saturation).
- Gaussian noise (small σ).
- Random gamma (simulates lighting variance).
- Temporal random crop (different 16-frame window from the 60-frame capture).
- Speed variation (±15%, via frame resampling).
- Frame dropout (occasional).
- Background swap using MOG2 mask from the empty-frame clip captured at session start. Classical CV, not a learned model.
- Horizontal flip — only on signs marked `flippable: true` in the vocabulary metadata. Two-handed asymmetric signs and signs whose handedness encodes meaning are never flipped.

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

1. Export from PyTorch to ONNX via `torch.onnx.export` with dynamic axes for batch dimension.
2. Verify ONNX output matches PyTorch float32 output within tolerance (max absolute error < 1e-4 on a 100-clip sample).
3. Apply post-training dynamic int8 quantization via `onnxruntime.quantization.quantize_dynamic`.
4. Verify quantized accuracy delta ≤ 1.0 percentage points versus float32 on the validation set. If the drop is larger, switch to quantization-aware training (QAT) in the next training run.
5. Final artifact target: ≤ 10 MB compressed.

---

## 7. No-pretrained-models evidence

Per the brief Requirement 7:

- The model architecture is implemented in this repository (`training/model/r2plus1d_small.py`) from primitives in `torch.nn`.
- Weight initialization uses `torch.nn.init.kaiming_normal_` for conv and linear layers. No `load_state_dict` call exists in the training entry point.
- No external weight files are downloaded, referenced, or required.
- The model is *not* a known landmark detector, pretrained classifier, or pretrained backbone.
- Programming frameworks (PyTorch, ONNX, ONNX Runtime) and data libraries (NumPy, OpenCV) are used as permitted by the brief.
- Classical CV algorithms (MOG2 background subtraction, optical flow) are used during augmentation only, not during inference. These are hand-coded algorithms, not learned models. Use is pending written confirmation (see `claude/CLAUDE.md`, Section 5).

A reviewer auditing the no-pretrained claim should look at: `training/model/`, `training/init.py`, and the absence of any pretrained-weight URLs in `training/config/`.

---

## 8. Performance targets

- First-visit model download (over reasonable broadband): ≤ 3 seconds.
- Cached-visit warm-up: ≤ 500 ms.
- Per-clip inference (16 frames, 112×112 input, WebGPU on mid-range integrated GPU): ≤ 300 ms.
- End-to-end "submit attempt" to result UI: ≤ 1 second.

If targets are missed, the response is to shrink the model — not to drop quality elsewhere.

---

## 9. Versioning

Every trained model has:

- A semantic-ish version string: `v{major}.{minor}.{patch}`.
- A content hash of the artifact.
- A bundled config (per-sign thresholds, class list, normalization params).
- A bundled validation report.
- A row in the `model_versions` table with `is_active` flag.

Promotion to active is manual for the pilot; the eval gate (`EVAL_GATE.md`) is the human's checklist. Slice-2 candidate: automated promotion gated by the same checklist in CI.
