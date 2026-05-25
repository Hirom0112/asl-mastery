# Model card — hand_landmarks_3d v0

> Per ADR 0012 Rule 5, every model card declares pretrained components,
> alongside architecture, parameter count, training data provenance,
> training process, validation metrics, and known failure modes.

## Overview

- **Model:** `hand_landmarks_3d_v0`
- **Task:** regress 21 hand keypoints as **2D** crop-relative (x, y) **plus
  per-keypoint depth z** (root-relative to the wrist, scale-normalized by the
  3D palm bone ‖kp9 − kp0‖). First from-scratch 3D-capable hand landmark
  regressor in this project — trained from scratch, no pretrained weights/labels.
- **Status:** trained — first 3D production checkpoint
- **Checkpoint:** `/runs/hand_landmarks_3d_v0/best.pt` (Modal volume
  `asl-mastery-data`); durable copy `/models/hand_landmarks_3d_v0/best.pt`;
  local `data/ckpts_new/hand_landmarks_3d_v0_best.pt`
- **Architecture file:** `training/detectors/hand_landmarks.py` (`predict_z=True`)
- **Training entrypoint:** `training/modal_app.py::train_hand_landmarks_3d` (A100);
  manifest builder `training/modal_app.py::build_freihand_3d_manifest`
- **Governing ADRs:** [0011](../decisions/0011-landmarks-and-templates-pivot.md), [0012](../decisions/0012-strict-from-scratch-cv-constraint.md), [0015](../decisions/0015-external-cv-datasets-provenance.md)

## Pretrained components

**None.** Every weight trained by this project (Kaiming-normal conv/linear,
zeros biases, ones BatchNorm). No `torchvision.models`, no foreign
`load_state_dict`. Identical backbone to `hand_landmarks_v0`; adds a depth head.

## Architecture

- Same convolutional backbone as `hand_landmarks_v0` (stem + 4 residual
  stages, GAP, 256-dim pooled feature).
- **Heads:** `coord_head` → 21×2 in [0, 1] (sigmoid); `vis_head` → 21 logits;
  **`depth_head` → 21 signed values (Linear 256→256→21, NO sigmoid)** — z is
  root-relative and scale-normalized, so it is unbounded/signed.
- Input: 224 × 224 RGB hand crop, [0, 1].
- **Parameter count:** 4,357,428 (~4.36M) — ~71k more than the 4.29M 2D model
  (the depth head only).

## Training data

- **Source:** FreiHAND only (it ships true 3D MANO joints + camera intrinsics;
  multi-view fit, vetted under ADR 0015). CMU HandDB is 2D-only in our loaders
  so it was not used here (the loss masks z for any 2D-only source via
  `has_depth`, so CMU can be folded in later for extra x,y supervision).
- **Depth target:** `z_i = (xyz[i].z − xyz[0].z) / ‖xyz[9] − xyz[0]‖` —
  wrist-relative, scaled by the 3D palm bone, matching the v3 feature schema
  ("21×3 wrist-rel scaled"). Dimensionless, signed, wrist (kp0) z ≡ 0.
- **Manifests:** `/labeled_frames/hand_keypoints/freihand_3d_{train,val}.json`,
  built by `build_freihand_3d_manifest` from the proven `external_freihand.json`
  (correct absolute volume paths + bboxes + 2D kps) augmented with `keypoints_z`.
- **Split:** 32,560 unique samples, deterministic — val = `idx % 20 == 0`
  (1,628), train = the rest (30,932). No augmented-copy leakage (unique-only).
- **Provenance audit:** `docs/data/external_datasets_audit.md` (FreiHAND entry).

## Training process

- **Loss:** coord L1 (visibility-masked) + 0.1·visibility BCE + `z_weight`(1.0)·
  depth L1 (masked by `visibility × has_depth`).
- **Depth is augmentation-invariant** — it is root-relative + scale-normalized
  about the optical axis, so hflip/rotation/scale/translate leave it unchanged;
  z rides through the existing 2D augment untouched.
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4. **Schedule:** cosine, 60
  epochs. **Grad clip:** max-norm 5.0. **Batch:** 256. **Workers:** 16.
- **GPU:** Modal A100. **Wall time:** ~45 min (~44 s/epoch). **Cost:** ~$2.

## Validation

- **Held-out:** `freihand_3d_val.json` (1,628 FreiHAND samples).
- **Best epoch:** 59 (min val_loss 0.1394).
- **2D mean keypoint px error:** **13.82 px @ 224** (trajectory 41.3 → 13.82,
  monotonic). **Depth mean abs error:** **0.0997** in normalized palm-length
  units (≈ 0.10 palm-lengths per keypoint).
- **vs `hand_landmarks_v0`:** v0 (2D-only, FreiHAND **+ CMU**) = 12.82 px on the
  combined val. This model (FreiHAND-**only**, plus a depth head) = 13.82 px 2D —
  comparable 2D accuracy on less data, **plus a working depth signal that did
  not exist before.** Both remain above the < 8 px roadmap ceiling, consistent
  with the documented from-scratch-constraint price; folding CMU back in for
  x,y and/or a longer/larger run are the levers if 2D px needs to drop.

## Known failure modes

- Inherits `hand_landmarks_v0`'s 2D modes (motion-blur fingertips, occluded
  thumb regressing toward mean).
- **Depth is monocular-inferred** (no stereo/multi-view at inference): absolute
  depth ordering of near-coplanar fingers is the weak spot. z is most reliable
  for gross flexion/extension (curl) — which is exactly the linguistic signal
  the v3 upgrade targets (A/S/E/N/M curl letters, chin/forehead signs).
- Skin-tone subgroup error not yet measured.

## Intended use

- v3 feature schema: handshape → 21×3 (x, y, z wrist-rel, scaled) for **avatar
  fingers**, **trustworthy per-finger feedback**, and the weak chin/forehead
  signs (mom/dad). **Recognition does NOT depend on this** — the deployed v2
  classifier (72.5% top-1 / 89.9% top-5, `sign_classifier_v2`) is 2D and
  already clears target; this is the precision upgrade.

## Browser inference

- ONNX export not yet wired (depth head must be added to `export_onnx.py`).

## v1 — FreiHAND + CMU (2026-05-24)

Same architecture/config as v0, but **train set doubled** by folding in CMU
HandDB (31,836 2D-only items, depth-masked) alongside FreiHAND's 30,932
3D items → **62,768 train**. Val unchanged (1,628 FreiHAND) so the numbers are
directly comparable. A100, 60 epochs, ~110 s/epoch (~1.85 hr, ~$4–5).

| metric (best) | v0 (FreiHAND) | v1 (FreiHAND+CMU) | Δ |
|---|---|---|---|
| 2D px err | 13.82 (ep59) | 13.98 (ep60) | +0.16 px (≈ noise) |
| depth err | 0.0997 | **0.0950 (ep57)** | **−4.7%** |

**Read:** CMU did **not** improve 2D px on the FreiHAND val (a wash / marginally
worse, within run-to-run noise), but it **lowered depth err ~5%**. Two caveats
make this *understate* CMU's value: (1) the val is **FreiHAND-only**, so CMU's
contribution to *in-the-wild* hand robustness isn't measured — and real webcam
hands resemble CMU's crops far more than FreiHAND's green-screen; (2) more,
more-diverse data trades a sliver of FreiHAND-val overfit for broader
generalization. **v1 is the keeper for the 3D/avatar use case** (depth is what
the head is for, and it generalizes better); v0 stays as the FreiHAND-clean
reference. Artifacts: `data/ckpts_new/hand_landmarks_3d_v1_best.pt`,
`/runs/hand_landmarks_3d_v1/`, durable `/models/hand_landmarks_3d_v1/`.

## Changelog

- **v1 (2026-05-24):** + CMU HandDB (2D, depth-masked), 62,768 train. Best
  epoch 57 — 13.98 px 2D / **0.0950 depth err**. A100, 60 epochs, ~$4–5.
- **v0 (2026-05-24):** first 3D checkpoint. Best epoch 59 — 13.82 px 2D /
  0.0997 depth err. A100, 60 epochs, ~$2.
