#!/usr/bin/env bash
# Fire the overnight training pipeline on Modal.
#
# Assumes scripts/datasets/push_to_modal.sh has already run.
#
# Sequence (each waits on the previous via Modal job-completion):
#   1. train_hand_detector       (Phase 1 detector)
#   2. train_hand_landmarks      (Phase 2 regressor)
#   3. train_face                (Phase 3.2; face_bbox manifest)
#   # train_pose deferred — small refactor pending, see modal_app.py
#
# Each step writes /runs/<run_id>/best.pt to the volume. You can close
# the laptop after this script returns "submitted" — the jobs continue
# on Modal regardless of laptop state or internet.
#
# Usage:
#   bash scripts/launch_overnight_training.sh           # uses defaults
#   bash scripts/launch_overnight_training.sh --epochs 30
#
# Modal CLI must be on PATH and `modal token current` must succeed.

set -uo pipefail

cd "$(dirname "$0")/.."

export PATH="$HOME/Library/Python/3.9/bin:$PATH"

EPOCHS=${1:-60}
STAMP=$(date -u +%Y%m%d_%H%M%SZ)

echo "=== launching hand_detector (${EPOCHS} epochs) ==="
modal run --detach training/modal_app.py::train_hand_detector \
    --train-manifest /labeled_frames/hand_bbox/external_train.json \
    --val-manifest /labeled_frames/hand_bbox/external_val.json \
    --run-id "hand_det_v0_${STAMP}" \
    --epochs "${EPOCHS}"

echo "=== launching hand_landmarks (${EPOCHS} epochs) ==="
modal run --detach training/modal_app.py::train_hand_landmarks \
    --train-manifest /labeled_frames/hand_keypoints/external_train.json \
    --val-manifest /labeled_frames/hand_keypoints/external_val.json \
    --run-id "hand_landmarks_v0_${STAMP}" \
    --epochs "${EPOCHS}"

echo "=== launching face_detector (${EPOCHS} epochs) ==="
modal run --detach training/modal_app.py::train_face \
    --train-manifest /labeled_frames/face_bbox/external_train.json \
    --val-manifest /labeled_frames/face_bbox/external_val.json \
    --run-id "face_det_v0_${STAMP}" \
    --epochs "${EPOCHS}"

echo
echo "all jobs submitted in --detach mode. They will continue running on Modal"
echo "even if you close this laptop. Check status with:"
echo
echo "    modal app list"
echo "    modal app logs asl-mastery   # streams logs of the most recent run"
echo
echo "Pull artifacts when done:"
echo "    modal volume get asl-mastery-data /runs/hand_det_v0_${STAMP} ./runs/"
echo "    modal volume get asl-mastery-data /runs/hand_landmarks_v0_${STAMP} ./runs/"
echo "    modal volume get asl-mastery-data /runs/face_det_v0_${STAMP} ./runs/"
