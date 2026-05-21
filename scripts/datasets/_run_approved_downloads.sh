#!/usr/bin/env bash
# Sequential runner for the user-approved auto-downloadable subset.
# Spawned by Claude during Session 14 after the user approved the
# data plan and elected to use the bigger seed dataset (skipping
# InterHand2.6M and WIDER FACE images, which require manual steps).
#
# Logs land at logs/datasets_download_<dataset>.log so progress is
# tail-able from another shell.

set -uo pipefail

cd "$(dirname "$0")/../.."

mkdir -p logs

run() {
  local name="$1" ; shift
  local log="logs/datasets_download_${name}.log"
  echo "=== $(date -u +%FT%TZ) starting ${name} -> ${log}"
  if python3 "scripts/datasets/download_${name}.py" --confirm >>"${log}" 2>&1; then
    echo "=== $(date -u +%FT%TZ) ${name} OK"
  else
    rc=$?
    echo "=== $(date -u +%FT%TZ) ${name} FAILED rc=${rc} (see ${log})"
  fi
}

# Smallest first so an early failure is visible fast.
run wider_face
run multiview_hand_pose
run freihand
run mpii_pose
run coco_wholebody
run cmu_panoptic_handdb

echo "=== $(date -u +%FT%TZ) all jobs done"
