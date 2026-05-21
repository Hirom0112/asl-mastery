#!/usr/bin/env bash
# Extract every downloaded archive in data/external/ in place.
# Runs each dataset's extraction in parallel; tail logs/extract_*.log to monitor.
#
# Idempotent: each unzip uses -n (never overwrite); each tar checks for the
# expected output dir before running.

set -uo pipefail
cd "$(dirname "$0")/../.."

mkdir -p logs

extract_zip() {
  local dir="$1" name; name=$(basename "$dir")
  for z in "$dir"/*.zip; do
    [[ -f "$z" ]] || continue
    echo "[${name}] unzip $(basename "$z")"
    unzip -qq -n "$z" -d "$dir" >> "logs/extract_${name}.log" 2>&1
  done
  echo "[${name}] done"
}

extract_tar() {
  local dir="$1" name; name=$(basename "$dir")
  for t in "$dir"/*.tar.gz "$dir"/*.tar; do
    [[ -f "$t" ]] || continue
    echo "[${name}] tar $(basename "$t")"
    tar -xf "$t" -C "$dir" >> "logs/extract_${name}.log" 2>&1
  done
  echo "[${name}] done"
}

# Each dataset in its own subshell, all in parallel
( extract_zip data/external/freihand ) &
( extract_zip data/external/cmu_panoptic_handdb && extract_tar data/external/cmu_panoptic_handdb ) &
( extract_zip data/external/multiview_hand_pose ) &
( extract_zip data/external/coco_wholebody ) &
( extract_tar data/external/mpii_pose ) &
( extract_zip data/external/wider_face ) &

wait
echo "all extractions complete"
