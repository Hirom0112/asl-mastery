#!/usr/bin/env bash
# Push the downloaded external datasets + project-built manifests +
# the slice-1b ASL clip corpus to the Modal volume asl-mastery-data.
#
# Run this once, after:
#   1. All scripts/datasets/download_*.py --confirm runs have finished.
#   2. python -m training.detectors.normalize_external --task all  has produced
#      data/labeled_frames/<task>/external_{train,val,combined}.json.
#
# Modal volume layout written here:
#   /external/<dataset_name>/...                       — raw downloaded archives + extracted contents
#   /labeled_frames/<task>/external_*.json             — project-internal manifests
#   /datasets/asl_clips/...                            — the cleaned 12K-clip ASL corpus
#   /vocabulary/slice1b_vocabulary.json                — frozen 75-sign vocab manifest

set -euo pipefail

cd "$(dirname "$0")/../.."

# Add the user-local Python bin to PATH so `modal` is found in non-login shells.
export PATH="$HOME/Library/Python/3.9/bin:$PATH"

VOL=asl-mastery-data

if ! command -v modal >/dev/null; then
  echo "ERROR: 'modal' CLI not found on PATH." >&2
  exit 1
fi

echo "=== pushing external datasets to ${VOL}:/external/ ==="
modal volume put "${VOL}" data/external /external -f

echo "=== pushing labeled-frame manifests to ${VOL}:/labeled_frames/ ==="
if [[ -d data/labeled_frames ]]; then
  modal volume put "${VOL}" data/labeled_frames /labeled_frames -f
else
  echo "  (no data/labeled_frames/ — run \`python -m training.detectors.normalize_external --task all\` first)"
fi

echo "=== pushing slice1b ASL clip corpus to ${VOL}:/datasets/asl_clips/ ==="
if [[ -d dataset/clean ]]; then
  modal volume put "${VOL}" dataset/clean /datasets/asl_clips -f
else
  echo "  (no dataset/clean/ — skipping)"
fi

echo "=== pushing frozen vocabulary manifest to ${VOL}:/vocabulary/ ==="
modal volume put "${VOL}" dataset/slice1b_vocabulary.json /vocabulary/slice1b_vocabulary.json -f

echo
echo "done. Volume contents:"
modal volume ls "${VOL}" /
