"""Keypoint shape constants — Python mirror of lib/keypoints.ts.

Both files must stay in lockstep. The cleaning pipeline writes
keypoint tensors with the layout defined here; the browser
classifier reads tensors with the layout defined in lib/keypoints.ts.
A divergence between the two means the model sees garbage at
inference time.
"""

from __future__ import annotations

NUM_HAND_LANDMARKS = 21

# MediaPipe pose landmark indices we consume. See docs/MODEL.md §1.
POSE_INDEX_SUBSET: tuple[int, ...] = (11, 12, 13, 14, 15, 16, 23, 24)

NUM_POSE_LANDMARKS = len(POSE_INDEX_SUBSET)

# (x, y, z) per landmark.
COORDS_PER_LANDMARK = 3

TOTAL_LANDMARKS = NUM_HAND_LANDMARKS * 2 + NUM_POSE_LANDMARKS
TOTAL_COORDS = TOTAL_LANDMARKS * COORDS_PER_LANDMARK

# 2-second capture at 30 fps → 60 frames, subsampled to 16.
TEMPORAL_LENGTH = 16

# Per docs/ARCHITECTURE.md §2.3 step 10 and docs/EVAL_GATE.md §1 #10.
DETECTION_FAILURE_THRESHOLD = 0.5

# Flat layout offsets per frame:
# [ left_hand (21 × 3) | right_hand (21 × 3) | pose_subset (8 × 3) ]
HAND_FLOATS = NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK
POSE_FLOATS = NUM_POSE_LANDMARKS * COORDS_PER_LANDMARK
LEFT_HAND_OFFSET = 0
RIGHT_HAND_OFFSET = HAND_FLOATS
POSE_OFFSET = HAND_FLOATS * 2

assert POSE_OFFSET + POSE_FLOATS == TOTAL_COORDS
