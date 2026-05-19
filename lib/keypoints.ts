// Keypoint subset definition for the ASL classifier.
//
// Per docs/MODEL.md §1 and docs/ARCHITECTURE.md §2.3, the temporal
// classifier consumes a fixed-size keypoint subset from MediaPipe
// Holistic. The exact pose-landmark indices are tentative — Phase 4
// will measure which contribute to per-sign accuracy and may prune.
//
// This module is the single source of truth for keypoint shape. The
// extractor, the training pipeline's NumPy assembly step, and the
// ONNX-export tensor specification all reference these constants.

export const NUM_HAND_LANDMARKS = 21 as const;

// MediaPipe pose landmark indices we consume. Names mirror the
// MediaPipe pose model's index → joint mapping.
//
//   11 = left_shoulder    12 = right_shoulder
//   13 = left_elbow       14 = right_elbow
//   15 = left_wrist       16 = right_wrist
//   23 = left_hip         24 = right_hip   (torso anchors)
//
// Face indices are NOT in slice 1 (non-manual markers are slice-2,
// per ADR 0006 and docs/MODEL.md §1).
export const POSE_INDEX_SUBSET = [11, 12, 13, 14, 15, 16, 23, 24] as const;

export const NUM_POSE_LANDMARKS = POSE_INDEX_SUBSET.length;

// Coordinate dimensionality. MediaPipe returns (x, y, z) per landmark
// where x/y are normalized image coordinates in [0, 1] and z is a
// monocular-depth estimate in roughly the same scale (negative is
// closer to the camera).
export const COORDS_PER_LANDMARK = 3 as const;

// Total keypoints per frame, before being flattened.
export const TOTAL_LANDMARKS = NUM_HAND_LANDMARKS * 2 + NUM_POSE_LANDMARKS;

// Total floats per frame after flatten. The classifier's input tensor
// shape is (B, T, K) where K = TOTAL_COORDS and T defaults to 16.
export const TOTAL_COORDS = TOTAL_LANDMARKS * COORDS_PER_LANDMARK;

// Default temporal length: 2-second capture at 30 fps → 60 frames,
// subsampled to 16. See docs/ARCHITECTURE.md §2.3 step 3 and
// docs/MODEL.md §1.
export const TEMPORAL_LENGTH = 16 as const;

// Detection failure threshold. If on more than this fraction of frames
// MediaPipe returned an empty hand result (both hands missing), the
// clip is surfaced as `detection_failed` rather than pass/fail. Per
// docs/ARCHITECTURE.md §2.3 step 10 and docs/EVAL_GATE.md §1
// criterion 10.
export const DETECTION_FAILURE_THRESHOLD = 0.5 as const;

// Indexing helpers — keep flatten/unflatten consistent across the
// training pipeline (Python) and the inference path (TS). Layout:
//   [ left_hand (21 × 3) | right_hand (21 × 3) | pose_subset (8 × 3) ]
export const HAND_FLOATS = NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK;
export const POSE_FLOATS = NUM_POSE_LANDMARKS * COORDS_PER_LANDMARK;

export const LEFT_HAND_OFFSET = 0;
export const RIGHT_HAND_OFFSET = HAND_FLOATS;
export const POSE_OFFSET = HAND_FLOATS * 2;

// Type aliases for clarity at call sites.
export type FrameKeypoints = Float32Array; // length TOTAL_COORDS
export type ClipKeypoints = Float32Array; // length TEMPORAL_LENGTH * TOTAL_COORDS
