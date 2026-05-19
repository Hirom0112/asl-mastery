// Convert a MediaPipe HolisticLandmarkerResult to the flat keypoint
// tensor shape the classifier consumes, per docs/MODEL.md §1.

import type {
  HolisticLandmarker,
  HolisticLandmarkerResult,
  NormalizedLandmark,
} from "@mediapipe/tasks-vision";

import {
  COORDS_PER_LANDMARK,
  DETECTION_FAILURE_THRESHOLD,
  LEFT_HAND_OFFSET,
  NUM_HAND_LANDMARKS,
  POSE_INDEX_SUBSET,
  POSE_OFFSET,
  RIGHT_HAND_OFFSET,
  TOTAL_COORDS,
  type ClipKeypoints,
  type FrameKeypoints,
} from "../keypoints";

export interface FrameResult {
  // Float32Array of length TOTAL_COORDS. Missing landmarks are zero.
  keypoints: FrameKeypoints;
  // True if MediaPipe returned at least one hand on this frame.
  hasHand: boolean;
  // True if pose landmarks were returned (used as a secondary signal
  // — a clip can be classified from hands alone if the framing is
  // tight, but hands are the primary signal).
  hasPose: boolean;
}

export interface ClipResult {
  // (T, K) tensor, flattened row-major: frame 0 floats, then frame 1, ...
  keypoints: ClipKeypoints;
  // Frames count actually produced. Caller is responsible for ensuring
  // T matches the model's expected temporal length before inference.
  frames: number;
  // Fraction of frames where MediaPipe did not detect a hand.
  handDetectionFailureRate: number;
  // True if handDetectionFailureRate exceeds DETECTION_FAILURE_THRESHOLD.
  // Surface as `detection_failed` per ARCHITECTURE §2.3 step 10.
  detectionFailed: boolean;
}

// Write a (NUM_HAND_LANDMARKS × 3) block of `out` starting at `offset`
// from a MediaPipe NormalizedLandmark array. Missing/empty input
// leaves the block zeroed.
function writeHand(
  out: Float32Array,
  offset: number,
  landmarks: NormalizedLandmark[] | undefined,
): boolean {
  if (!landmarks || landmarks.length < NUM_HAND_LANDMARKS) return false;
  for (let i = 0; i < NUM_HAND_LANDMARKS; i++) {
    const lm = landmarks[i];
    const base = offset + i * COORDS_PER_LANDMARK;
    out[base] = lm.x;
    out[base + 1] = lm.y;
    out[base + 2] = lm.z ?? 0;
  }
  return true;
}

function writePose(
  out: Float32Array,
  offset: number,
  landmarks: NormalizedLandmark[] | undefined,
): boolean {
  if (!landmarks || landmarks.length === 0) return false;
  for (let i = 0; i < POSE_INDEX_SUBSET.length; i++) {
    const idx = POSE_INDEX_SUBSET[i];
    const lm = landmarks[idx];
    const base = offset + i * COORDS_PER_LANDMARK;
    if (lm) {
      out[base] = lm.x;
      out[base + 1] = lm.y;
      out[base + 2] = lm.z ?? 0;
    }
    // Missing pose index leaves zeros; consistent with the train-time
    // assumption that missing landmarks are zeroed (matches keypoint
    // dropout augmentation behavior, docs/MODEL.md §3).
  }
  return true;
}

export function resultToFrame(result: HolisticLandmarkerResult): FrameResult {
  const out = new Float32Array(TOTAL_COORDS);
  const hasLeft = writeHand(out, LEFT_HAND_OFFSET, result.leftHandLandmarks?.[0]);
  const hasRight = writeHand(out, RIGHT_HAND_OFFSET, result.rightHandLandmarks?.[0]);
  const hasPose = writePose(out, POSE_OFFSET, result.poseLandmarks?.[0]);
  return { keypoints: out, hasHand: hasLeft || hasRight, hasPose };
}

// Extract keypoint sequences from a sequence of video frames. Frames
// come pre-cropped to the green box (per ARCHITECTURE §2.3 step 4).
// Caller controls timestamps to match the MediaPipe Tasks VIDEO
// running-mode contract (monotonically increasing milliseconds).
export function extractClip(
  landmarker: HolisticLandmarker,
  frames: { image: HTMLVideoElement | HTMLCanvasElement | ImageBitmap; timestampMs: number }[],
): ClipResult {
  const T = frames.length;
  const out = new Float32Array(T * TOTAL_COORDS);
  let handFailureCount = 0;

  for (let t = 0; t < T; t++) {
    const { image, timestampMs } = frames[t];
    const r = landmarker.detectForVideo(image, timestampMs);
    const frame = resultToFrame(r);
    out.set(frame.keypoints, t * TOTAL_COORDS);
    if (!frame.hasHand) handFailureCount++;
  }

  const handDetectionFailureRate = T === 0 ? 1 : handFailureCount / T;
  return {
    keypoints: out,
    frames: T,
    handDetectionFailureRate,
    detectionFailed: handDetectionFailureRate > DETECTION_FAILURE_THRESHOLD,
  };
}
