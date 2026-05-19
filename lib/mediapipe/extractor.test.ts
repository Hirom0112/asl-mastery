import { describe, expect, it } from "vitest";

import type { HolisticLandmarkerResult, NormalizedLandmark } from "@mediapipe/tasks-vision";

import {
  COORDS_PER_LANDMARK,
  HAND_FLOATS,
  LEFT_HAND_OFFSET,
  NUM_HAND_LANDMARKS,
  POSE_INDEX_SUBSET,
  POSE_OFFSET,
  RIGHT_HAND_OFFSET,
  TOTAL_COORDS,
} from "../keypoints";
import { resultToFrame } from "./extractor";

function makeLandmark(x: number, y: number, z: number): NormalizedLandmark {
  return { x, y, z, visibility: 1 };
}

function hand(seed: number): NormalizedLandmark[] {
  return Array.from({ length: NUM_HAND_LANDMARKS }, (_, i) =>
    makeLandmark(seed + i * 0.001, seed + i * 0.001 + 0.5, i * 0.01),
  );
}

function fullPose(): NormalizedLandmark[] {
  // 33 landmarks; only the indices in POSE_INDEX_SUBSET should land in
  // the output. We give each index its own coordinate so the test
  // catches mis-indexing.
  return Array.from({ length: 33 }, (_, i) => makeLandmark(i * 0.01, i * 0.01 + 0.1, 0));
}

function emptyResult(overrides: Partial<HolisticLandmarkerResult> = {}): HolisticLandmarkerResult {
  return {
    faceLandmarks: [],
    faceBlendshapes: [],
    poseLandmarks: [],
    poseWorldLandmarks: [],
    poseSegmentationMasks: [],
    leftHandLandmarks: [],
    leftHandWorldLandmarks: [],
    rightHandLandmarks: [],
    rightHandWorldLandmarks: [],
    ...overrides,
  };
}

describe("resultToFrame", () => {
  it("returns a zeroed frame when MediaPipe detected nothing", () => {
    const frame = resultToFrame(emptyResult());
    expect(frame.keypoints).toHaveLength(TOTAL_COORDS);
    expect(frame.keypoints.every((v) => v === 0)).toBe(true);
    expect(frame.hasHand).toBe(false);
    expect(frame.hasPose).toBe(false);
  });

  it("writes left + right hands into the layout described by keypoints.ts", () => {
    const left = hand(0.1);
    const right = hand(0.7);
    const frame = resultToFrame(
      emptyResult({ leftHandLandmarks: [left], rightHandLandmarks: [right] }),
    );

    expect(frame.hasHand).toBe(true);
    // First three floats are the left hand's landmark 0 coordinates.
    expect(frame.keypoints[LEFT_HAND_OFFSET + 0]).toBeCloseTo(left[0].x);
    expect(frame.keypoints[LEFT_HAND_OFFSET + 1]).toBeCloseTo(left[0].y);
    expect(frame.keypoints[LEFT_HAND_OFFSET + 2]).toBeCloseTo(left[0].z!);
    // Right hand block immediately follows the left hand block.
    expect(frame.keypoints[RIGHT_HAND_OFFSET + 0]).toBeCloseTo(right[0].x);
    expect(frame.keypoints[RIGHT_HAND_OFFSET + HAND_FLOATS - 1]).toBeCloseTo(
      right[NUM_HAND_LANDMARKS - 1].z!,
    );
  });

  it("selects only POSE_INDEX_SUBSET from the 33-landmark MediaPipe pose array", () => {
    const pose = fullPose();
    const frame = resultToFrame(emptyResult({ poseLandmarks: [pose] }));
    expect(frame.hasPose).toBe(true);
    // Each entry in POSE_INDEX_SUBSET should write its (x, y, z) into
    // POSE_OFFSET + i * COORDS_PER_LANDMARK.
    POSE_INDEX_SUBSET.forEach((mpIdx, i) => {
      const base = POSE_OFFSET + i * COORDS_PER_LANDMARK;
      expect(frame.keypoints[base + 0]).toBeCloseTo(pose[mpIdx].x);
      expect(frame.keypoints[base + 1]).toBeCloseTo(pose[mpIdx].y);
      expect(frame.keypoints[base + 2]).toBeCloseTo(pose[mpIdx].z!);
    });
  });

  it("marks hasHand=true if only one hand was detected", () => {
    const frame = resultToFrame(emptyResult({ rightHandLandmarks: [hand(0.4)] }));
    expect(frame.hasHand).toBe(true);
    // Left-hand block stays zeroed.
    for (let i = LEFT_HAND_OFFSET; i < LEFT_HAND_OFFSET + HAND_FLOATS; i++) {
      expect(frame.keypoints[i]).toBe(0);
    }
  });
});
