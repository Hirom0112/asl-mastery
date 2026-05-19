import { describe, expect, it } from "vitest";

import {
  COORDS_PER_LANDMARK,
  HAND_FLOATS,
  LEFT_HAND_OFFSET,
  NUM_HAND_LANDMARKS,
  NUM_POSE_LANDMARKS,
  POSE_FLOATS,
  POSE_INDEX_SUBSET,
  POSE_OFFSET,
  RIGHT_HAND_OFFSET,
  TEMPORAL_LENGTH,
  TOTAL_COORDS,
  TOTAL_LANDMARKS,
} from "./keypoints";

describe("keypoint subset", () => {
  it("counts 21 + 21 hand landmarks + 8 pose landmarks", () => {
    expect(NUM_HAND_LANDMARKS).toBe(21);
    expect(NUM_POSE_LANDMARKS).toBe(8);
    expect(POSE_INDEX_SUBSET).toEqual([11, 12, 13, 14, 15, 16, 23, 24]);
    expect(TOTAL_LANDMARKS).toBe(50);
  });

  it("derives TOTAL_COORDS = 150 with (x, y, z) per landmark", () => {
    expect(COORDS_PER_LANDMARK).toBe(3);
    expect(TOTAL_COORDS).toBe(150);
    expect(HAND_FLOATS).toBe(63);
    expect(POSE_FLOATS).toBe(24);
  });

  it("lays out flat frame as [left_hand | right_hand | pose] with contiguous offsets", () => {
    expect(LEFT_HAND_OFFSET).toBe(0);
    expect(RIGHT_HAND_OFFSET).toBe(HAND_FLOATS);
    expect(POSE_OFFSET).toBe(HAND_FLOATS * 2);
    expect(POSE_OFFSET + POSE_FLOATS).toBe(TOTAL_COORDS);
  });

  it("defaults temporal length to 16 frames", () => {
    expect(TEMPORAL_LENGTH).toBe(16);
  });
});
