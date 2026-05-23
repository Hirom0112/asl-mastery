import { describe, expect, it } from "vitest";

import {
  FEATURES_PER_FRAME,
  HAND_SLOT_DIMS,
  StreamingMatcher,
  dtwScore,
  frameToFeatures,
  mahalanobisScore,
  mirrorFeatures,
  resampleTrajectory,
  type SignTemplate,
} from "./sign_matcher";

// A pose-only frame with neck at (100,100) and shoulders at ±50 px.
// Gives anchor=(100,100), scale=100, so normalized coords are (dx/100, dy/100).
function poseOnlyFrame() {
  return {
    pose: [
      [100, 80], // nose
      [100, 100], // neck
      [50, 100], // r_shoulder
      [150, 100], // l_shoulder
      [50, 130],
      [150, 130],
      [50, 160],
      [150, 160],
    ] as const,
  };
}

describe("frameToFeatures", () => {
  it("returns a 100-d Float32Array with neck normalized to origin", () => {
    const { feats, usedSlot1 } = frameToFeatures(poseOnlyFrame());
    expect(feats.length).toBe(FEATURES_PER_FRAME);
    expect(usedSlot1).toBe(false);
    // neck pose index 1 → POSE_BASE + 1*2 = 86,87
    expect(feats[86]).toBeCloseTo(0, 5);
    expect(feats[87]).toBeCloseTo(0, 5);
    // nose at (100,80): dx=0, dy=-20, scale=100 → (0, -0.2)
    expect(feats[84]).toBeCloseTo(0, 5);
    expect(feats[85]).toBeCloseTo(-0.2, 5);
    // hand slots untouched: NaN
    expect(Number.isNaN(feats[0])).toBe(true);
  });

  it("assigns hand slot 0 for wrist left of anchor; slot 1 for right", () => {
    const frame = {
      ...poseOnlyFrame(),
      hands: [
        { keypoints: Array.from({ length: 21 }, (_, i) => [80, 100 + i] as const) },
        { keypoints: Array.from({ length: 21 }, (_, i) => [120, 100 + i] as const) },
      ],
    };
    const { feats, usedSlot1 } = frameToFeatures(frame);
    expect(usedSlot1).toBe(true);
    // slot 0 wrist (index 0 = (80,100)) → ((80-100)/100, 0) = (-0.2, 0)
    expect(feats[0]).toBeCloseTo(-0.2, 5);
    expect(feats[1]).toBeCloseTo(0, 5);
    // slot 1 wrist at HAND_SLOT_DIMS offset → ((120-100)/100, 0) = (0.2, 0)
    expect(feats[HAND_SLOT_DIMS]).toBeCloseTo(0.2, 5);
    expect(feats[HAND_SLOT_DIMS + 1]).toBeCloseTo(0, 5);
  });
});

describe("mirrorFeatures", () => {
  it("is an involution (mirror twice = identity)", () => {
    const T = 4;
    const F = FEATURES_PER_FRAME;
    const traj = new Float32Array(T * F);
    for (let i = 0; i < traj.length; i++) traj[i] = Math.sin(i * 0.13);
    const m1 = mirrorFeatures(traj, T, F);
    const m2 = mirrorFeatures(m1, T, F);
    for (let i = 0; i < traj.length; i++) expect(m2[i]).toBeCloseTo(traj[i], 5);
  });

  it("negates x-components and swaps hand slots", () => {
    const F = FEATURES_PER_FRAME;
    const traj = new Float32Array(F);
    // slot 0 wrist x at index 0
    traj[0] = 0.3;
    traj[1] = 0.5;
    traj[HAND_SLOT_DIMS] = -0.4;
    traj[HAND_SLOT_DIMS + 1] = 0.6;
    const m = mirrorFeatures(traj, 1, F);
    // slot 0 should now hold the negated-x version of slot 1
    expect(m[0]).toBeCloseTo(0.4, 5);
    expect(m[1]).toBeCloseTo(0.6, 5);
    expect(m[HAND_SLOT_DIMS]).toBeCloseTo(-0.3, 5);
    expect(m[HAND_SLOT_DIMS + 1]).toBeCloseTo(0.5, 5);
  });
});

describe("resampleTrajectory", () => {
  it("identity when n === T", () => {
    const T = 4;
    const F = 2;
    const traj = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7]);
    const out = resampleTrajectory(traj, T, T, F);
    for (let i = 0; i < traj.length; i++) expect(out[i]).toBe(traj[i]);
  });

  it("linearly interpolates and snaps endpoints", () => {
    // n=2, T=5, single feature ramping 0 → 10
    const traj = new Float32Array([0, 10]);
    const out = resampleTrajectory(traj, 2, 5, 1);
    expect(out[0]).toBeCloseTo(0, 5);
    expect(out[1]).toBeCloseTo(2.5, 5);
    expect(out[2]).toBeCloseTo(5, 5);
    expect(out[3]).toBeCloseTo(7.5, 5);
    expect(out[4]).toBeCloseTo(10, 5);
  });
});

function makeFlatTemplate(T: number, F: number, fillMean = 0, fillVar = 1): SignTemplate {
  const mean = new Float32Array(T * F);
  const variance = new Float32Array(T * F);
  mean.fill(fillMean);
  variance.fill(fillVar);
  return {
    signId: "stub",
    T,
    F,
    mean,
    var: variance,
    nClips: 50, // above nearest-clip threshold → use DTW
    dominantHandOnly: false,
  };
}

describe("dtwScore + mahalanobisScore", () => {
  it("zero distance when trajectory == mean", () => {
    const T = 8;
    const F = FEATURES_PER_FRAME;
    const tpl = makeFlatTemplate(T, F, 0.5, 1.0);
    const traj = new Float32Array(T * F);
    traj.fill(0.5);
    const s = dtwScore(traj, tpl.mean, tpl.var, T, F, false);
    expect(s).toBeCloseTo(0, 5);
  });

  it("symmetric mirror keeps score finite", () => {
    const T = 8;
    const F = FEATURES_PER_FRAME;
    const tpl = makeFlatTemplate(T, F);
    const traj = new Float32Array(T * F);
    traj.fill(0.1);
    const s = mahalanobisScore(traj, tpl);
    expect(Number.isFinite(s)).toBe(true);
    expect(s).toBeGreaterThanOrEqual(0);
  });
});

describe("StreamingMatcher", () => {
  it("not ready until window fills", () => {
    const T = 4;
    const F = FEATURES_PER_FRAME;
    const tpl = makeFlatTemplate(T, F);
    const m = new StreamingMatcher(
      { stub: tpl },
      { timeSteps: T, windowFrames: T, consecutiveRequired: 2 },
    );
    expect(m.ready()).toBe(false);
    for (let i = 0; i < T - 1; i++) m.pushFeatures(new Float32Array(F));
    expect(m.ready()).toBe(false);
    m.pushFeatures(new Float32Array(F));
    expect(m.ready()).toBe(true);
  });

  it("emits unlocked after consecutive_required passing windows", () => {
    const T = 4;
    const F = FEATURES_PER_FRAME;
    const tpl = makeFlatTemplate(T, F, 0, 1);
    const m = new StreamingMatcher(
      { stub: tpl },
      {
        timeSteps: T,
        windowFrames: T,
        consecutiveRequired: 2,
        thresholds: { stub: { threshold: 1.0 } },
      },
    );
    // push zeros — score against mean=0 should be 0, well under threshold 1.0
    for (let i = 0; i < T; i++) m.pushFeatures(new Float32Array(F));
    const s1 = m.step("stub");
    expect(s1.ready).toBe(true);
    expect(s1.targetPassedNow).toBe(true);
    expect(s1.unlocked).toBe(false); // first pass only
    m.pushFeatures(new Float32Array(F));
    const s2 = m.step("stub");
    expect(s2.unlocked).toBe(true);
  });

  it("resets streak on a miss", () => {
    const T = 4;
    const F = FEATURES_PER_FRAME;
    const tpl = makeFlatTemplate(T, F, 0, 1);
    const m = new StreamingMatcher(
      { stub: tpl },
      {
        timeSteps: T,
        windowFrames: T + 2,
        consecutiveRequired: 3,
        thresholds: { stub: { threshold: 0.01 } },
      },
    );
    // passing window (zeros)
    for (let i = 0; i < T; i++) m.pushFeatures(new Float32Array(F));
    expect(m.step("stub").targetStreak).toBe(1);
    // miss: push a high-variance frame
    const bad = new Float32Array(F);
    bad.fill(5);
    m.pushFeatures(bad);
    expect(m.step("stub").targetStreak).toBe(0);
  });
});
