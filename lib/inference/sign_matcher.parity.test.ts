import fs from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { StreamingMatcher, type SignTemplate, type ThresholdEntry } from "./sign_matcher";

// Parity fixture exported from training/detectors/sign_matcher.py.
// See lib/inference/__fixtures__/again_clip.json — regenerated via the
// scratch script in this repo's session-notes when templates_v7 lands.
interface FixtureTemplate {
  T: number;
  F: number;
  mean: number[];
  var: number[];
  n_clips: number;
  dominant_hand_only: boolean;
  n_clusters: number;
}

interface FixtureTrace {
  frame_i: number;
  scores: Record<string, number>;
  target_streak: number;
  unlocked: boolean;
  top1: string | null;
}

interface Fixture {
  frames: unknown[];
  templates: Record<string, FixtureTemplate>;
  thresholds: Record<string, ThresholdEntry & { threshold: number }>;
  target: string;
  slice_ids: string[];
  time_steps: number;
  window_frames: number;
  consecutive_required: number;
  trace: FixtureTrace[];
}

function loadFixture(): Fixture {
  const p = path.join(__dirname, "__fixtures__", "again_clip.json");
  return JSON.parse(fs.readFileSync(p, "utf8")) as Fixture;
}

function buildTemplate(sid: string, t: FixtureTemplate): SignTemplate {
  return {
    signId: sid,
    T: t.T,
    F: t.F,
    mean: new Float32Array(t.mean),
    var: new Float32Array(t.var),
    nClips: t.n_clips,
    dominantHandOnly: t.dominant_hand_only,
  };
}

describe("StreamingMatcher parity with Python", () => {
  it("reproduces Python's per-frame scores and unlock timing", () => {
    const fx = loadFixture();
    const templates: Record<string, SignTemplate> = {};
    for (const [sid, t] of Object.entries(fx.templates)) {
      templates[sid] = buildTemplate(sid, t);
    }

    const m = new StreamingMatcher(templates, {
      timeSteps: fx.time_steps,
      windowFrames: fx.window_frames,
      consecutiveRequired: fx.consecutive_required,
      thresholds: fx.thresholds,
    });

    const tsTrace: FixtureTrace[] = [];
    for (let i = 0; i < fx.frames.length; i++) {
      // Cast: the fixture's `frames` are raw frame dicts as written by
      // extract_trajectories. pushFrame's RawFrame interface accepts them.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      m.pushFrame(fx.frames[i] as any);
      const s = m.step(fx.target, fx.slice_ids);
      if (!s.ready) continue;
      tsTrace.push({
        frame_i: i,
        scores: s.scores,
        target_streak: s.targetStreak,
        unlocked: s.unlocked,
        top1: s.top1,
      });
      if (s.unlocked) break;
    }

    expect(tsTrace.length).toBe(fx.trace.length);
    for (let k = 0; k < tsTrace.length; k++) {
      const exp = fx.trace[k];
      const got = tsTrace[k];
      expect(got.frame_i).toBe(exp.frame_i);
      expect(got.top1).toBe(exp.top1);
      expect(got.target_streak).toBe(exp.target_streak);
      expect(got.unlocked).toBe(exp.unlocked);
      // Score-by-score: numerics should match within float32 precision.
      // Python uses float64 in DTW accumulation; the TS port matches.
      // Tolerance: 1e-4 (relative scores are ~0.07-0.5).
      for (const sid of fx.slice_ids) {
        expect(got.scores[sid]).toBeCloseTo(exp.scores[sid], 3);
      }
    }
  });
});
