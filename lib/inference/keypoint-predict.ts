"use client";

// End-to-end in-browser recognition for the from-scratch KEYPOINT pipeline.
// Replaces the dead 3D-CNN-on-raw-video path (classifier.ts) for the deployed
// v4 FACE-ANCHORED keypoint classifier (81.6% top1, signer-disjoint).
//
//   recorded frames → ONNX hand/face detect + landmark + FACE-ANCHORED pose
//     (keypoints.extractFramesFaceAnchored: face-smooth + face-guard + PoseEMA)
//   → 108D hand-relative features (sign_matcher.frameToFeaturesV2)
//   → drop handless frames → NaN-aware resample to 32
//   → sign_classifier_v4.onnx → softmax → top-k.
//
// The face-anchored feature pipeline is what the v4 classifier was trained on
// (it beat the hand-anchored 75.8 by +5.8 top-1). Mirrors
// extract_trajectories_v2.py --pose-anchor face. Artifacts served from
// /public/models (no DB model_versions row needed).

import type { InferenceSession } from "onnxruntime-web";

import type { ClassifierPrediction } from "./classifier";
import { extractFramesFaceAnchored, loadKeypointModels, type KeypointModels } from "./keypoints";
import { FEATURES_PER_FRAME_V2, frameToFeaturesV2, resampleTrajectory } from "./sign_matcher";

const TIME_STEPS = 32;
const MODELS_BASE = "/models";
// 0.3 = pass when the prompted sign reaches 30% confidence. An 80-class softmax
// puts a CORRECT answer at only ~0.3-0.5, so 0.5 was too strict (correct
// attempts failed / flickered). 0.3 is forgiving but still meaningful.
const DEFAULT_THRESHOLD = 0.3;

export interface KeypointClassifierConfig {
  classes: string[];
  temperature?: number;
  perSignThresholds?: Record<string, number>;
}

let modelsPromise: Promise<KeypointModels> | null = null;
let clfPromise: Promise<InferenceSession> | null = null;
let cfgPromise: Promise<KeypointClassifierConfig> | null = null;

async function getModels(): Promise<KeypointModels> {
  if (!modelsPromise) modelsPromise = loadKeypointModels(MODELS_BASE);
  return modelsPromise;
}

async function getClassifier(): Promise<InferenceSession> {
  if (!clfPromise) {
    clfPromise = (async () => {
      const ort = await import("onnxruntime-web");
      return ort.InferenceSession.create(`${MODELS_BASE}/sign_classifier_v4.onnx`, {
        executionProviders: ["webgpu", "wasm"],
        logSeverityLevel: 3, // errors only — silence benign EP-assignment warnings
      });
    })();
  }
  return clfPromise;
}

async function getConfig(): Promise<KeypointClassifierConfig> {
  if (!cfgPromise) {
    cfgPromise = fetch(`${MODELS_BASE}/sign_classifier_v4.config.json`).then((r) => {
      if (!r.ok) throw new Error(`config fetch ${r.status}`);
      return r.json();
    });
  }
  return cfgPromise;
}

// Warm up models on mount so the first attempt isn't slow (optional caller use).
export async function preloadKeypointModels(): Promise<void> {
  await Promise.all([getModels(), getClassifier(), getConfig()]);
}

function softmax(logits: Float32Array, temperature = 1): number[] {
  const t = temperature || 1;
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] / t > max) max = logits[i] / t;
  const exps = Array.from(logits, (v) => Math.exp(v / t - max));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / sum);
}

// frames → (TIME_STEPS, 108) trajectory, or null if no hands were ever seen.
async function buildTrajectory(
  models: KeypointModels,
  frames: ImageBitmap[],
  w: number,
  h: number,
): Promise<Float32Array | null> {
  const F = FEATURES_PER_FRAME_V2;
  // Face-anchored extraction needs the whole clip (temporal face-smooth +
  // PoseEMA), so process all frames together, then build per-frame features.
  const rawFrames = await extractFramesFaceAnchored(models, frames, w, h);
  const rows: Float32Array[] = [];
  for (const raw of rawFrames) {
    if (!raw.hands || raw.hands.length === 0) continue; // drop_handless_frames
    rows.push(frameToFeaturesV2(raw).feats);
  }
  if (rows.length === 0) return null;
  const flat = new Float32Array(rows.length * F);
  rows.forEach((r, i) => flat.set(r, i * F));
  return resampleTrajectory(flat, rows.length, TIME_STEPS, F);
}

export async function predictFromFrames(
  frames: ImageBitmap[],
  frameWidth: number,
  frameHeight: number,
  targetClassId: string,
): Promise<ClassifierPrediction> {
  const [models, clf, cfg] = await Promise.all([getModels(), getClassifier(), getConfig()]);
  const threshold = cfg.perSignThresholds?.[targetClassId] ?? DEFAULT_THRESHOLD;

  const traj = await buildTrajectory(models, frames, frameWidth, frameHeight);
  if (!traj) {
    // No hands detected across the whole clip → graceful "try again".
    return { predictedClassId: "", confidence: 0, topK: [], passed: false, threshold };
  }
  for (let i = 0; i < traj.length; i++) if (Number.isNaN(traj[i])) traj[i] = 0;

  const ort = await import("onnxruntime-web");
  const input = new ort.Tensor("float32", traj, [1, TIME_STEPS, FEATURES_PER_FRAME_V2]);
  const out = await clf.run({ [clf.inputNames[0]]: input });
  const logits = out[clf.outputNames[0]].data as Float32Array;
  const probs = softmax(logits, cfg.temperature ?? 1);

  const ranked = Array.from(probs, (p, i) => ({ classId: cfg.classes[i], probability: p })).sort(
    (a, b) => b.probability - a.probability,
  );
  const topK = ranked.slice(0, 4);
  const top = topK[0];
  const topProb = ranked[0]?.probability ?? 0;
  const targetRank = ranked.findIndex((r) => r.classId === targetClassId);
  const targetProb = targetRank >= 0 ? ranked[targetRank].probability : 0;

  // "Match strength" = how dominant the prompted sign is among the model's
  // guesses (targetProb / topProb). It reads meaningfully high when the learner
  // signs correctly (100% when the sign IS the model's top pick) instead of the
  // raw 80-class softmax prob (always ~0.3-0.5, which felt "always low").
  const matchStrength = topProb > 0 ? targetProb / topProb : 0;

  // Pass only when the model's TOP guess is the prompted sign, or a *close*
  // second (≥50% of the top prob). The old top-4 rule passed clearly-wrong
  // attempts (a distant 2nd-4th place "passed"); this requires a real match.
  const passed = targetRank === 0 || (targetRank === 1 && matchStrength >= 0.5);

  return { predictedClassId: top.classId, confidence: matchStrength, topK, passed, threshold };
}
