"use client";

// End-to-end in-browser recognition for the from-scratch KEYPOINT pipeline.
// Replaces the dead 3D-CNN-on-raw-video path (classifier.ts) for the deployed
// v3 keypoint classifier (75.8% top1).
//
//   recorded frames → ONNX detector/landmark/pose (keypoints.ts)
//   → 108D hand-relative features (sign_matcher.frameToFeaturesV2)
//   → drop handless frames → NaN-aware resample to 32
//   → sign_classifier_v3.onnx → softmax → top-k.
//
// Mirrors training/detectors/sign_matcher.trajectory_from_frames + the
// SignClassifier (which zeros NaN internally; we also zero defensively).
// Artifacts are served from /public/models (no DB model_versions row needed).

import type { InferenceSession } from "onnxruntime-web";

import type { ClassifierPrediction } from "./classifier";
import { extractFrame, loadKeypointModels, type KeypointModels } from "./keypoints";
import { FEATURES_PER_FRAME_V2, frameToFeaturesV2, resampleTrajectory } from "./sign_matcher";

const TIME_STEPS = 32;
const MODELS_BASE = "/models";
const DEFAULT_THRESHOLD = 0.5;

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
      return ort.InferenceSession.create(`${MODELS_BASE}/sign_classifier_v3.onnx`, {
        executionProviders: ["webgpu", "wasm"],
      });
    })();
  }
  return clfPromise;
}

async function getConfig(): Promise<KeypointClassifierConfig> {
  if (!cfgPromise) {
    cfgPromise = fetch(`${MODELS_BASE}/sign_classifier_v3.config.json`).then((r) => {
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
  const rows: Float32Array[] = [];
  for (const fr of frames) {
    const raw = await extractFrame(models, fr, w, h);
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
  const topK = ranked.slice(0, 3);
  const top = topK[0];
  // Pedagogical pass: the user signed the PROMPTED sign as top-1 with enough
  // confidence. Low confidence / wrong top-1 → "try again" hint upstream.
  const passed = top.classId === targetClassId && top.probability >= threshold;
  return { predictedClassId: top.classId, confidence: top.probability, topK, passed, threshold };
}
