// ONNX Runtime Web wrapper around the slice-1 classifier.
//
// Loads the active model artifact from R2 (the URL comes from the
// active row in model_versions). Caches a single InferenceSession
// per page lifetime. The stub fallback below returns a deterministic
// prediction so Phase 5b's UI works end-to-end before the real
// model lands.

import type { InferenceSession } from "onnxruntime-web";

import { TEMPORAL_LENGTH, TOTAL_COORDS, type ClipKeypoints } from "@/lib/keypoints";

export interface ClassifierConfig {
  classes: string[];
  temperature: number;
  perSignThresholds: Record<string, number>;
}

export interface ClassifierPrediction {
  predictedClassId: string;
  confidence: number;
  topK: { classId: string; probability: number }[];
  passed: boolean;
  threshold: number;
}

let sessionPromise: Promise<InferenceSession> | null = null;

async function getOrLoadSession(modelUrl: string): Promise<InferenceSession> {
  if (sessionPromise) return sessionPromise;
  sessionPromise = (async () => {
    const ort = await import("onnxruntime-web");
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["webgpu", "wasm"],
    });
    return session;
  })();
  try {
    return await sessionPromise;
  } catch (err) {
    sessionPromise = null;
    throw err;
  }
}

function softmaxWithTemperature(logits: number[], temperature: number): number[] {
  const T = temperature > 0 ? temperature : 1;
  const scaled = logits.map((z) => z / T);
  const max = Math.max(...scaled);
  const exps = scaled.map((z) => Math.exp(z - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

export async function predict(
  keypoints: ClipKeypoints,
  targetClassId: string,
  modelUrl: string,
  config: ClassifierConfig,
): Promise<ClassifierPrediction> {
  if (keypoints.length !== TEMPORAL_LENGTH * TOTAL_COORDS) {
    throw new Error(
      `keypoint tensor length mismatch: got ${keypoints.length}, expected ${TEMPORAL_LENGTH * TOTAL_COORDS}`,
    );
  }

  const session = await getOrLoadSession(modelUrl);
  const ort = await import("onnxruntime-web");
  const tensor = new ort.Tensor("float32", keypoints, [1, TEMPORAL_LENGTH, TOTAL_COORDS]);
  const outputs = await session.run({ keypoints: tensor });
  const logits = Array.from(outputs.logits.data as Float32Array);

  const probs = softmaxWithTemperature(logits, config.temperature);
  const ranked = probs
    .map((p, i) => ({ classId: config.classes[i], probability: p }))
    .sort((a, b) => b.probability - a.probability);

  const top = ranked[0];
  const threshold = config.perSignThresholds[targetClassId] ?? 0.5;
  const targetProb = ranked.find((r) => r.classId === targetClassId)?.probability ?? 0;
  const passed = top.classId === targetClassId && targetProb >= threshold;

  return {
    predictedClassId: top.classId,
    confidence: top.probability,
    topK: ranked.slice(0, 3),
    passed,
    threshold,
  };
}

// -- Stub classifier for the period before the real model artifact
//    lands. Returns a deterministic prediction so the Phase 5b UI
//    can be exercised end-to-end. Swap-in: the practice screen calls
//    predictWithFallback; once getActiveModelVersion() returns a real
//    artifact, it routes through predict() above.

export function stubPredict(targetClassId: string): ClassifierPrediction {
  // Deterministic per target so the same prompt always shows the
  // same "result". 80% pass rate cosmetically, but reproducible.
  let h = 0;
  for (const ch of targetClassId) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  const passed = h % 5 !== 0; // ~80%
  return {
    predictedClassId: targetClassId,
    confidence: passed ? 0.92 : 0.31,
    topK: [
      { classId: targetClassId, probability: passed ? 0.92 : 0.31 },
      { classId: "_other", probability: passed ? 0.05 : 0.42 },
      { classId: "_other2", probability: passed ? 0.03 : 0.27 },
    ],
    passed,
    threshold: 0.5,
  };
}
