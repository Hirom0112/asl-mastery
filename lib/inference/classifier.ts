// ONNX Runtime Web wrapper around the v3.x classifier.
//
// Under ADR 0010 (reversal of ADR 0006, restoring ADR 0001 Path B)
// the classifier is an end-to-end small 3D CNN trained from scratch
// on raw RGB video tensors. The MediaPipe Holistic landmark stage
// that ran between frame capture and classification under ADR 0006
// is removed. The camera-capture component emits a raw video tensor
// of shape (1, T=16, H, W, 3), float32 in [0, 1], channel-last.
//
// Loads the active model artifact URL from R2 (the URL comes from
// the active row in model_versions). Caches a single InferenceSession
// per page lifetime. The stub fallback below returns a deterministic
// prediction so the practice UI works end-to-end while no model is
// active — which is the production state until v3.0 ships.

import type { InferenceSession } from "onnxruntime-web";

// --- Input tensor shape contract ---------------------------------
// The classifier consumes a video tensor of shape (B, T, H, W, 3).
// T = 16 frames over a 2-second capture window. H and W are the
// model's expected per-frame spatial dimensions; the per-version
// config below can override these so v3 → v4 with a different
// crop size is a config change, not a code change.

export const VIDEO_TEMPORAL_LENGTH = 16 as const;
export const DEFAULT_VIDEO_HEIGHT = 96 as const;
export const DEFAULT_VIDEO_WIDTH = 96 as const;
export const VIDEO_CHANNELS = 3 as const;

export type VideoTensor = Float32Array; // length T * H * W * 3

export function videoTensorLength(h: number, w: number): number {
  return VIDEO_TEMPORAL_LENGTH * h * w * VIDEO_CHANNELS;
}

export interface ClassifierConfig {
  classes: string[];
  temperature: number;
  perSignThresholds: Record<string, number>;
  // Optional spatial dims override (per-version). Defaults to 96x96.
  inputHeight?: number;
  inputWidth?: number;
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
      logSeverityLevel: 3, // errors only — silence benign EP-assignment warnings
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
  videoTensor: VideoTensor,
  targetClassId: string,
  modelUrl: string,
  config: ClassifierConfig,
): Promise<ClassifierPrediction> {
  const h = config.inputHeight ?? DEFAULT_VIDEO_HEIGHT;
  const w = config.inputWidth ?? DEFAULT_VIDEO_WIDTH;
  const expected = videoTensorLength(h, w);
  if (videoTensor.length !== expected) {
    throw new Error(
      `video tensor length mismatch: got ${videoTensor.length}, expected ${expected} for (${VIDEO_TEMPORAL_LENGTH}, ${h}, ${w}, ${VIDEO_CHANNELS})`,
    );
  }

  const session = await getOrLoadSession(modelUrl);
  const ort = await import("onnxruntime-web");
  // Shape: (B=1, T, H, W, 3). Channel-last to match camera-capture's
  // canvas-derived layout. The exported ONNX model's first op (in T4)
  // performs the permute to whatever internal layout the 3D CNN
  // wants — keeping the channel-last contract at the wire keeps the
  // browser side simple.
  const tensor = new ort.Tensor("float32", videoTensor, [
    1,
    VIDEO_TEMPORAL_LENGTH,
    h,
    w,
    VIDEO_CHANNELS,
  ]);
  const outputs = await session.run({ video: tensor });
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

// -- Stub classifier for the period when no model is active.
//    Returns a deterministic prediction so the practice UI works
//    end-to-end. Production state right now (post-ADR-0010 T1
//    deactivation): every model_versions row has is_active=false,
//    so getActiveModelVersion() returns null and the runner routes
//    every attempt through stubPredict() until v3.0 ships.

export function stubPredict(targetClassId: string): ClassifierPrediction {
  // Deterministic per target so the same prompt always shows the
  // same "result". ~80% pass rate cosmetically, but reproducible.
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
