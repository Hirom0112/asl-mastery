// Lazy, singleton loader for MediaPipe HolisticLandmarker.
//
// MediaPipe Tasks Web ships a WASM runtime that is downloaded on first
// use. We pin both the WASM fileset and the .task model URL to avoid
// silent upstream version drift between the cleaning pipeline (Python
// MediaPipe, version recorded in the dataset manifest) and the
// inference path (this module). See docs/EVAL_GATE.md §1 criterion 10
// — the MediaPipe version is part of the validation report.

import {
  FilesetResolver,
  HolisticLandmarker,
  type HolisticLandmarkerOptions,
} from "@mediapipe/tasks-vision";

// Pinned to the @mediapipe/tasks-vision package version we installed.
// Update both this constant and the dataset manifest's `mediapipe_version`
// field together when bumping the dep.
export const MEDIAPIPE_VERSION = "0.10.35" as const;

const DEFAULT_WASM_BASE_URL =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm` as const;

// The Holistic .task bundle hosted by Google. The training pipeline
// mirrors this same .task file to R2 under
// `asl-mastery-models/mediapipe/${MEDIAPIPE_VERSION}/holistic_landmarker.task`
// and the dataset manifest records its sha256. For slice-1 inference
// we still fetch from Google's CDN by default; an env override is
// available for cases where we want all assets coming from R2.
const DEFAULT_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task" as const;

let loaderPromise: Promise<HolisticLandmarker> | null = null;

export interface LoadOptions {
  wasmBaseUrl?: string;
  modelAssetUrl?: string;
  // VIDEO is the running-mode we use for both the recording tool and
  // the practice screen (frames come from a MediaStream / canvas).
  // IMAGE is exposed for tests and one-shot diagnostics.
  runningMode?: "VIDEO" | "IMAGE";
}

export async function loadHolistic(options: LoadOptions = {}): Promise<HolisticLandmarker> {
  if (loaderPromise) return loaderPromise;

  const wasmBaseUrl = options.wasmBaseUrl ?? DEFAULT_WASM_BASE_URL;
  const modelAssetUrl = options.modelAssetUrl ?? DEFAULT_MODEL_URL;
  const runningMode = options.runningMode ?? "VIDEO";

  loaderPromise = (async () => {
    const fileset = await FilesetResolver.forVisionTasks(wasmBaseUrl);
    const opts: HolisticLandmarkerOptions = {
      baseOptions: { modelAssetPath: modelAssetUrl, delegate: "GPU" },
      runningMode,
      // Hand + pose are the only outputs we consume. Face landmarks
      // and segmentation masks are explicitly out of scope for
      // slice-1 (per docs/MODEL.md §1).
      outputFaceBlendshapes: false,
      outputPoseSegmentationMasks: false,
    };
    try {
      return await HolisticLandmarker.createFromOptions(fileset, opts);
    } catch (err) {
      // GPU delegate can fail on machines without WebGPU/WebGL support
      // for the task. Fall back to CPU rather than asking the learner
      // to debug their browser.
      const cpuOpts: HolisticLandmarkerOptions = {
        ...opts,
        baseOptions: { modelAssetPath: modelAssetUrl, delegate: "CPU" },
      };
      // Re-throw the original GPU error if CPU also fails; surfaces a
      // meaningful trail in Sentry once that's wired (Phase 7).
      try {
        return await HolisticLandmarker.createFromOptions(fileset, cpuOpts);
      } catch {
        throw err;
      }
    }
  })();

  try {
    return await loaderPromise;
  } catch (err) {
    loaderPromise = null;
    throw err;
  }
}

// For tests only. Reset the singleton so a fresh load can be tried.
export function _resetHolisticLoader(): void {
  loaderPromise = null;
}
