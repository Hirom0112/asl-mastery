// In-browser keypoint extraction — runs the from-scratch ONNX detector +
// landmark + pose models on each webcam frame to produce the RawFrame that
// frameToFeaturesV2 (sign_matcher.ts) turns into the 108D hand-norm features
// the v3 classifier consumes.
//
// Faithful TS port of scripts/live_demo.py's detect_hands +
// landmarks_for_bboxes (+ a simplified pose pass). Model artifacts (served
// from /public/models, not /artifacts/onnx):
//   /models/hand_detector_v0.onnx          (in 1×3×320×320, STRIDE 4 → 80² heatmap+size)
//   /models/hand_landmarks_v2_combined.onnx (in 1×3×224×224 → coords[21,2] in [0,1])
//   /models/pose_detector_v0.onnx          (in 1×3×256×256 → coords[8,2] in [0,1])
// All models expect RGB, CHW, float32 in [0,1].
//
// NOTE: the landmark model is v2_combined (+COCO-WholeBody, 10.84px clean /
// 21.61px in-the-wild) — the SAME landmarker the v3 classifier's trajectories
// were extracted with. Earlier we shipped the green-screen-only v0 here, which
// was a train/serve skew (classifier fed worse landmarks than it trained on).
//
// NOTE: this is written to match the Python pipeline numerically but has NOT
// yet been validated against a live webcam. Known item to verify in-browser:
// the POSE anchor — the Python demo crops pose on a face/upper-body box; here
// we run pose on the full frame (simpler) which can shift the body anchor used
// for body-relative location features. Revisit after the first browser run.

import type { InferenceSession, Tensor } from "onnxruntime-web";

import type { RawFrame } from "./sign_matcher";

const DET_SIZE = 320;
const DET_STRIDE = 4;
const LM_SIZE = 224;
const POSE_SIZE = 256;
const NUM_HAND_KP = 21;
const NUM_POSE_KP = 8;

// detect_hands thresholds (mirror live_demo defaults)
const DET_THRESHOLD = 0.3;
const SECOND_HAND_THRESHOLD = 0.5;
const IOU_DEDUP = 0.5;
const MAX_HANDS = 2;
const DET_TOPK = 8;
const LM_PAD_FRAC = 0.2;

type Box = [number, number, number, number]; // x0,y0,x1,y1 in source px

export interface KeypointModels {
  detector: InferenceSession;
  landmarks: InferenceSession;
  pose: InferenceSession;
}

let ortMod: typeof import("onnxruntime-web") | null = null;
async function ort() {
  if (!ortMod) ortMod = await import("onnxruntime-web");
  return ortMod;
}

export async function loadKeypointModels(baseUrl = "/models"): Promise<KeypointModels> {
  const o = await ort();
  const opts = { executionProviders: ["webgpu", "wasm"] as const };
  const [detector, landmarks, pose] = await Promise.all([
    o.InferenceSession.create(`${baseUrl}/hand_detector_v0.onnx`, opts),
    o.InferenceSession.create(`${baseUrl}/hand_landmarks_v2_combined.onnx`, opts),
    o.InferenceSession.create(`${baseUrl}/pose_detector_v0.onnx`, opts),
  ]);
  return { detector, landmarks, pose };
}

// --- preprocessing: draw a (cropped) region of `src` into a size×size RGB CHW
// float32 [0,1] tensor via an offscreen canvas. --------------------------------
const _canvas = typeof document !== "undefined" ? document.createElement("canvas") : null;

function regionToCHW(
  src: CanvasImageSource,
  sx: number,
  sy: number,
  sw: number,
  sh: number,
  size: number,
): Float32Array {
  if (!_canvas) throw new Error("keypoints.ts requires a DOM canvas (client-only)");
  _canvas.width = size;
  _canvas.height = size;
  const ctx = _canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(src, sx, sy, sw, sh, 0, 0, size, size);
  const { data } = ctx.getImageData(0, 0, size, size); // RGBA, row-major
  const chw = new Float32Array(3 * size * size);
  const plane = size * size;
  for (let p = 0; p < plane; p++) {
    chw[p] = data[p * 4] / 255; // R
    chw[plane + p] = data[p * 4 + 1] / 255; // G
    chw[2 * plane + p] = data[p * 4 + 2] / 255; // B
  }
  return chw;
}

async function makeTensor(chw: Float32Array, size: number): Promise<Tensor> {
  const o = await ort();
  return new o.Tensor("float32", chw, [1, 3, size, size]);
}

function iou(a: Box, b: Box): number {
  const x0 = Math.max(a[0], b[0]),
    y0 = Math.max(a[1], b[1]);
  const x1 = Math.min(a[2], b[2]),
    y1 = Math.min(a[3], b[3]);
  const iw = Math.max(0, x1 - x0),
    ih = Math.max(0, y1 - y0);
  const inter = iw * ih;
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  const u = areaA + areaB - inter;
  return u <= 0 ? 0 : inter / u;
}

// --- detector: port of live_demo.detect_hands -------------------------------
async function detectHands(
  m: KeypointModels,
  src: CanvasImageSource,
  srcW: number,
  srcH: number,
): Promise<{ boxes: Box[]; scores: number[] }> {
  const chw = regionToCHW(src, 0, 0, srcW, srcH, DET_SIZE);
  const out = await m.detector.run({ [m.detector.inputNames[0]]: await makeTensor(chw, DET_SIZE) });
  // outputs: heatmap (1,1,G,G) logits, size (1,2,G,G). Names are model-defined;
  // assume order [heatmap, size] per the exported HandDetector.
  const hm = out[m.detector.outputNames[0]];
  const szMap = out[m.detector.outputNames[1]];
  const G = hm.dims[hm.dims.length - 1] as number;
  const hmData = hm.data as Float32Array;
  const szData = szMap.data as Float32Array;
  const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

  // 3×3 max-pool peak detection: a cell is a peak if its prob == local 3×3 max.
  const scored: { s: number; box: Box }[] = [];
  const sxScale = srcW / DET_SIZE,
    syScale = srcH / DET_SIZE;
  for (let cy = 0; cy < G; cy++) {
    for (let cx = 0; cx < G; cx++) {
      const prob = sigmoid(hmData[cy * G + cx]);
      if (prob < DET_THRESHOLD) continue;
      let isPeak = true;
      for (let dy = -1; dy <= 1 && isPeak; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ny = cy + dy,
            nx = cx + dx;
          if (ny < 0 || ny >= G || nx < 0 || nx >= G) continue;
          if (sigmoid(hmData[ny * G + nx]) > prob) {
            isPeak = false;
            break;
          }
        }
      }
      if (!isPeak) continue;
      const w = szData[0 * G * G + cy * G + cx]; // size channel 0 = width
      const h = szData[1 * G * G + cy * G + cx]; // channel 1 = height
      const xc = (cx + 0.5) * DET_STRIDE,
        yc = (cy + 0.5) * DET_STRIDE;
      const box: Box = [
        (xc - w / 2) * sxScale,
        (yc - h / 2) * syScale,
        (xc + w / 2) * sxScale,
        (yc + h / 2) * syScale,
      ];
      scored.push({ s: prob, box });
    }
  }
  scored.sort((a, b) => b.s - a.s);

  const kept: { s: number; box: Box }[] = [];
  for (const { s, box } of scored.slice(0, DET_TOPK)) {
    if (kept.length && s < SECOND_HAND_THRESHOLD) break;
    if (kept.some((k) => iou(box, k.box) > IOU_DEDUP)) continue;
    kept.push({ s, box });
    if (kept.length >= MAX_HANDS) break;
  }
  return { boxes: kept.map((k) => k.box), scores: kept.map((k) => k.s) };
}

// --- landmarks: port of live_demo.landmarks_for_bboxes ----------------------
async function landmarksForBoxes(
  m: KeypointModels,
  src: CanvasImageSource,
  srcW: number,
  srcH: number,
  boxes: Box[],
): Promise<Array<Array<[number, number]>>> {
  const result: Array<Array<[number, number]>> = [];
  for (const [x0, y0, x1, y1] of boxes) {
    const bw = x1 - x0,
      bh = y1 - y0;
    const pad = LM_PAD_FRAC * Math.max(bw, bh);
    let cx0 = Math.max(0, Math.floor(x0 - pad));
    let cy0 = Math.max(0, Math.floor(y0 - pad));
    let cx1 = Math.min(srcW, Math.ceil(x1 + pad));
    let cy1 = Math.min(srcH, Math.ceil(y1 + pad));
    if (cx1 <= cx0 || cy1 <= cy0) {
      cx0 = 0;
      cy0 = 0;
      cx1 = srcW;
      cy1 = srcH;
    }
    const cw = cx1 - cx0,
      ch = cy1 - cy0;
    const chw = regionToCHW(src, cx0, cy0, cw, ch, LM_SIZE);
    const out = await m.landmarks.run({
      [m.landmarks.inputNames[0]]: await makeTensor(chw, LM_SIZE),
    });
    const coords = out[m.landmarks.outputNames[0]].data as Float32Array; // (1,21,2) in [0,1]
    const kps: Array<[number, number]> = [];
    for (let i = 0; i < NUM_HAND_KP; i++) {
      kps.push([cx0 + coords[i * 2] * cw, cy0 + coords[i * 2 + 1] * ch]);
    }
    result.push(kps);
  }
  return result;
}

// --- pose: simplified full-frame pass (see NOTE at top — anchor needs
// browser validation; the Python demo uses a face/upper-body crop) ----------
async function poseForFrame(
  m: KeypointModels,
  src: CanvasImageSource,
  srcW: number,
  srcH: number,
): Promise<Array<[number, number] | null>> {
  const chw = regionToCHW(src, 0, 0, srcW, srcH, POSE_SIZE);
  const out = await m.pose.run({ [m.pose.inputNames[0]]: await makeTensor(chw, POSE_SIZE) });
  const coords = out[m.pose.outputNames[0]].data as Float32Array; // (1,8,2) in [0,1]
  const pose: Array<[number, number] | null> = [];
  for (let i = 0; i < NUM_POSE_KP; i++) {
    pose.push([coords[i * 2] * srcW, coords[i * 2 + 1] * srcH]);
  }
  return pose;
}

// --- public: one frame → RawFrame for frameToFeaturesV2 ---------------------
export async function extractFrame(
  m: KeypointModels,
  src: CanvasImageSource,
  srcW: number,
  srcH: number,
): Promise<RawFrame> {
  const { boxes } = await detectHands(m, src, srcW, srcH);
  const handKps = await landmarksForBoxes(m, src, srcW, srcH, boxes);
  const pose = await poseForFrame(m, src, srcW, srcH);
  return { hands: handKps.map((keypoints) => ({ keypoints })), pose };
}
