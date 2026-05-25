// In-browser keypoint extraction — runs the from-scratch ONNX detector +
// landmark + pose (+ face) models over the recorded frames to produce the
// RawFrames that frameToFeaturesV2 (sign_matcher.ts) turns into the 108D
// hand-norm features the classifier consumes.
//
// FACE-ANCHORED pipeline (the 80.6%/93.5% model): faithful TS port of
// extract_trajectories_v2.py --pose-anchor face + _pose_inference. Per the A/B
// on Modal, anchoring the pose crop on the (stable) face — instead of the hand
// boxes — trains a materially better classifier (+4.8 top-1 over hand-anchored
// 75.8). So inference must produce the SAME face-anchored features:
//   per frame: hand detect + largest-face detect
//   → temporal smooth the face box across the clip (smooth_box)
//   → face-guard (drop ear/jaw hand false-positives)
//   → landmarks on kept hands
//   → pose cropped from the FACE-anchored upper-body box (fallback: hands)
//   → PoseEMA across the clip.
// extractFramesFaceAnchored() runs the whole sequence (temporal ops need all
// frames). The single-frame hand-anchored extractFrame() is kept for reference.
//
// Model artifacts (served from /public/models):
//   /models/hand_detector_v2.onnx            (1×3×320×320, STRIDE 4 → 80² heatmap+size)
//   /models/hand_landmarks_v2_combined.onnx  (1×3×224×224 → coords[21,2] in [0,1])
//   /models/pose_detector_v0.onnx            (1×3×256×256 → coords[8,2] in [0,1])
//   /models/face_detector_v0.onnx            (1×3×320×320, same head as hand det)
// All models expect RGB, CHW, float32 in [0,1].

import type { InferenceSession, Tensor } from "onnxruntime-web";

import type { RawFrame } from "./sign_matcher";

const DET_SIZE = 320;
const DET_STRIDE = 4;
const LM_SIZE = 224;
const POSE_SIZE = 256;
const NUM_HAND_KP = 21;
const NUM_POSE_KP = 8;

// Detector decode — mirrors extract_trajectories_v2._detect_hands_batched (the
// pipeline the classifier trained on), NOT live_demo. Very low threshold (the
// detector is undercalibrated, max heatmap prob ~0.27 on real ASL), strict
// top-2 peaks, NO IoU dedup, second hand only if its score clears 0.15,
// degenerate-box drop (the size head occasionally emits ~1px boxes).
const DET_THRESHOLD = 0.02;
const DET_TOPK = 2;
const SECOND_HAND_THRESHOLD = 0.15;
const MIN_BBOX_PX = 8;
const MAX_ASPECT_RATIO = 3.5;
const MAX_HANDS = 2;
const LM_PAD_FRAC = 0.2;

// Upper-body pose-crop heuristic — mirrors _pose_inference.upper_body_bbox_from_hands.
const POSE_UP_FACTOR = 3.5;
const POSE_SIDE_FACTOR = 1.5;
const POSE_DOWN_FACTOR = 0.5;
const POSE_BBOX_PAD_FRAC = 0.2;
const POSE_MIN_SIZE_FRAC = 0.4;
const POSE_MAX_ASPECT = 1.6;

// Face-anchored mode (mirrors extract_trajectories_v2 --pose-anchor face).
const FACE_DET_THRESHOLD = 0.3; // FACE_CONF_THRESHOLD
const FACE_TOPK = 5;
const FACE_SMOOTH_ALPHA = 0.4; // smooth_box
const FACE_MAX_MISS = 8;
const POSE_EMA_ALPHA = 0.25; // PoseEMA
// upper_body_bbox_from_face factors
const FACE_POSE_DOWN_FACTOR = 4.2;
const FACE_POSE_SIDE_FACTOR = 1.9;
const FACE_POSE_UP_FACTOR = 0.5;
const FACE_POSE_PAD_FRAC = 0.12;
// face-guard thresholds
const FACE_GUARD_INSIDE_FRAC = 0.6;
const FACE_GUARD_AREA_FRAC = 0.22;

type Box = [number, number, number, number]; // x0,y0,x1,y1 in source px

export interface KeypointModels {
  detector: InferenceSession;
  landmarks: InferenceSession;
  pose: InferenceSession;
  face: InferenceSession;
}

let ortMod: typeof import("onnxruntime-web") | null = null;
async function ort() {
  if (!ortMod) ortMod = await import("onnxruntime-web");
  return ortMod;
}

export async function loadKeypointModels(baseUrl = "/models"): Promise<KeypointModels> {
  const o = await ort();
  // logSeverityLevel 3 = errors only: silences ORT's benign "node not assigned
  // to preferred EP" warnings (shape ops fall back to CPU by design).
  const opts = { executionProviders: ["webgpu", "wasm"] as const, logSeverityLevel: 3 as const };
  const [detector, landmarks, pose, face] = await Promise.all([
    o.InferenceSession.create(`${baseUrl}/hand_detector_v2.onnx`, opts),
    o.InferenceSession.create(`${baseUrl}/hand_landmarks_v2_combined.onnx`, opts),
    o.InferenceSession.create(`${baseUrl}/pose_detector_v0.onnx`, opts),
    o.InferenceSession.create(`${baseUrl}/face_detector_v0.onnx`, opts),
  ]);
  return { detector, landmarks, pose, face };
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

// --- detector: port of extract_trajectories_v2._detect_hands_batched --------
async function detectHands(
  m: KeypointModels,
  src: CanvasImageSource,
  srcW: number,
  srcH: number,
): Promise<{ boxes: Box[]; scores: number[] }> {
  const chw = regionToCHW(src, 0, 0, srcW, srcH, DET_SIZE);
  const out = await m.detector.run({ [m.detector.inputNames[0]]: await makeTensor(chw, DET_SIZE) });
  // outputs: heatmap (1,1,G,G) logits, size (1,2,G,G), in [heatmap, size] order.
  const hm = out[m.detector.outputNames[0]];
  const szMap = out[m.detector.outputNames[1]];
  const G = hm.dims[hm.dims.length - 1] as number;
  const hmData = hm.data as Float32Array;
  const szData = szMap.data as Float32Array;
  const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

  // 3×3 max-pool peak detection: a cell is a peak if its prob == local 3×3 max
  // and clears the (very low) threshold. Same as the Python max_pool2d peaks.
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

  // Top-2 peaks → drop degenerate boxes → top-1 always + a second hand only if
  // its score clears SECOND_HAND_THRESHOLD. No IoU dedup (matches extraction).
  const valid: { s: number; box: Box }[] = [];
  for (const cand of scored.slice(0, DET_TOPK)) {
    const bw = cand.box[2] - cand.box[0];
    const bh = cand.box[3] - cand.box[1];
    if (bw < MIN_BBOX_PX || bh < MIN_BBOX_PX) continue;
    if (Math.max(bw, bh) / Math.max(Math.min(bw, bh), 1e-6) > MAX_ASPECT_RATIO) continue;
    valid.push(cand);
  }
  const kept: { s: number; box: Box }[] = [];
  if (valid.length) {
    kept.push(valid[0]);
    if (valid.length > 1 && valid[1].s >= SECOND_HAND_THRESHOLD) kept.push(valid[1]);
  }
  return {
    boxes: kept.slice(0, MAX_HANDS).map((k) => k.box),
    scores: kept.slice(0, MAX_HANDS).map((k) => k.s),
  };
}

// --- landmarks: port of extract_trajectories_v2._landmarks_batched ----------
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
    let cx1 = Math.min(srcW, Math.floor(x1 + pad));
    let cy1 = Math.min(srcH, Math.floor(y1 + pad));
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

// --- pose: hand-anchored upper-body crop (port of
// _pose_inference.upper_body_bbox_from_hands + pose_batched_with_bboxes). -----
function upperBodyBboxFromHands(boxes: Box[], imgW: number, imgH: number): Box | null {
  if (!boxes.length) return null;
  let x0 = Math.min(...boxes.map((b) => b[0]));
  let y0 = Math.min(...boxes.map((b) => b[1]));
  let x1 = Math.max(...boxes.map((b) => b[2]));
  let y1 = Math.max(...boxes.map((b) => b[3]));
  const hw = Math.max(...boxes.map((b) => b[2] - b[0]));
  const hh = Math.max(...boxes.map((b) => b[3] - b[1]));
  x0 -= POSE_SIDE_FACTOR * hw;
  x1 += POSE_SIDE_FACTOR * hw;
  y0 -= POSE_UP_FACTOR * hh;
  y1 += POSE_DOWN_FACTOR * hh;
  const pad = POSE_BBOX_PAD_FRAC * Math.max(x1 - x0, y1 - y0);
  x0 -= pad;
  y0 -= pad;
  x1 += pad;
  y1 += pad;
  // Enforce a minimum size vs the frame (guards the "hands at face level" case).
  const minSide = POSE_MIN_SIZE_FRAC * Math.min(imgW, imgH);
  const curW = x1 - x0;
  const curH = y1 - y0;
  if (curW < minSide) {
    const cx = (x0 + x1) / 2;
    x0 = cx - minSide / 2;
    x1 = cx + minSide / 2;
  }
  if (curH < minSide) {
    const cy = (y0 + y1) / 2;
    y0 = cy - minSide / 2;
    y1 = cy + minSide / 2;
  }
  x0 = Math.max(0, Math.floor(x0));
  y0 = Math.max(0, Math.floor(y0));
  x1 = Math.min(imgW, Math.floor(x1));
  y1 = Math.min(imgH, Math.floor(y1));
  if (x1 <= x0 || y1 <= y0) return null;
  // Cap aspect-ratio distortion at 1.6× (pose net trained on ~1:1 crops).
  const w = x1 - x0;
  const h = y1 - y0;
  if (w > h * POSE_MAX_ASPECT) {
    const need = Math.floor(w / POSE_MAX_ASPECT) - h;
    const addDown = Math.min(need, imgH - y1);
    y1 += addDown;
    const addUp = Math.min(need - addDown, y0);
    y0 -= addUp;
  } else if (h > w * POSE_MAX_ASPECT) {
    const need = Math.floor(h / POSE_MAX_ASPECT) - w;
    const half = Math.floor(need / 2);
    const addLeft = Math.min(half, x0);
    const addRight = Math.min(need - addLeft, imgW - x1);
    x0 -= addLeft;
    x1 += addRight;
  }
  return [x0, y0, x1, y1];
}

async function poseForBoxes(
  m: KeypointModels,
  src: CanvasImageSource,
  srcW: number,
  srcH: number,
  boxes: Box[],
): Promise<Array<[number, number] | null>> {
  const bbox = upperBodyBboxFromHands(boxes, srcW, srcH);
  // No hands → no upper-body crop → pose missing (frame is dropped downstream).
  if (!bbox) return new Array<[number, number] | null>(NUM_POSE_KP).fill(null);
  const [bx0, by0, bx1, by1] = bbox;
  const cw = bx1 - bx0,
    ch = by1 - by0;
  const chw = regionToCHW(src, bx0, by0, cw, ch, POSE_SIZE);
  const out = await m.pose.run({ [m.pose.inputNames[0]]: await makeTensor(chw, POSE_SIZE) });
  const coords = out[m.pose.outputNames[0]].data as Float32Array; // (1,8,2) in [0,1] of crop
  const pose: Array<[number, number] | null> = [];
  for (let i = 0; i < NUM_POSE_KP; i++) {
    pose.push([bx0 + coords[i * 2] * cw, by0 + coords[i * 2 + 1] * ch]);
  }
  return pose;
}

// --- hand-anchored single-frame path (kept for reference / A-B) -------------
export async function extractFrame(
  m: KeypointModels,
  src: CanvasImageSource,
  srcW: number,
  srcH: number,
): Promise<RawFrame> {
  const { boxes } = await detectHands(m, src, srcW, srcH);
  const handKps = await landmarksForBoxes(m, src, srcW, srcH, boxes);
  const pose = await poseForBoxes(m, src, srcW, srcH, boxes);
  return { hands: handKps.map((keypoints) => ({ keypoints })), pose };
}

// ===========================================================================
// FACE-ANCHORED pipeline (the deployed 80.6% path) — mirrors
// extract_trajectories_v2.py --pose-anchor face + _pose_inference.
// ===========================================================================

// Largest face box above threshold (the person in front) — port of
// _detect_largest_box_batched on the face detector.
async function detectLargestFace(
  m: KeypointModels,
  src: CanvasImageSource,
  srcW: number,
  srcH: number,
): Promise<Box | null> {
  const chw = regionToCHW(src, 0, 0, srcW, srcH, DET_SIZE);
  const out = await m.face.run({ [m.face.inputNames[0]]: await makeTensor(chw, DET_SIZE) });
  const hm = out[m.face.outputNames[0]];
  const szMap = out[m.face.outputNames[1]];
  const G = hm.dims[hm.dims.length - 1] as number;
  const hmData = hm.data as Float32Array;
  const szData = szMap.data as Float32Array;
  const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
  const sxScale = srcW / DET_SIZE,
    syScale = srcH / DET_SIZE;
  const scored: { s: number; box: Box }[] = [];
  for (let cy = 0; cy < G; cy++) {
    for (let cx = 0; cx < G; cx++) {
      const prob = sigmoid(hmData[cy * G + cx]);
      if (prob < FACE_DET_THRESHOLD) continue;
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
      const w = szData[cy * G + cx];
      const h = szData[G * G + cy * G + cx];
      const xc = (cx + 0.5) * DET_STRIDE,
        yc = (cy + 0.5) * DET_STRIDE;
      scored.push({
        s: prob,
        box: [
          (xc - w / 2) * sxScale,
          (yc - h / 2) * syScale,
          (xc + w / 2) * sxScale,
          (yc + h / 2) * syScale,
        ],
      });
    }
  }
  scored.sort((a, b) => b.s - a.s);
  let best: Box | null = null;
  let bestArea = -1;
  for (const c of scored.slice(0, FACE_TOPK)) {
    const area = (c.box[2] - c.box[0]) * (c.box[3] - c.box[1]);
    if (area > bestArea) {
      bestArea = area;
      best = c.box;
    }
  }
  return best;
}

// Upper-body crop anchored on the FACE — port of upper_body_bbox_from_face.
function upperBodyBboxFromFace(face: Box | null, imgW: number, imgH: number): Box | null {
  if (!face) return null;
  const [fx0, fy0, fx1, fy1] = face;
  const fw = fx1 - fx0,
    fh = fy1 - fy0;
  if (fw <= 0 || fh <= 0) return null;
  const cx = (fx0 + fx1) / 2;
  let x0 = cx - FACE_POSE_SIDE_FACTOR * fw;
  let x1 = cx + FACE_POSE_SIDE_FACTOR * fw;
  let y0 = fy0 - FACE_POSE_UP_FACTOR * fh;
  let y1 = fy1 + FACE_POSE_DOWN_FACTOR * fh;
  const pad = FACE_POSE_PAD_FRAC * Math.max(x1 - x0, y1 - y0);
  x0 -= pad;
  y0 -= pad;
  x1 += pad;
  y1 += pad;
  x0 = Math.max(0, Math.floor(x0));
  y0 = Math.max(0, Math.floor(y0));
  x1 = Math.min(imgW, Math.floor(x1));
  y1 = Math.min(imgH, Math.floor(y1));
  if (x1 <= x0 || y1 <= y0) return null;
  return [x0, y0, x1, y1];
}

// Forward EMA over a per-frame box sequence — port of smooth_box / _smooth_boxes_temporal.
function smoothBoxesTemporal(boxes: (Box | null)[]): (Box | null)[] {
  const out: (Box | null)[] = [];
  let state: Box | null = null;
  let miss = 0;
  for (const b of boxes) {
    if (!b) {
      miss++;
      out.push(miss <= FACE_MAX_MISS ? state : null);
      if (miss > FACE_MAX_MISS) state = null;
      continue;
    }
    miss = 0;
    if (!state) state = [b[0], b[1], b[2], b[3]];
    else
      state = [
        FACE_SMOOTH_ALPHA * b[0] + (1 - FACE_SMOOTH_ALPHA) * state[0],
        FACE_SMOOTH_ALPHA * b[1] + (1 - FACE_SMOOTH_ALPHA) * state[1],
        FACE_SMOOTH_ALPHA * b[2] + (1 - FACE_SMOOTH_ALPHA) * state[2],
        FACE_SMOOTH_ALPHA * b[3] + (1 - FACE_SMOOTH_ALPHA) * state[3],
      ];
    out.push(state);
  }
  return out;
}

// Drop ear/jaw hand false-positives — port of _apply_face_guard.
function applyFaceGuard(handBoxes: Box[], face: Box | null): Box[] {
  if (!face || !handBoxes.length) return handBoxes;
  const [fgx1, fgy1, fgx2, fgy2] = face;
  const faceArea = Math.max(1e-6, (fgx2 - fgx1) * (fgy2 - fgy1));
  const mw = 0.35 * (fgx2 - fgx1),
    mh = 0.25 * (fgy2 - fgy1);
  const guard: Box = [fgx1 - mw, fgy1 - mh, fgx2 + mw, fgy2 + mh];
  const fracInside = (b: Box) => {
    const ix0 = Math.max(b[0], guard[0]),
      iy0 = Math.max(b[1], guard[1]);
    const ix1 = Math.min(b[2], guard[2]),
      iy1 = Math.min(b[3], guard[3]);
    const inter = Math.max(0, ix1 - ix0) * Math.max(0, iy1 - iy0);
    const ba = Math.max(1e-6, (b[2] - b[0]) * (b[3] - b[1]));
    return inter / ba;
  };
  return handBoxes.filter(
    (b) =>
      !(
        fracInside(b) > FACE_GUARD_INSIDE_FRAC &&
        (b[2] - b[0]) * (b[3] - b[1]) < FACE_GUARD_AREA_FRAC * faceArea
      ),
  );
}

// Pose on the FACE-anchored upper-body crop (fallback: hand-anchored) — port of
// pose_batched_with_face_bboxes (single frame).
async function poseFaceAnchored(
  m: KeypointModels,
  src: CanvasImageSource,
  srcW: number,
  srcH: number,
  face: Box | null,
  handBoxes: Box[],
): Promise<Array<[number, number] | null>> {
  let bbox = upperBodyBboxFromFace(face, srcW, srcH);
  if (!bbox) bbox = upperBodyBboxFromHands(handBoxes, srcW, srcH);
  if (!bbox) return new Array<[number, number] | null>(NUM_POSE_KP).fill(null);
  const [bx0, by0, bx1, by1] = bbox;
  const cw = bx1 - bx0,
    ch = by1 - by0;
  const chw = regionToCHW(src, bx0, by0, cw, ch, POSE_SIZE);
  const out = await m.pose.run({ [m.pose.inputNames[0]]: await makeTensor(chw, POSE_SIZE) });
  const coords = out[m.pose.outputNames[0]].data as Float32Array;
  const pose: Array<[number, number] | null> = [];
  for (let i = 0; i < NUM_POSE_KP; i++) {
    pose.push([bx0 + coords[i * 2] * cw, by0 + coords[i * 2 + 1] * ch]);
  }
  return pose;
}

// Temporal pose smoothing across the clip — port of PoseEMA ((0,0)/missing
// keypoints pass the prior value through).
function poseEMAOverFrames(
  poses: Array<Array<[number, number] | null>>,
): Array<Array<[number, number] | null>> {
  const out: Array<Array<[number, number] | null>> = [];
  let state: Array<[number, number] | null> | null = null;
  for (const p of poses) {
    if (!state) {
      state = p.map((k) => (k ? ([k[0], k[1]] as [number, number]) : null));
      out.push(state.map((k) => (k ? ([k[0], k[1]] as [number, number]) : null)));
      continue;
    }
    const cur: Array<[number, number] | null> = [];
    for (let i = 0; i < NUM_POSE_KP; i++) {
      const k = p[i];
      const prev = state[i];
      if (!k || (k[0] === 0 && k[1] === 0)) {
        cur.push(prev ? [prev[0], prev[1]] : null);
        continue;
      }
      if (!prev) {
        state[i] = [k[0], k[1]];
        cur.push([k[0], k[1]]);
        continue;
      }
      const x = POSE_EMA_ALPHA * k[0] + (1 - POSE_EMA_ALPHA) * prev[0];
      const y = POSE_EMA_ALPHA * k[1] + (1 - POSE_EMA_ALPHA) * prev[1];
      state[i] = [x, y];
      cur.push([x, y]);
    }
    out.push(cur);
  }
  return out;
}

// Public: recorded frames → per-frame RawFrames using the FACE-ANCHORED
// pipeline (temporal ops require the whole sequence, so this is batch-over-clip
// rather than per-frame). Mirrors extract_trajectories_v2 --pose-anchor face.
export async function extractFramesFaceAnchored(
  m: KeypointModels,
  frames: CanvasImageSource[],
  srcW: number,
  srcH: number,
): Promise<RawFrame[]> {
  const n = frames.length;
  // Pass 1: per-frame raw hand boxes + largest face box.
  const handBoxesPerFrame: Box[][] = [];
  const faceRaw: (Box | null)[] = [];
  for (let i = 0; i < n; i++) {
    const { boxes } = await detectHands(m, frames[i], srcW, srcH);
    handBoxesPerFrame.push(boxes);
    faceRaw.push(await detectLargestFace(m, frames[i], srcW, srcH));
  }
  // Temporal-smooth the face box across the clip.
  const faceSmooth = smoothBoxesTemporal(faceRaw);
  // Pass 2: face-guard hands → landmarks → face-anchored pose.
  const handKpsPerFrame: Array<Array<Array<[number, number]>>> = [];
  const poseRaw: Array<Array<[number, number] | null>> = [];
  for (let i = 0; i < n; i++) {
    const guarded = applyFaceGuard(handBoxesPerFrame[i], faceSmooth[i]);
    handKpsPerFrame.push(await landmarksForBoxes(m, frames[i], srcW, srcH, guarded));
    poseRaw.push(await poseFaceAnchored(m, frames[i], srcW, srcH, faceSmooth[i], guarded));
  }
  // Temporal-smooth the pose across the clip (PoseEMA).
  const poseSmooth = poseEMAOverFrames(poseRaw);
  const out: RawFrame[] = [];
  for (let i = 0; i < n; i++) {
    out.push({
      hands: handKpsPerFrame[i].map((keypoints) => ({ keypoints })),
      pose: poseSmooth[i],
    });
  }
  return out;
}
