// TypeScript port of training/detectors/{fit_templates,sign_matcher}.py.
//
// Keeps the same numeric semantics so a trajectory scored here matches what
// the Python pipeline produces on the same input (modulo float64↔float32
// rounding). The streaming surface mirrors StreamingMatcher in the Python
// module exactly; that's the API the practice runner will consume frame-by-
// frame from the live webcam under ADR-0012.

export const NUM_HAND_KP = 21;
export const NUM_POSE_KP = 8;
export const HAND_SLOT_DIMS = NUM_HAND_KP * 2; // 42
export const POSE_BASE = 2 * HAND_SLOT_DIMS; // 84
export const FEATURES_PER_FRAME = 2 * NUM_HAND_KP * 2 + NUM_POSE_KP * 2; // 100

export const DEFAULT_SCALE_FLOOR = 1.0;
export const NEAREST_CLIP_FALLBACK_THRESHOLD = 8;
export const NEAREST_CLIP_ISOTROPIC_VAR = 0.05;
export const DTW_RADIUS = 8;

// Sentinel for missing-feature elements. Python uses NaN; JS Float32 NaN is
// fine but we route through a typed sentinel to avoid branching on isNaN
// inside hot loops.
const MISSING = Number.NaN;
const isMissing = (x: number): boolean => Number.isNaN(x);

export interface HandFrame {
  // 21 keypoints; each is [x, y] in image pixels.
  keypoints: ReadonlyArray<readonly [number, number]>;
}

export interface RawFrame {
  hands?: ReadonlyArray<HandFrame>;
  // 8 pose keypoints in order [nose, neck, r_shoulder, l_shoulder, r_elbow,
  // l_elbow, r_wrist, l_wrist], each [x, y] or null/zero-pair if missing.
  pose?: ReadonlyArray<readonly [number, number] | null>;
}

// Template format: a single template (k=1) or a list of clusters (k>1 per
// P2 #12). The matcher accepts both shapes.
export interface SignTemplate {
  signId: string;
  T: number;
  F: number;
  // (T, F) flat row-major
  mean: Float32Array;
  // (T, F) flat row-major
  var: Float32Array;
  // optional raw clips, (n, T, F) flat row-major; used as nearest-clip
  // fallback when nClips < NEAREST_CLIP_FALLBACK_THRESHOLD
  clips?: Float32Array;
  nClips: number;
  dominantHandOnly: boolean;
}

export interface ThresholdEntry {
  threshold: number;
  achievedPrecision?: number;
  achievedRecall?: number;
}

// ---------------------------------------------------------------------------
// Feature extraction (mirrors fit_templates._frame_to_features)
// ---------------------------------------------------------------------------

interface PoseAnchorScale {
  ax: number;
  ay: number;
  scale: number;
}

function valid(p: readonly [number, number] | null | undefined): p is readonly [number, number] {
  return p != null && !(p[0] === 0 && p[1] === 0);
}

function poseAnchorAndScale(frame: RawFrame): PoseAnchorScale | null {
  const pose = frame.pose;
  if (!pose || pose.length < 4) return null;
  const nose = pose[0];
  const neck = pose[1];
  const rSh = pose[2];
  const lSh = pose[3];

  let ax: number;
  let ay: number;
  if (valid(neck)) {
    ax = neck[0];
    ay = neck[1];
  } else if (valid(rSh) && valid(lSh)) {
    ax = (rSh[0] + lSh[0]) / 2;
    ay = (rSh[1] + lSh[1]) / 2;
  } else if (valid(nose)) {
    ax = nose[0];
    ay = nose[1];
  } else {
    return null;
  }

  let scale: number;
  if (valid(rSh) && valid(lSh)) {
    scale = Math.hypot(rSh[0] - lSh[0], rSh[1] - lSh[1]);
  } else if (valid(nose) && valid(neck)) {
    scale = 1.5 * Math.hypot(nose[0] - neck[0], nose[1] - neck[1]);
  } else {
    scale = 0;
  }
  scale = Math.max(scale, DEFAULT_SCALE_FLOOR);
  return { ax, ay, scale };
}

export function frameToFeatures(frame: RawFrame): {
  feats: Float32Array;
  usedSlot1: boolean;
} {
  const feats = new Float32Array(FEATURES_PER_FRAME);
  feats.fill(MISSING);
  const pa = poseAnchorAndScale(frame);
  if (pa == null) return { feats, usedSlot1: false };
  const { ax, ay, scale } = pa;
  const invS = 1 / scale;

  let usedSlot1 = false;
  const hands = frame.hands ?? [];
  for (let h = 0; h < Math.min(hands.length, 2); h++) {
    const kps = hands[h].keypoints;
    if (kps.length < NUM_HAND_KP) continue;
    const wristX = kps[0][0];
    const slot = wristX < ax ? 0 : 1;
    const base = slot * HAND_SLOT_DIMS;
    // Collision handling: if this slot already has a closer-to-anchor wrist,
    // skip; otherwise overwrite. Matches the Python's prefer-closer rule.
    if (!isMissing(feats[base])) {
      const existingWx = feats[base] / invS + ax;
      if (Math.abs(wristX - ax) >= Math.abs(existingWx - ax)) continue;
    }
    for (let i = 0; i < NUM_HAND_KP; i++) {
      feats[base + i * 2 + 0] = (kps[i][0] - ax) * invS;
      feats[base + i * 2 + 1] = (kps[i][1] - ay) * invS;
    }
    if (slot === 1) usedSlot1 = true;
  }

  const pose = frame.pose ?? [];
  const nPose = Math.min(NUM_POSE_KP, pose.length);
  for (let i = 0; i < nPose; i++) {
    const p = pose[i];
    if (p == null) continue;
    feats[POSE_BASE + i * 2 + 0] = (p[0] - ax) * invS;
    feats[POSE_BASE + i * 2 + 1] = (p[1] - ay) * invS;
  }
  return { feats, usedSlot1 };
}

// ---------------------------------------------------------------------------
// v2 feature extraction (108D hand-relative) — mirrors
// fit_templates._frame_to_features_v2. This is what the deployed v3 classifier
// (sign_classifier_v3.onnx) consumes. Per slot (46):
//   [handshape 0:42 wrist-origin + hand-scaled][location 42:44 body-rel]
//   [orientation 44:46 hand-scaled]. Full (108): [slot0 0:46][slot1 46:92][pose 92:108].
// Handshape + orientation are hand-relative (wrist-origin, ‖kp9-kp0‖ scale);
// location + pose stay body-normalized (anchor + shoulder scale).
// ---------------------------------------------------------------------------

export const HAND_SLOT_DIMS_V2 = NUM_HAND_KP * 2 + 2 + 2; // 46
export const POSE_BASE_V2 = 2 * HAND_SLOT_DIMS_V2; // 92
export const FEATURES_PER_FRAME_V2 = POSE_BASE_V2 + NUM_POSE_KP * 2; // 108
const HAND_SCALE_EPSILON = 1e-6;

export function frameToFeaturesV2(frame: RawFrame): {
  feats: Float32Array;
  usedSlot1: boolean;
} {
  const feats = new Float32Array(FEATURES_PER_FRAME_V2);
  feats.fill(MISSING);
  const pa = poseAnchorAndScale(frame);
  if (pa == null) return { feats, usedSlot1: false };
  const { ax, ay, scale } = pa;
  const invS = 1 / scale;

  let usedSlot1 = false;
  const hands = frame.hands ?? [];
  for (let h = 0; h < Math.min(hands.length, 2); h++) {
    const kps = hands[h].keypoints;
    if (kps.length < NUM_HAND_KP) continue;
    const wristX = kps[0][0];
    const slot = wristX < ax ? 0 : 1;
    const base = slot * HAND_SLOT_DIMS_V2;
    // Collision: location lives at base+42; prefer the hand nearer the anchor.
    if (!isMissing(feats[base + 42])) {
      const existingWx = feats[base + 42] / invS + ax;
      if (Math.abs(wristX - ax) >= Math.abs(existingWx - ax)) continue;
    }
    const kp0x = kps[0][0];
    const kp0y = kps[0][1];
    const kp9x = kps[9][0];
    const kp9y = kps[9][1];
    const handScale = Math.hypot(kp9x - kp0x, kp9y - kp0y);
    // Location (always available): wrist relative to the body anchor.
    feats[base + 42] = (kp0x - ax) * invS;
    feats[base + 43] = (kp0y - ay) * invS;
    // Handshape + orientation need a non-degenerate hand scale.
    if (handScale >= HAND_SCALE_EPSILON) {
      const invH = 1 / handScale;
      for (let i = 0; i < NUM_HAND_KP; i++) {
        feats[base + i * 2 + 0] = (kps[i][0] - kp0x) * invH;
        feats[base + i * 2 + 1] = (kps[i][1] - kp0y) * invH;
      }
      feats[base + 44] = (kp9x - kp0x) * invH;
      feats[base + 45] = (kp9y - kp0y) * invH;
    }
    if (slot === 1) usedSlot1 = true;
  }

  const pose = frame.pose ?? [];
  const nPose = Math.min(NUM_POSE_KP, pose.length);
  for (let i = 0; i < nPose; i++) {
    const p = pose[i];
    if (p == null) continue;
    feats[POSE_BASE_V2 + i * 2 + 0] = (p[0] - ax) * invS;
    feats[POSE_BASE_V2 + i * 2 + 1] = (p[1] - ay) * invS;
  }
  return { feats, usedSlot1 };
}

// ---------------------------------------------------------------------------
// Mirror + resample
// ---------------------------------------------------------------------------

export function mirrorFeatures(traj: Float32Array, T: number, F: number): Float32Array {
  const out = new Float32Array(traj.length);
  for (let t = 0; t < T; t++) {
    for (let f = 0; f < F; f++) {
      const v = traj[t * F + f];
      // negate every even-indexed feature (x components)
      out[t * F + f] = f % 2 === 0 ? -v : v;
    }
    // swap hand slot 0 <-> slot 1
    for (let i = 0; i < HAND_SLOT_DIMS; i++) {
      const a = out[t * F + i];
      out[t * F + i] = out[t * F + HAND_SLOT_DIMS + i];
      out[t * F + HAND_SLOT_DIMS + i] = a;
    }
  }
  return out;
}

// Linear time-resample (n, F) → (T, F), NaN-aware per column.
export function resampleTrajectory(
  traj: Float32Array,
  n: number,
  T: number,
  F: number,
): Float32Array {
  if (n === T) return traj;
  const out = new Float32Array(T * F);
  if (n < 2) {
    // tile the only row
    for (let t = 0; t < T; t++) {
      for (let f = 0; f < F; f++) out[t * F + f] = traj[f];
    }
    return out;
  }
  for (let f = 0; f < F; f++) {
    // collect valid src points for this feature column
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < n; i++) {
      const v = traj[i * F + f];
      if (!isMissing(v)) {
        xs.push(i / (n - 1));
        ys.push(v);
      }
    }
    if (xs.length === 0) {
      for (let t = 0; t < T; t++) out[t * F + f] = MISSING;
      continue;
    }
    if (xs.length === 1) {
      for (let t = 0; t < T; t++) out[t * F + f] = ys[0];
      continue;
    }
    for (let t = 0; t < T; t++) {
      const dx = t / (T - 1);
      // binary-search xs for the bracketing interval
      let lo = 0;
      let hi = xs.length - 1;
      if (dx <= xs[0]) {
        out[t * F + f] = ys[0];
        continue;
      }
      if (dx >= xs[hi]) {
        out[t * F + f] = ys[hi];
        continue;
      }
      while (hi - lo > 1) {
        const m = (lo + hi) >> 1;
        if (xs[m] <= dx) lo = m;
        else hi = m;
      }
      const frac = (dx - xs[lo]) / (xs[hi] - xs[lo]);
      out[t * F + f] = ys[lo] + frac * (ys[hi] - ys[lo]);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

function perFrameMahalanobis(
  a: Float32Array,
  aOff: number,
  b: Float32Array,
  bOff: number,
  varRow: Float32Array,
  vOff: number,
  F: number,
  dominantHandOnly: boolean,
): number {
  let sum = 0;
  let nValid = 0;
  for (let f = 0; f < F; f++) {
    if (dominantHandOnly && f >= HAND_SLOT_DIMS && f < 2 * HAND_SLOT_DIMS) continue;
    const av = a[aOff + f];
    const bv = b[bOff + f];
    if (isMissing(av) || isMissing(bv)) continue;
    const d = av - bv;
    sum += (d * d) / varRow[vOff + f];
    nValid++;
  }
  if (nValid === 0) return 1e9;
  return sum / nValid;
}

export function dtwScore(
  traj: Float32Array,
  mean: Float32Array,
  varTF: Float32Array,
  T: number,
  F: number,
  dominantHandOnly: boolean,
  radius: number = DTW_RADIUS,
): number {
  const INF = 1e18;
  const W = T + 1;
  const D = new Float64Array(W * W);
  D.fill(INF);
  D[0] = 0;
  for (let i = 1; i <= T; i++) {
    const jMin = Math.max(1, i - radius);
    const jMax = Math.min(T, i + radius);
    for (let j = jMin; j <= jMax; j++) {
      const cost = perFrameMahalanobis(
        traj,
        (i - 1) * F,
        mean,
        (j - 1) * F,
        varTF,
        (j - 1) * F,
        F,
        dominantHandOnly,
      );
      const a = D[(i - 1) * W + j];
      const b = D[i * W + (j - 1)];
      const c = D[(i - 1) * W + (j - 1)];
      let best = a < b ? a : b;
      if (c < best) best = c;
      D[i * W + j] = cost + best;
    }
  }
  return D[T * W + T] / (2 * T);
}

function gaussianScore(
  traj: Float32Array,
  mean: Float32Array,
  varTF: Float32Array,
  T: number,
  F: number,
  dominantHandOnly: boolean,
): number {
  let sum = 0;
  let nValid = 0;
  for (let t = 0; t < T; t++) {
    for (let f = 0; f < F; f++) {
      if (dominantHandOnly && f >= HAND_SLOT_DIMS && f < 2 * HAND_SLOT_DIMS) continue;
      const v = traj[t * F + f];
      const m = mean[t * F + f];
      if (isMissing(v) || isMissing(m)) continue;
      const d = v - m;
      sum += (d * d) / varTF[t * F + f];
      nValid++;
    }
  }
  if (nValid === 0) return Infinity;
  return sum / nValid;
}

function nearestClipScore(
  traj: Float32Array,
  clips: Float32Array,
  nClips: number,
  T: number,
  F: number,
  dominantHandOnly: boolean,
): number {
  let best = Infinity;
  const stride = T * F;
  for (let c = 0; c < nClips; c++) {
    const cOff = c * stride;
    let sum = 0;
    let nValid = 0;
    for (let t = 0; t < T; t++) {
      for (let f = 0; f < F; f++) {
        if (dominantHandOnly && f >= HAND_SLOT_DIMS && f < 2 * HAND_SLOT_DIMS) continue;
        const v = traj[t * F + f];
        const u = clips[cOff + t * F + f];
        if (isMissing(v) || isMissing(u)) continue;
        const d = v - u;
        sum += d * d;
        nValid++;
      }
    }
    if (nValid === 0) continue;
    const s = sum / NEAREST_CLIP_ISOTROPIC_VAR / nValid;
    if (s < best) best = s;
  }
  return best;
}

export function mahalanobisScore(traj: Float32Array, tpl: SignTemplate, useDtw = true): number {
  const { mean, var: variance, T, F, dominantHandOnly, clips, nClips } = tpl;
  const mirrored = mirrorFeatures(traj, T, F);

  if (clips && nClips > 0 && nClips < NEAREST_CLIP_FALLBACK_THRESHOLD) {
    const a = nearestClipScore(traj, clips, nClips, T, F, dominantHandOnly);
    const b = nearestClipScore(mirrored, clips, nClips, T, F, dominantHandOnly);
    return Math.min(a, b);
  }

  if (useDtw) {
    const a = dtwScore(traj, mean, variance, T, F, dominantHandOnly);
    const b = dtwScore(mirrored, mean, variance, T, F, dominantHandOnly);
    return Math.min(a, b);
  }

  const a = gaussianScore(traj, mean, variance, T, F, dominantHandOnly);
  const b = gaussianScore(mirrored, mean, variance, T, F, dominantHandOnly);
  return Math.min(a, b);
}

// ---------------------------------------------------------------------------
// Streaming matcher (mirrors Python's StreamingMatcher)
// ---------------------------------------------------------------------------

export interface StreamingStep {
  ready: boolean;
  scores: Record<string, number>;
  targetPassedNow: boolean;
  targetStreak: number;
  unlocked: boolean;
  top1: string | null;
  top1Score: number | null;
}

export interface StreamingMatcherOptions {
  timeSteps?: number;
  windowFrames?: number;
  consecutiveRequired?: number;
  thresholds?: Record<string, ThresholdEntry>;
}

export class StreamingMatcher {
  private readonly T: number;
  private readonly window: number;
  private readonly consecutiveRequired: number;
  private readonly F: number;
  private readonly templates: Record<string, SignTemplate>;
  private readonly thresholds: Record<string, ThresholdEntry>;
  // ring of (F,) feature vectors
  private readonly frames: Float32Array[] = [];
  private readonly streaks = new Map<string, number>();

  constructor(templates: Record<string, SignTemplate>, opts: StreamingMatcherOptions = {}) {
    this.templates = templates;
    this.thresholds = opts.thresholds ?? {};
    this.T = opts.timeSteps ?? 32;
    this.window = Math.max(opts.windowFrames ?? 48, this.T);
    this.consecutiveRequired = Math.max(opts.consecutiveRequired ?? 3, 1);
    // F is fixed by the feature spec; could be derived from any template,
    // but pin it to FEATURES_PER_FRAME so empty-template states still work.
    this.F = FEATURES_PER_FRAME;
  }

  pushFeatures(feat: Float32Array): void {
    if (this.frames.length >= this.window) this.frames.shift();
    this.frames.push(feat);
  }

  pushFrame(frame: RawFrame): void {
    const { feats } = frameToFeatures(frame);
    this.pushFeatures(feats);
  }

  reset(): void {
    this.frames.length = 0;
    this.streaks.clear();
  }

  ready(): boolean {
    return this.frames.length >= this.T;
  }

  currentTrajectory(): Float32Array | null {
    if (!this.ready()) return null;
    const n = this.frames.length;
    const flat = new Float32Array(n * this.F);
    for (let i = 0; i < n; i++) flat.set(this.frames[i], i * this.F);
    return resampleTrajectory(flat, n, this.T, this.F);
  }

  step(targetSignId: string, sliceSignIds?: string[]): StreamingStep {
    const slice = sliceSignIds ?? [targetSignId];
    const traj = this.currentTrajectory();
    if (traj == null) {
      return {
        ready: false,
        scores: {},
        targetPassedNow: false,
        targetStreak: this.streaks.get(targetSignId) ?? 0,
        unlocked: false,
        top1: null,
        top1Score: null,
      };
    }

    const scores: Record<string, number> = {};
    for (const sid of slice) {
      const tpl = this.templates[sid];
      if (!tpl) continue;
      scores[sid] = mahalanobisScore(traj, tpl);
    }

    for (const [sid, sc] of Object.entries(scores)) {
      const thr = this.thresholdFor(sid);
      if (sc <= thr) {
        this.streaks.set(sid, (this.streaks.get(sid) ?? 0) + 1);
      } else {
        this.streaks.set(sid, 0);
      }
    }

    let top1: string | null = null;
    let top1Score = Infinity;
    for (const [sid, sc] of Object.entries(scores)) {
      if (sc < top1Score) {
        top1Score = sc;
        top1 = sid;
      }
    }

    const targetScore = scores[targetSignId] ?? Infinity;
    const targetPassedNow = targetScore <= this.thresholdFor(targetSignId);
    const targetStreak = this.streaks.get(targetSignId) ?? 0;
    return {
      ready: true,
      scores,
      targetPassedNow,
      targetStreak,
      unlocked: targetStreak >= this.consecutiveRequired,
      top1,
      top1Score: top1 == null ? null : top1Score,
    };
  }

  private thresholdFor(signId: string): number {
    const entry = this.thresholds[signId];
    if (!entry) return Infinity;
    const thr = entry.threshold;
    if (!Number.isFinite(thr)) return Infinity;
    return thr;
  }
}
