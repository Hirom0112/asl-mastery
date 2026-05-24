"""Phase 4 Slice 4.2 — fit per-sign trajectory templates.

For each sign in the frozen 75-sign vocabulary:
  - Gather all per-clip trajectories from data/trajectories/<sign>/*.json
  - Time-normalize each trajectory to T fixed frames (default 32)
  - Stack the (T, num_features) trajectories across clips
  - Compute mean trajectory + per-feature per-timestep diagonal variance
  - Save as data/templates/<sign>.npz

Features per frame (flattened):
  - 2 hands × 21 keypoints × 2 coords = 84
  - 8 pose keypoints × 2 coords = 16
  - Total: 100 floats per frame

Coordinates:
  - Translation: anchored to neck (or shoulder-midpoint, or nose).
  - Scale: divided by shoulder-pixel-distance so user distance from camera
    does not dominate Mahalanobis. Fallback: 1.5 × neck-to-nose, else 1.0.

Hand slot assignment (v6 — mirror-aware):
  - Slot 0 = wrist.x < neck.x ("on the left of frame")
  - Slot 1 = the other hand
  - At inference, callers should also score the horizontally mirrored
    trajectory and take min — handles webcam mirror flip.

Per-sign metadata saved in the .npz:
  - mean (T, F), var (T, F) — pooled Gaussian per-step template
  - dominant_hand_only (bool) — set when <30% of clips show a 2nd hand;
    matcher should zero-out slot-1 features when this flag is set
  - clips (n_clips, T, F) — raw resampled clips, used by the matcher's
    nearest-clip fallback for low-clip signs
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


NUM_HAND_KP = 21
NUM_POSE_KP = 8
HAND_EMBED_DIM = 128                       # HandshapeEncoder output (Phase 4.6)
FEATURES_PER_FRAME = 2 * NUM_HAND_KP * 2 + NUM_POSE_KP * 2  # 100 (baseline, no embed)
FEATURES_PER_FRAME_WITH_EMBED = 2 * (NUM_HAND_KP * 2 + HAND_EMBED_DIM) + NUM_POSE_KP * 2  # 356
HAND_SLOT_DIMS = NUM_HAND_KP * 2           # 42 (kpts-only)
HAND_SLOT_DIMS_WITH_EMBED = NUM_HAND_KP * 2 + HAND_EMBED_DIM  # 170 (kpts + embed)
POSE_BASE = 2 * HAND_SLOT_DIMS             # 84
POSE_BASE_WITH_EMBED = 2 * HAND_SLOT_DIMS_WITH_EMBED  # 340

# v2 hand-relative layout (§9): per slot [handshape 0:42][location 42:44][orientation 44:46]
HAND_SLOT_DIMS_V2 = NUM_HAND_KP * 2 + 2 + 2  # 46
POSE_BASE_V2 = 2 * HAND_SLOT_DIMS_V2         # 92
FEATURES_PER_FRAME_V2 = POSE_BASE_V2 + NUM_POSE_KP * 2  # 108

DEFAULT_DOMINANT_HAND_THRESHOLD = 0.30
DEFAULT_SCALE_FLOOR = 1.0  # avoid divide-by-zero when shoulders aren't found
HAND_SCALE_EPSILON = 1e-6  # below this ||kp9-kp0||, leave handshape/orientation NaN


def _pose_anchor_and_scale(frame: dict) -> tuple[tuple[float, float], float] | None:
    """Return ((ax, ay), scale) — origin point + body-size normalizer.

    Origin = neck if visible, else mean of shoulders, else nose.
    Scale = ||r_sh - l_sh|| (shoulder-pixel-distance). Falls back to
    1.5 × ||nose - neck|| if shoulders missing, then DEFAULT_SCALE_FLOOR.
    """
    pose = frame.get("pose")
    if not pose or len(pose) < 4:
        return None
    nose, neck, r_sh, l_sh = pose[0], pose[1], pose[2], pose[3]

    def _valid(p) -> bool:
        return p is not None and not (p[0] == 0 and p[1] == 0)

    if _valid(neck):
        anchor = (float(neck[0]), float(neck[1]))
    elif _valid(r_sh) and _valid(l_sh):
        anchor = ((float(r_sh[0]) + float(l_sh[0])) / 2.0,
                  (float(r_sh[1]) + float(l_sh[1])) / 2.0)
    elif _valid(nose):
        anchor = (float(nose[0]), float(nose[1]))
    else:
        return None

    if _valid(r_sh) and _valid(l_sh):
        dx = float(r_sh[0]) - float(l_sh[0])
        dy = float(r_sh[1]) - float(l_sh[1])
        scale = math.hypot(dx, dy)
    elif _valid(nose) and _valid(neck):
        scale = 1.5 * math.hypot(float(nose[0]) - float(neck[0]),
                                  float(nose[1]) - float(neck[1]))
    else:
        scale = 0.0
    scale = max(scale, DEFAULT_SCALE_FLOOR)
    return anchor, scale


def _frame_to_features(frame: dict, with_embedding: bool = False,
                       norm: str = "body") -> tuple[np.ndarray, bool]:
    """Flatten one frame into the per-frame feature vector (NaN where missing).

    Returns (feats, has_second_hand).

    norm="body" (DEFAULT, v1): body-relative keypoints.
      with_embedding=False (100D, Phase 4 baseline):
        [slot0 kpts: 42][slot1 kpts: 42][pose: 16]
      with_embedding=True (356D, Phase 4.6 handshape encoder):
        [slot0 kpts: 42 + slot0 embed: 128][slot1 kpts+embed: 170][pose: 16]
      Translation: subtract pose anchor. Scale: divide by shoulder-pixel-distance.

    norm="hand" (v2, §9; 108D, no embeddings): hand-relative handshape.
      Per slot (46): [handshape 0:42] = (kp_i-kp0)/||kp9-kp0|| (wrist-relative,
      hand-scaled); [location 42:44] = (kp0-anchor)/body_scale; [orientation
      44:46] = unit vec of (kp9-kp0). Full: [slot0 0:46][slot1 46:92][pose 92:108].
      Pose stays body-normalized like v1.

    Hand slot: 0 if wrist.x < neck.x (anatomically left side of frame),
    1 otherwise. At inference, callers should also score the mirrored
    trajectory and take min, since webcam preview is typically mirrored.
    """
    if norm not in ("body", "hand"):
        raise ValueError(f"_frame_to_features: unknown norm={norm!r}")
    if norm == "hand" and with_embedding:
        raise ValueError("v2 hand normalization does not support embeddings")

    if norm == "hand":
        return _frame_to_features_v2(frame)

    if with_embedding:
        dim = FEATURES_PER_FRAME_WITH_EMBED
        slot_dims = HAND_SLOT_DIMS_WITH_EMBED
        pose_base = POSE_BASE_WITH_EMBED
    else:
        dim = FEATURES_PER_FRAME
        slot_dims = HAND_SLOT_DIMS
        pose_base = POSE_BASE

    feats = np.full(dim, np.nan, dtype=np.float32)
    pa = _pose_anchor_and_scale(frame)
    if pa is None:
        return feats, False
    (ax, ay), scale = pa
    inv_s = 1.0 / scale

    hands = frame.get("hands") or []
    used_slot1 = False
    if hands:
        for hand in hands[:2]:
            kps = hand.get("keypoints") or []
            if len(kps) < NUM_HAND_KP:
                continue
            wrist_x = float(kps[0][0])
            slot = 0 if wrist_x < ax else 1
            base = slot * slot_dims
            if not np.isnan(feats[base]):
                existing_wx = feats[base + 0] / inv_s + ax
                if abs(wrist_x - ax) >= abs(existing_wx - ax):
                    continue
            for i in range(NUM_HAND_KP):
                feats[base + i * 2 + 0] = (float(kps[i][0]) - ax) * inv_s
                feats[base + i * 2 + 1] = (float(kps[i][1]) - ay) * inv_s
            if with_embedding:
                emb = hand.get("embedding")
                if emb is not None and len(emb) == HAND_EMBED_DIM:
                    embed_base = base + NUM_HAND_KP * 2
                    feats[embed_base:embed_base + HAND_EMBED_DIM] = np.asarray(emb, dtype=np.float32)
                # If embedding is missing the corresponding 128 slots remain NaN —
                # downstream NaN handling (nan_to_num in the classifier forward)
                # treats them as 0, which is the desired "missing-modality" signal.
            if slot == 1:
                used_slot1 = True

    pose = frame.get("pose") or []
    for i in range(min(NUM_POSE_KP, len(pose))):
        p = pose[i]
        if p is None:
            continue
        feats[pose_base + i * 2 + 0] = (float(p[0]) - ax) * inv_s
        feats[pose_base + i * 2 + 1] = (float(p[1]) - ay) * inv_s
    return feats, used_slot1


def _frame_to_features_v2(frame: dict) -> tuple[np.ndarray, bool]:
    """v2 (norm="hand") per-frame feature vector — see §9 / _frame_to_features.

    Layout per slot (46): [handshape 0:42][location 42:44][orientation 44:46].
    Full vector (108): [slot0 0:46][slot1 46:92][pose 92:108].
    Handshape + orientation are hand-relative (wrist-origin, hand-scaled);
    location + pose stay body-normalized (anchor + body scale).
    """
    feats = np.full(FEATURES_PER_FRAME_V2, np.nan, dtype=np.float32)
    pa = _pose_anchor_and_scale(frame)
    if pa is None:
        return feats, False
    (ax, ay), scale = pa
    inv_s = 1.0 / scale

    hands = frame.get("hands") or []
    used_slot1 = False
    if hands:
        for hand in hands[:2]:
            kps = hand.get("keypoints") or []
            if len(kps) < NUM_HAND_KP:
                continue
            wrist_x = float(kps[0][0])
            slot = 0 if wrist_x < ax else 1
            base = slot * HAND_SLOT_DIMS_V2
            if not np.isnan(feats[base + 42]):
                # location already filled — keep the hand nearer the anchor.
                existing_wx = feats[base + 42] / inv_s + ax
                if abs(wrist_x - ax) >= abs(existing_wx - ax):
                    continue
            kp0x, kp0y = float(kps[0][0]), float(kps[0][1])
            kp9x, kp9y = float(kps[9][0]), float(kps[9][1])
            hand_scale = math.hypot(kp9x - kp0x, kp9y - kp0y)
            # location (always available): wrist relative to body anchor.
            feats[base + 42] = (kp0x - ax) * inv_s
            feats[base + 43] = (kp0y - ay) * inv_s
            # handshape + orientation need a non-degenerate hand scale; guard
            # divide-by-zero by leaving them NaN (don't emit inf/nan blowups).
            if hand_scale >= HAND_SCALE_EPSILON:
                inv_h = 1.0 / hand_scale
                for i in range(NUM_HAND_KP):
                    feats[base + i * 2 + 0] = (float(kps[i][0]) - kp0x) * inv_h
                    feats[base + i * 2 + 1] = (float(kps[i][1]) - kp0y) * inv_h
                feats[base + 44] = (kp9x - kp0x) * inv_h
                feats[base + 45] = (kp9y - kp0y) * inv_h
            if slot == 1:
                used_slot1 = True

    pose = frame.get("pose") or []
    for i in range(min(NUM_POSE_KP, len(pose))):
        p = pose[i]
        if p is None:
            continue
        feats[POSE_BASE_V2 + i * 2 + 0] = (float(p[0]) - ax) * inv_s
        feats[POSE_BASE_V2 + i * 2 + 1] = (float(p[1]) - ay) * inv_s
    return feats, used_slot1


def trajectory_has_embedding(frames: list[dict]) -> bool:
    """Inspect a trajectory's frames to determine if any hand has a 128D embedding.
    Used by loaders to auto-set with_embedding for _frame_to_features."""
    for f in frames:
        for hand in (f.get("hands") or []):
            emb = hand.get("embedding")
            if emb is not None and len(emb) == HAND_EMBED_DIM:
                return True
    return False


def _mirror_features(traj_TF: np.ndarray) -> np.ndarray:
    """Mirror a (T, F) trajectory horizontally. Supports F=100 (kpts-only) or
    F=356 (kpts + 128D handshape embedding per slot).

    For the embedding portion: the encoder is NOT trained mirror-equivariant
    (no hflip in its augmentations because handedness is meaningful), so we
    just swap embeddings between slots — the mirrored "same-shape" assumption
    holds for symmetric two-handed signs (most ASL) and is mildly wrong for
    one-handed signs. Acceptable noise.

    Negation rule: ONLY negate x-components within the kpt portions of each
    slot and the pose portion. Never negate the embedding values.
    """
    F = traj_TF.shape[1]
    out = traj_TF.copy()
    if F == FEATURES_PER_FRAME:
        slot_dims = HAND_SLOT_DIMS
        pose_base = POSE_BASE
        # 100D: every even index is an x-component (kpts + pose are interleaved x,y)
        out[:, 0::2] = -out[:, 0::2]
    elif F == FEATURES_PER_FRAME_V2:
        slot_dims = HAND_SLOT_DIMS_V2
        pose_base = POSE_BASE_V2
        # 108D: per slot negate x of handshape [base:base+42:2], location
        # [base+42], orientation [base+44]; pose x [92:108:2]. Then swap slots.
        for slot in (0, 1):
            base = slot * slot_dims
            out[:, base:base + NUM_HAND_KP * 2:2] = -out[:, base:base + NUM_HAND_KP * 2:2]
            out[:, base + 42] = -out[:, base + 42]
            out[:, base + 44] = -out[:, base + 44]
        out[:, pose_base:pose_base + NUM_POSE_KP * 2:2] = -out[:, pose_base:pose_base + NUM_POSE_KP * 2:2]
    elif F == FEATURES_PER_FRAME_WITH_EMBED:
        slot_dims = HAND_SLOT_DIMS_WITH_EMBED
        pose_base = POSE_BASE_WITH_EMBED
        # 356D: negate x only within the kpt sub-block of each slot + pose block.
        # Slot layout: [0..42)=kpts(x,y interleaved), [42..170)=embedding (no negate)
        for slot in (0, 1):
            base = slot * slot_dims
            out[:, base:base + NUM_HAND_KP * 2:2] = -out[:, base:base + NUM_HAND_KP * 2:2]
        # Pose: 16D, x,y interleaved → negate every other
        out[:, pose_base:pose_base + NUM_POSE_KP * 2:2] = -out[:, pose_base:pose_base + NUM_POSE_KP * 2:2]
    else:
        raise ValueError(f"_mirror_features: unsupported feature dim {F}; expected "
                         f"{FEATURES_PER_FRAME}, {FEATURES_PER_FRAME_V2} or {FEATURES_PER_FRAME_WITH_EMBED}")
    # Swap slot 0 ↔ slot 1 (full block including embedding when present)
    slot0 = out[:, 0:slot_dims].copy()
    out[:, 0:slot_dims] = out[:, slot_dims:2 * slot_dims]
    out[:, slot_dims:2 * slot_dims] = slot0
    return out


def _resample_trajectory(traj: np.ndarray, T: int) -> np.ndarray:
    """Linear time-resample a (n, F) trajectory to (T, F)."""
    n, F = traj.shape
    if n == T:
        return traj
    if n < 2:
        return np.tile(traj, (T, 1)).reshape(T, F)
    src_t = np.linspace(0.0, 1.0, n)
    dst_t = np.linspace(0.0, 1.0, T)
    out = np.zeros((T, F), dtype=np.float32)
    for f in range(F):
        # np.interp doesn't natively handle NaN; mask then interp on valid
        col = traj[:, f]
        valid = ~np.isnan(col)
        if not valid.any():
            out[:, f] = np.nan
        elif valid.all():
            out[:, f] = np.interp(dst_t, src_t, col)
        else:
            out[:, f] = np.interp(dst_t, src_t[valid], col[valid])
    return out


def _trajectory_from_file(path: Path) -> tuple[np.ndarray, float] | None:
    """Returns (trajectory, frac_frames_with_second_hand) or None."""
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    frames = data.get("frames") or []
    if not frames:
        return None
    rows = []
    slot1_count = 0
    for f in frames:
        feats, used_slot1 = _frame_to_features(f)
        rows.append(feats)
        if used_slot1:
            slot1_count += 1
    return np.stack(rows, axis=0), slot1_count / len(frames)


def _clip_distance(a: np.ndarray, b: np.ndarray) -> float:
    """NaN-tolerant per-element MSE between two (T, F) clips. Used as the
    intra-sign distance for k-medoids clustering.
    """
    diff = a - b
    valid = ~np.isnan(diff)
    if not valid.any():
        return float("inf")
    diff = np.where(valid, diff, 0.0)
    return float((diff ** 2).sum() / valid.sum())


def _kmedoids_cluster(arr: np.ndarray, k: int, max_iter: int = 25,
                       seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Cluster (n, T, F) clips into k groups via k-medoids on MSE distance.

    Returns (medoid_indices [k], assignments [n]). For low n or k=1 returns
    a single-cluster trivial answer. Used by fit_template when k > 1 to
    pull out modally-distinct sub-templates (e.g. "WHERE" left-vs-right
    head turn variants) that get washed out by a single pooled Gaussian.
    """
    n = arr.shape[0]
    if k <= 1 or n <= k:
        return np.arange(min(k, n)), np.zeros(n, dtype=np.int64)

    # Pairwise distance matrix — O(n^2 T F), tolerable for n ≤ a few hundred.
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = _clip_distance(arr[i], arr[j])
            D[i, j] = D[j, i] = d

    rng = np.random.default_rng(seed)
    # k-means++ style seeding on the precomputed distance matrix
    medoids = [int(rng.integers(n))]
    while len(medoids) < k:
        min_d = D[:, medoids].min(axis=1)
        # square distances bias toward far points; mask already-chosen
        probs = min_d ** 2
        for m in medoids:
            probs[m] = 0.0
        s = probs.sum()
        if s <= 0:
            # all remaining points are duplicates of medoids; just pick one
            remaining = [i for i in range(n) if i not in medoids]
            medoids.append(int(remaining[0]))
            continue
        probs = probs / s
        medoids.append(int(rng.choice(n, p=probs)))

    medoids_arr = np.array(medoids, dtype=np.int64)
    assignments = np.argmin(D[:, medoids_arr], axis=1)

    for _ in range(max_iter):
        new_medoids = medoids_arr.copy()
        for c in range(k):
            members = np.where(assignments == c)[0]
            if len(members) == 0:
                continue
            sub = D[np.ix_(members, members)]
            within_sum = sub.sum(axis=1)
            new_medoids[c] = members[int(np.argmin(within_sum))]
        new_assignments = np.argmin(D[:, new_medoids], axis=1)
        if np.array_equal(new_medoids, medoids_arr) and np.array_equal(
            new_assignments, assignments
        ):
            break
        medoids_arr = new_medoids
        assignments = new_assignments
    return medoids_arr, assignments


def fit_template(
    sign_id: str,
    trajectories_dir: Path,
    out_dir: Path,
    T: int = 32,
    min_clips: int = 5,
    dominant_hand_threshold: float = DEFAULT_DOMINANT_HAND_THRESHOLD,
    forced_one_handed: set[str] | None = None,
    k_medoids: int = 1,
    min_clips_per_cluster: int = 4,
) -> dict:
    sign_dir = trajectories_dir / sign_id
    if not sign_dir.exists():
        return {"sign_id": sign_id, "status": "no_dir"}
    json_files = sorted(sign_dir.glob("*.json"))
    if len(json_files) < min_clips:
        return {"sign_id": sign_id, "status": "too_few_clips", "n": len(json_files)}

    stacked: list[np.ndarray] = []
    per_clip_slot1_frac: list[float] = []
    for p in json_files:
        r = _trajectory_from_file(p)
        if r is None:
            continue
        traj, slot1_frac = r
        stacked.append(_resample_trajectory(traj, T))
        per_clip_slot1_frac.append(slot1_frac)
    if len(stacked) < min_clips:
        return {"sign_id": sign_id, "status": "too_few_usable", "n": len(stacked)}

    arr = np.stack(stacked, axis=0)  # (n_clips, T, F)
    mean = np.nanmean(arr, axis=0)
    var = np.nanvar(arr, axis=0)
    var = np.maximum(var, 0.001)  # floor in normalized-coord units

    # P2 #12: optional k-medoids multi-template. We only invoke if the sign
    # has enough clips to actually populate k clusters with > min_clips_per_cluster
    # members each; otherwise fall back silently to the pooled k=1 template.
    cluster_means: np.ndarray | None = None
    cluster_vars: np.ndarray | None = None
    cluster_assignments: np.ndarray | None = None
    effective_k = 1
    if k_medoids > 1 and arr.shape[0] >= k_medoids * min_clips_per_cluster:
        _, assignments = _kmedoids_cluster(arr, k_medoids)
        sizes = np.bincount(assignments, minlength=k_medoids)
        if sizes.min() >= min_clips_per_cluster:
            cluster_means_list = []
            cluster_vars_list = []
            for c in range(k_medoids):
                members = arr[assignments == c]
                cm = np.nanmean(members, axis=0)
                cv = np.maximum(np.nanvar(members, axis=0), 0.001)
                cluster_means_list.append(cm.astype(np.float32))
                cluster_vars_list.append(cv.astype(np.float32))
            cluster_means = np.stack(cluster_means_list, axis=0)  # (k, T, F)
            cluster_vars = np.stack(cluster_vars_list, axis=0)
            cluster_assignments = assignments.astype(np.int32)
            effective_k = k_medoids

    # Per-sign one-hand flag: motion-based.
    # Hand detector returns top-2 every frame, so slot1_frac is always
    # high (resting hand fills slot 1). The real signal is that a *resting*
    # slot-1 hand barely moves across the clip while slot-0 moves a lot.
    # Compare per-clip motion std of slot 0 vs slot 1; flag the sign as
    # one-handed when slot-1 motion is consistently < 30% of slot-0 motion.
    slot0_std = np.nanstd(arr[:, :, 0:HAND_SLOT_DIMS], axis=1).mean()  # (clips, F0).mean
    slot1_std = np.nanstd(arr[:, :, HAND_SLOT_DIMS:2 * HAND_SLOT_DIMS], axis=1).mean()
    motion_ratio = float(slot1_std / max(slot0_std, 1e-6))
    if np.isnan(motion_ratio):
        motion_ratio = 1.0
    clip_avg_slot1 = float(np.mean(per_clip_slot1_frac)) if per_clip_slot1_frac else 0.0
    # Manual override wins: hand-labeled one-handed signs always get the
    # mask. Auto-detect (motion ratio) is a fallback for unlabeled signs.
    if forced_one_handed and sign_id in forced_one_handed:
        dominant_hand_only = True
    else:
        dominant_hand_only = motion_ratio < dominant_hand_threshold

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sign_id}.npz"
    save_kwargs: dict = dict(
        mean=mean.astype(np.float32),
        var=var.astype(np.float32),
        clips=arr.astype(np.float32),
        n_clips=len(stacked),
        T=T,
        dominant_hand_only=bool(dominant_hand_only),
        slot1_frac=clip_avg_slot1,
        motion_ratio=motion_ratio,
        n_clusters=effective_k,
    )
    if cluster_means is not None:
        save_kwargs["cluster_means"] = cluster_means  # (k, T, F)
        save_kwargs["cluster_vars"] = cluster_vars
        save_kwargs["cluster_assignments"] = cluster_assignments
    np.savez_compressed(out_path, **save_kwargs)
    return {
        "sign_id": sign_id,
        "status": "ok",
        "n": len(stacked),
        "dominant_hand_only": bool(dominant_hand_only),
        "motion_ratio": round(motion_ratio, 3),
        "slot1_frac": round(clip_avg_slot1, 3),
        "n_clusters": effective_k,
        "out": str(out_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories-dir", type=Path, default=Path("data/trajectories"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/templates"))
    ap.add_argument("--vocab-json", type=Path, default=Path("dataset/slice1b_vocabulary.json"))
    ap.add_argument("--time-steps", type=int, default=32)
    ap.add_argument("--min-clips", type=int, default=5)
    ap.add_argument("--dominant-hand-threshold", type=float,
                    default=DEFAULT_DOMINANT_HAND_THRESHOLD)
    ap.add_argument("--one-handed-json", type=Path,
                    default=Path("dataset/one_handed_signs.json"),
                    help="JSON with {one_handed: [sign_id, ...]}; forces "
                         "dominant_hand_only=True for those signs.")
    ap.add_argument("--k-medoids", type=int, default=1,
                    help="P2 #12: cluster clips into k sub-templates per "
                         "sign. Falls back to k=1 for signs with too few "
                         "clips to populate all k clusters. Default 1.")
    args = ap.parse_args()

    vocab = json.loads(args.vocab_json.read_text())
    sign_ids = [s["sign_id"] for s in vocab["kept_signs"]]

    forced_one_handed: set[str] = set()
    if args.one_handed_json.exists():
        forced_one_handed = set(json.loads(args.one_handed_json.read_text())
                                .get("one_handed", []))
        print(f"loaded {len(forced_one_handed)} manually one-handed signs from "
              f"{args.one_handed_json}")

    results = []
    for sid in sign_ids:
        r = fit_template(sid, args.trajectories_dir, args.out_dir,
                         T=args.time_steps, min_clips=args.min_clips,
                         dominant_hand_threshold=args.dominant_hand_threshold,
                         forced_one_handed=forced_one_handed,
                         k_medoids=args.k_medoids)
        results.append(r)
        print(f"  {sid}: {r}")

    summary_path = args.out_dir / "_fit_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    ok = sum(1 for r in results if r["status"] == "ok")
    one_handed = sum(1 for r in results if r.get("dominant_hand_only"))
    print(f"\nfit {ok}/{len(results)} signs ({one_handed} one-handed); "
          f"summary at {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
