"""Sign matcher inference + per-sign threshold calibration.

This module contains:
  - load_templates(): load fitted templates produced by fit_templates.py.
  - mahalanobis_score(): per-sign similarity (lower = closer).
    Applies the per-sign dominant_hand_only mask and the mirror-min
    canonicalization that handles webcam mirroring without retraining.
  - predict(): top-1 sign over the loaded vocabulary; full-vocab confidence.
  - predict_in_slice(): same as predict() but softmax-renormalized over a
    closed N-sign confusion set ("Sign 3 of 8"-style lessons).
  - calibrate_thresholds(): per-sign pass threshold @ target precision.
    Saves per-sign positive-score quantile breakpoints into the output so
    the UX layer can map raw distances to 0–100 grades.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from training.detectors.fit_templates import (
    FEATURES_PER_FRAME,
    HAND_SLOT_DIMS,
    _frame_to_features,
    _mirror_features,
    _resample_trajectory,
)


NEAREST_CLIP_FALLBACK_THRESHOLD = 8  # n_clips below which we use nearest-clip
NEAREST_CLIP_ISOTROPIC_VAR = 0.05  # in normalized-coord units²
QUANTILE_BREAKPOINTS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

# DTW alignment — Sakoe-Chiba radius in time steps. Lets the learner sign
# faster or slower than the template; without it, frame index = time index
# and any speed mismatch turns into pure distance penalty.
DTW_RADIUS = 8
USE_DTW = True


# ---------------------------------------------------------------------------
# Template loading + scoring
# ---------------------------------------------------------------------------


def load_templates(templates_dir: Path) -> dict[str, dict]:
    """Returns {sign_id: {"mean": (T, F), "var": (T, F), "clips": (n, T, F),
    "n_clips": int, "T": int, "dominant_hand_only": bool}}.

    Backwards-compatible: older .npz files without `clips` or
    `dominant_hand_only` load with safe defaults.
    """
    out: dict[str, dict] = {}
    for p in sorted(templates_dir.glob("*.npz")):
        if p.stem.startswith("_"):
            continue
        data = np.load(p)
        entry = {
            "mean": data["mean"],
            "var": data["var"],
            "n_clips": int(data.get("n_clips", 0)),
            "T": int(data.get("T", data["mean"].shape[0])),
            "dominant_hand_only": bool(data["dominant_hand_only"])
                if "dominant_hand_only" in data.files else False,
            "clips": data["clips"] if "clips" in data.files else None,
            # P2 #12: optional multi-template clusters. When present, the
            # matcher scores each cluster's (mean, var) and takes the min.
            # `cluster_means` shape (k, T, F); `cluster_vars` same.
            "n_clusters": int(data.get("n_clusters", 1)),
            "cluster_means": data["cluster_means"] if "cluster_means" in data.files else None,
            "cluster_vars": data["cluster_vars"] if "cluster_vars" in data.files else None,
        }
        out[p.stem] = entry
    return out


def _apply_dominant_hand_mask(diff: np.ndarray, dominant_hand_only: bool) -> np.ndarray:
    """If a sign is one-handed, treat slot-1 hand features as missing (NaN)
    so they neither contribute to the distance nor inflate n_valid.
    """
    if not dominant_hand_only:
        return diff
    diff = diff.copy()
    diff[:, HAND_SLOT_DIMS:2 * HAND_SLOT_DIMS] = np.nan
    return diff


def _gaussian_score(traj_TF: np.ndarray, mean: np.ndarray, var: np.ndarray,
                    dominant_hand_only: bool) -> float:
    """Per-element diagonal Mahalanobis, averaged over visible features."""
    diff = traj_TF - mean
    diff = _apply_dominant_hand_mask(diff, dominant_hand_only)
    valid = ~np.isnan(diff)
    if not valid.any():
        return float("inf")
    diff = np.where(valid, diff, 0.0)
    sq = (diff ** 2) / var
    return float(sq.sum() / max(valid.sum(), 1))


def _per_frame_mahalanobis(a: np.ndarray, b: np.ndarray, var_row: np.ndarray,
                           dominant_hand_only: bool) -> float:
    """Diagonal Mahalanobis between two F-dim feature vectors, NaN-tolerant."""
    diff = a - b
    if dominant_hand_only:
        # Zero out slot-1 hand dims
        diff = diff.copy()
        diff[HAND_SLOT_DIMS:2 * HAND_SLOT_DIMS] = np.nan
    valid = ~np.isnan(diff)
    if not valid.any():
        return 1e9
    diff = np.where(valid, diff, 0.0)
    return float((diff ** 2 / var_row).sum() / max(valid.sum(), 1))


def _dtw_score(traj_TF: np.ndarray, mean_TF: np.ndarray, var_TF: np.ndarray,
               dominant_hand_only: bool, radius: int = DTW_RADIUS) -> float:
    """Sakoe-Chiba band-constrained DTW.

    Replaces frame-index alignment with optimal time warp. Cost per cell is
    the per-element diagonal Mahalanobis between the learner frame and the
    template's per-step Gaussian. Normalizes by warp-path length so the
    score scale matches `_gaussian_score`.

    With T=32 and radius=8 this is ~500 cells × O(F) ≈ 50k ops per call;
    well under 1 ms per template at inference.
    """
    T = traj_TF.shape[0]
    inf = 1e18
    # D[i][j] = best cost to align prefix traj[:i+1] with template[:j+1]
    D = np.full((T + 1, T + 1), inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, T + 1):
        j_min = max(1, i - radius)
        j_max = min(T, i + radius)
        for j in range(j_min, j_max + 1):
            cost = _per_frame_mahalanobis(
                traj_TF[i - 1], mean_TF[j - 1], var_TF[j - 1],
                dominant_hand_only)
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    # Normalize by path length (T diagonal steps minimum)
    return float(D[T, T] / (2 * T))


def _nearest_clip_score(traj_TF: np.ndarray, clips: np.ndarray,
                        dominant_hand_only: bool) -> float:
    """min L2 vs each raw clip, isotropic variance. Used for low-n signs."""
    best = float("inf")
    for clip in clips:
        diff = traj_TF - clip
        diff = _apply_dominant_hand_mask(diff, dominant_hand_only)
        valid = ~np.isnan(diff)
        if not valid.any():
            continue
        diff = np.where(valid, diff, 0.0)
        s = float((diff ** 2).sum() / NEAREST_CLIP_ISOTROPIC_VAR
                  / max(valid.sum(), 1))
        if s < best:
            best = s
    return best


def mahalanobis_score(traj_TF: np.ndarray, template: dict) -> float:
    """Per-sign similarity. Lower = closer match.

    Side-effects: scores the mirrored trajectory too and returns the min,
    so webcam-mirror handedness flips do not penalize correct signs.
    For low-n signs (n_clips < NEAREST_CLIP_FALLBACK_THRESHOLD), falls back
    to nearest-raw-clip distance instead of pooled Gaussian.
    """
    mean = template["mean"]
    var = template["var"]  # variance-floored at 0.001 in fit_templates.py
    dho = template.get("dominant_hand_only", False)
    use_nc = (template.get("clips") is not None
              and template.get("n_clips", 0) > 0
              and template["n_clips"] < NEAREST_CLIP_FALLBACK_THRESHOLD)

    if use_nc:
        clips = template["clips"]
        s1 = _nearest_clip_score(traj_TF, clips, dho)
        s2 = _nearest_clip_score(_mirror_features(traj_TF), clips, dho)
        return min(s1, s2)

    # P2 #12: if this template has clusters, score each and take min.
    # Falls through to single-template path when n_clusters <= 1.
    n_clusters = int(template.get("n_clusters", 1) or 1)
    cluster_means = template.get("cluster_means")
    cluster_vars = template.get("cluster_vars")
    mirror = _mirror_features(traj_TF)

    if n_clusters > 1 and cluster_means is not None and cluster_vars is not None:
        best = float("inf")
        for c in range(n_clusters):
            cm = cluster_means[c]
            cv = cluster_vars[c]
            if USE_DTW:
                s = min(_dtw_score(traj_TF, cm, cv, dho),
                        _dtw_score(mirror, cm, cv, dho))
            else:
                s = min(_gaussian_score(traj_TF, cm, cv, dho),
                        _gaussian_score(mirror, cm, cv, dho))
            if s < best:
                best = s
        return best

    if USE_DTW:
        s1 = _dtw_score(traj_TF, mean, var, dho)
        s2 = _dtw_score(mirror, mean, var, dho)
        return min(s1, s2)

    s1 = _gaussian_score(traj_TF, mean, var, dho)
    s2 = _gaussian_score(mirror, mean, var, dho)
    return min(s1, s2)


def drop_handless_frames(frames: list[dict]) -> list[dict]:
    """Keep only frames with at least one detected hand.

    Per-frame inspection (scripts/inspect_recall_failures.py) showed ~40% of
    raw-clip frames are idle lead-in / lead-out (hands down or off-screen) where
    the detector correctly reports no hand. Resampling those into the fixed-T
    trajectory wastes ~40% of the time steps on hands-down padding and dilutes
    the actual sign. Dropping them concentrates all T steps on real signing.

    Falls back to the original list if fewer than 2 hand-bearing frames remain
    (never returns an unusably short trajectory).
    """
    kept = [f for f in frames if f.get("hands")]
    return kept if len(kept) >= 2 else frames


def trajectory_from_frames(frames: list[dict], T: int) -> np.ndarray:
    """frames: list of frame dicts as written by extract_trajectories_v2.py.
    Auto-detects 100D (kpts-only) vs 356D (kpts + 128D handshape embedding)
    based on whether any hand record has an "embedding" field.
    Idle (handless) frames are dropped first so the T steps cover actual
    signing. Returns (T, F) float32 with NaN where missing.
    """
    from training.detectors.fit_templates import trajectory_has_embedding
    frames = drop_handless_frames(frames)
    with_embed = trajectory_has_embedding(frames)
    rows = np.stack([_frame_to_features(f, with_embedding=with_embed)[0]
                     for f in frames], axis=0)
    return _resample_trajectory(rows, T)


def predict(
    traj_TF: np.ndarray,
    templates: dict[str, dict],
) -> tuple[str, float, dict[str, float]]:
    """Return (best_sign, full_vocab_confidence, per_sign_score_dict)."""
    scores = {sid: mahalanobis_score(traj_TF, t) for sid, t in templates.items()}
    sids = list(scores.keys())
    raw = -np.array([scores[s] for s in sids], dtype=np.float64)
    raw = raw - raw.max()
    p = np.exp(raw)
    p = p / p.sum()
    best_idx = int(np.argmax(p))
    return sids[best_idx], float(p[best_idx]), scores


def predict_in_slice(
    traj_TF: np.ndarray,
    templates: dict[str, dict],
    slice_sign_ids: list[str],
) -> tuple[str, float, dict[str, float]]:
    """Top-1 + confidence restricted to a closed N-sign lesson slice.

    The deployed UX is "Sign 3 of 8" — the realistic confusion set is the
    slice, not the full 75-sign vocab. Computing softmax over the slice
    only avoids artificially low confidences from far-away vocab signs.
    """
    sub = {s: templates[s] for s in slice_sign_ids if s in templates}
    if not sub:
        return "", 0.0, {}
    return predict(traj_TF, sub)


# ---------------------------------------------------------------------------
# Streaming inference (P2 #13) — rolling-window predictor for live UX
# ---------------------------------------------------------------------------


class StreamingMatcher:
    """Rolling-window matcher for live webcam inference.

    Lifecycle:
      m = StreamingMatcher(templates, thresholds, time_steps=32, window_frames=48)
      for frame_dict in webcam_stream:
          state = m.push(frame_dict, target_sign_id="coat")
          if state.unlocked:
              # threshold-pass for target hit `consecutive_required` windows
              ...

    Cost per push():
      * 1 _frame_to_features call (~µs)
      * 1 resample to T frames
      * len(slice) × 2 × DTW(T=32, radius=8) ≈ 1k cells × F each
        With a 3-sign slice this is ~3 ms in NumPy. Fine for 30 FPS.

    The class is intentionally narrow — same signatures we want to mirror
    in the TS port (see HANDOFF.md Day-2). Don't move logic into it that
    isn't also cheap to do in JS.
    """

    def __init__(
        self,
        templates: dict[str, dict],
        thresholds: dict[str, dict] | None = None,
        time_steps: int = 32,
        window_frames: int = 48,
        consecutive_required: int = 3,
    ) -> None:
        self.templates = templates
        self.thresholds = thresholds or {}
        self.T = time_steps
        # Raw-frame ring buffer. window_frames > T lets us absorb camera
        # jitter (dropped frames, irregular dt) and still resample to T.
        # Default 48 ≈ 1.6 s at 30 FPS, which is the typical sign length.
        self.window = max(window_frames, time_steps)
        self.consecutive_required = max(consecutive_required, 1)
        self._frames: deque[np.ndarray] = deque(maxlen=self.window)
        # Per-sign pass streak: count of trailing windows whose score for
        # that sign was at or below threshold. Resets to 0 on first miss.
        self._streaks: dict[str, int] = defaultdict(int)
        self._last_scores: dict[str, float] = {}

    # -- frame ingestion ----------------------------------------------------

    def push_features(self, feat_F: np.ndarray) -> None:
        """Append a pre-computed (F,) feature vector. Use when the caller
        already ran _frame_to_features (e.g. TS port doing it in WebGPU).

        Skips all-NaN (handless) frames so the live window matches the
        idle-trimmed trajectories used in training (drop_handless_frames).
        Mirror this skip in the TS port to avoid train/inference skew.
        """
        if not np.isfinite(feat_F).any():
            return
        self._frames.append(feat_F)

    def push_frame(self, frame: dict) -> None:
        """Append a raw frame dict (as written by extract_trajectories).
        Handless (idle) frames are skipped so the live window matches the
        idle-trimmed trajectories used in training (drop_handless_frames)."""
        if not frame.get("hands"):
            return
        feat, _ = _frame_to_features(frame)
        self._frames.append(feat)

    def reset(self) -> None:
        self._frames.clear()
        self._streaks.clear()
        self._last_scores.clear()

    def ready(self) -> bool:
        """True once enough frames have accumulated to score a window."""
        return len(self._frames) >= self.T

    # -- prediction ---------------------------------------------------------

    def current_trajectory(self) -> np.ndarray | None:
        """Return the buffered frames resampled to (T, F), or None if not
        yet enough frames have been pushed.
        """
        if not self.ready():
            return None
        arr = np.stack(list(self._frames), axis=0)
        return _resample_trajectory(arr, self.T)

    def score_slice(self, slice_sign_ids: list[str]) -> dict[str, float] | None:
        """Score the current window against each sign in `slice_sign_ids`.
        Returns None if buffer isn't full yet.
        """
        traj = self.current_trajectory()
        if traj is None:
            return None
        scores = {}
        for sid in slice_sign_ids:
            t = self.templates.get(sid)
            if t is None:
                continue
            scores[sid] = mahalanobis_score(traj, t)
        self._last_scores = scores
        return scores

    def step(
        self,
        target_sign_id: str,
        slice_sign_ids: list[str] | None = None,
    ) -> "StreamingStep":
        """Run one streaming step: score the slice, advance per-sign streaks,
        and report whether `target_sign_id` has hit the unlock criterion.

        `slice_sign_ids` defaults to [target_sign_id]. For lesson UIs that
        want to show top-k confusers, pass a 3–8 sign slice; streaks are
        still tracked per-sign so multiple unlocks could in principle fire.
        """
        if slice_sign_ids is None:
            slice_sign_ids = [target_sign_id]

        scores = self.score_slice(slice_sign_ids)
        if scores is None:
            return StreamingStep(
                ready=False, scores={}, target_passed_now=False,
                target_streak=self._streaks.get(target_sign_id, 0),
                unlocked=False, top1=None, top1_score=None,
            )

        for sid, sc in scores.items():
            thr = self._threshold_for(sid)
            if sc <= thr:
                self._streaks[sid] += 1
            else:
                self._streaks[sid] = 0

        top1_sid = min(scores, key=lambda s: scores[s])
        target_score = scores.get(target_sign_id, float("inf"))
        target_thr = self._threshold_for(target_sign_id)
        target_passed = target_score <= target_thr
        target_streak = self._streaks.get(target_sign_id, 0)
        unlocked = target_streak >= self.consecutive_required

        return StreamingStep(
            ready=True,
            scores=scores,
            target_passed_now=target_passed,
            target_streak=target_streak,
            unlocked=unlocked,
            top1=top1_sid,
            top1_score=scores[top1_sid],
        )

    def _threshold_for(self, sign_id: str) -> float:
        entry = self.thresholds.get(sign_id)
        if not entry:
            return float("inf")
        thr = entry.get("threshold")
        if thr is None or not np.isfinite(thr):
            return float("inf")
        return float(thr)


class StreamingStep:
    """Per-frame result of StreamingMatcher.step().

    Plain attribute container — designed to map 1:1 to a TS interface in
    the frontend port.
    """

    __slots__ = (
        "ready", "scores", "target_passed_now", "target_streak",
        "unlocked", "top1", "top1_score",
    )

    def __init__(
        self,
        ready: bool,
        scores: dict[str, float],
        target_passed_now: bool,
        target_streak: int,
        unlocked: bool,
        top1: str | None,
        top1_score: float | None,
    ) -> None:
        self.ready = ready
        self.scores = scores
        self.target_passed_now = target_passed_now
        self.target_streak = target_streak
        self.unlocked = unlocked
        self.top1 = top1
        self.top1_score = top1_score

    def __repr__(self) -> str:
        return (
            f"StreamingStep(ready={self.ready}, top1={self.top1!r}, "
            f"top1_score={self.top1_score!r}, target_streak={self.target_streak}, "
            f"unlocked={self.unlocked})"
        )


# ---------------------------------------------------------------------------
# Per-sign threshold calibration (Phase 4 Slice 4.3)
# ---------------------------------------------------------------------------


def _score_quantiles(scores: list[float]) -> dict[str, float]:
    """Percentile breakpoints over a list of positive scores. Used by the
    UX layer to map a learner's raw distance to a 0–100 grade per sign.
    """
    if not scores:
        return {}
    arr = np.array(scores, dtype=np.float64)
    out = {}
    for q in QUANTILE_BREAKPOINTS:
        out[f"q{int(q * 100):02d}"] = float(np.quantile(arr, q))
    return out


def calibrate_thresholds(
    templates: dict[str, dict],
    val_trajectories_dir: Path,
    target_precision: float = 0.95,
) -> dict[str, dict]:
    """Build per-sign positive/negative score distributions over the
    validation set and pick a pass threshold @ target precision. Also
    records per-sign positive-score quantiles for UX-grade mapping.
    """
    val_traj_by_sign: dict[str, list[np.ndarray]] = defaultdict(list)
    for sign_dir in sorted(val_trajectories_dir.glob("*")):
        if not sign_dir.is_dir():
            continue
        sign = sign_dir.name
        T = templates.get(sign, {}).get("T", 32)
        for jp in sorted(sign_dir.glob("*.json")):
            try:
                data = json.loads(jp.read_text())
                traj = trajectory_from_frames(data["frames"], T)
                val_traj_by_sign[sign].append(traj)
            except Exception:
                continue

    out: dict[str, dict] = {}
    all_signs = list(templates.keys())
    for sign in all_signs:
        template = templates[sign]
        positives = [mahalanobis_score(t, template) for t in val_traj_by_sign.get(sign, [])]

        negatives: list[float] = []
        for other_sign in all_signs:
            if other_sign == sign:
                continue
            for t in val_traj_by_sign.get(other_sign, [])[:5]:
                negatives.append(mahalanobis_score(t, template))

        if not positives or not negatives:
            out[sign] = {
                "threshold": float("inf"),
                "achieved_precision": 0.0,
                "achieved_recall": 0.0,
                "n_positives": len(positives),
                "n_negatives": len(negatives),
                "positive_quantiles": _score_quantiles(positives),
                "notes": "insufficient val data",
            }
            continue

        cand = sorted(set(positives + negatives))
        chosen = None
        chosen_metrics = None
        for thr in cand:
            tp = sum(1 for s in positives if s <= thr)
            fp = sum(1 for s in negatives if s <= thr)
            if tp == 0:
                continue
            prec = tp / (tp + fp)
            rec = tp / max(len(positives), 1)
            if prec >= target_precision:
                chosen = thr
                chosen_metrics = (prec, rec)
                break

        if chosen is None:
            best_prec = 0.0
            best_thr = float("inf")
            best_rec = 0.0
            for thr in cand:
                tp = sum(1 for s in positives if s <= thr)
                fp = sum(1 for s in negatives if s <= thr)
                if tp == 0:
                    continue
                prec = tp / (tp + fp)
                if prec > best_prec:
                    best_prec = prec
                    best_thr = thr
                    best_rec = tp / max(len(positives), 1)
            chosen = best_thr
            chosen_metrics = (best_prec, best_rec)

        out[sign] = {
            "threshold": float(chosen),
            "achieved_precision": float(chosen_metrics[0]),
            "achieved_recall": float(chosen_metrics[1]),
            "n_positives": len(positives),
            "n_negatives": len(negatives),
            "positive_quantiles": _score_quantiles(positives),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates-dir", type=Path, default=Path("data/templates"))
    ap.add_argument("--val-trajectories-dir", type=Path, default=Path("data/trajectories/val"))
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument("--out", type=Path, default=Path("data/templates/_thresholds.json"))
    args = ap.parse_args()

    templates = load_templates(args.templates_dir)
    print(f"loaded {len(templates)} templates")
    n_one_handed = sum(1 for t in templates.values() if t.get("dominant_hand_only"))
    n_low_clip = sum(1 for t in templates.values()
                     if 0 < t.get("n_clips", 0) < NEAREST_CLIP_FALLBACK_THRESHOLD)
    print(f"  one-handed: {n_one_handed}; nearest-clip fallback: {n_low_clip}")

    result = calibrate_thresholds(
        templates, args.val_trajectories_dir, target_precision=args.target_precision
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")

    mean_p = float(np.mean([r["achieved_precision"] for r in result.values()]))
    mean_r = float(np.mean([r["achieved_recall"] for r in result.values()]))
    hit95 = sum(1 for r in result.values() if r["achieved_precision"] >= 0.95)
    print(f"mean P={mean_p:.3f} R={mean_r:.3f}; {hit95}/{len(result)} signs hit 0.95 precision")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
