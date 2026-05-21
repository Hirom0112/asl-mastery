"""Sign matcher inference + per-sign threshold calibration.

This module contains:
  - load_templates(): load fitted templates produced by fit_templates.py
  - mahalanobis_score(): Mahalanobis-distance similarity score, learner
    trajectory vs one sign's template. Returns a single positive scalar
    (lower = closer to template).
  - predict(): per-frame inference helper — best sign + confidence.
  - calibrate_thresholds(): Phase 4 Slice 4.3 — build positive/negative
    distributions per sign over the validation corpus, pick a per-sign
    pass threshold that prioritizes precision (false-pass is worse than
    false-fail in a learning context, per docs/EVAL_GATE.md).

The matcher consumes the same flattened 100-d feature vectors that
fit_templates.py emits (so resampling + anchoring already happened
during template fitting and must happen again at inference).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from training.detectors.fit_templates import (
    FEATURES_PER_FRAME,
    _frame_to_features,
    _resample_trajectory,
)


# ---------------------------------------------------------------------------
# Template loading + scoring
# ---------------------------------------------------------------------------


def load_templates(templates_dir: Path) -> dict[str, dict]:
    """Returns {sign_id: {"mean": (T, F), "var": (T, F), "n_clips": int, "T": int}}."""
    out = {}
    for p in sorted(templates_dir.glob("*.npz")):
        if p.stem.startswith("_"):
            continue
        data = np.load(p)
        out[p.stem] = {
            "mean": data["mean"],
            "var": data["var"],
            "n_clips": int(data.get("n_clips", 0)),
            "T": int(data.get("T", data["mean"].shape[0])),
        }
    return out


def mahalanobis_score(traj_TF: np.ndarray, template: dict) -> float:
    """Per-element diagonal Mahalanobis distance, averaged over visible
    (non-NaN) features. Lower = closer match to template.

    Both traj_TF and template["mean"] must have the same (T, F) shape;
    callers are responsible for resampling.
    """
    mean = template["mean"]
    var = template["var"]  # variance-floored at 1.0 in fit_templates.py
    diff = traj_TF - mean
    # Where the learner had NaN (missing keypoint), skip
    valid = ~np.isnan(diff)
    diff = np.where(valid, diff, 0.0)
    sq_mahal = (diff ** 2) / var
    n_valid = max(valid.sum(), 1)
    return float(sq_mahal.sum() / n_valid)


def trajectory_from_frames(frames: list[dict], T: int) -> np.ndarray:
    """frames: list of frame dicts as written by extract_trajectories.py.
    Returns (T, FEATURES_PER_FRAME) float32 with NaN where missing.
    """
    rows = np.stack([_frame_to_features(f) for f in frames], axis=0)
    return _resample_trajectory(rows, T)


def predict(
    traj_TF: np.ndarray,
    templates: dict[str, dict],
) -> tuple[str, float, dict[str, float]]:
    """Return (best_sign, confidence_softmax, per_sign_score_dict).

    Confidence is the softmax probability of the best sign under a
    softmax-over-negated-distances. Higher = more confident.
    """
    scores = {sid: mahalanobis_score(traj_TF, t) for sid, t in templates.items()}
    # Softmax over negative scores (closer = higher prob)
    sids = list(scores.keys())
    raw = -np.array([scores[s] for s in sids], dtype=np.float64)
    # Numerical stability
    raw = raw - raw.max()
    p = np.exp(raw)
    p = p / p.sum()
    best_idx = int(np.argmax(p))
    return sids[best_idx], float(p[best_idx]), scores


# ---------------------------------------------------------------------------
# Per-sign threshold calibration (Phase 4 Slice 4.3)
# ---------------------------------------------------------------------------


def calibrate_thresholds(
    templates: dict[str, dict],
    val_trajectories_dir: Path,
    target_precision: float = 0.95,
) -> dict[str, dict]:
    """For each sign, build:
      - positives: scores of validation clips of THIS sign vs THIS sign's template
      - negatives: scores of validation clips of OTHER signs vs THIS sign's template
    Pick the threshold that yields target_precision on positives ∪ negatives.

    Returns {sign_id: {
        threshold: float,             # pass if mahalanobis_score <= threshold
        positives: [scores...],
        negatives_sampled: [scores...],
        achieved_precision: float,
        achieved_recall: float,
        n_positives: int,
        n_negatives: int,
    }}.
    """
    # Pre-load per-sign trajectories from the validation directory.
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

        # Negatives: sample up to 5 clips per OTHER sign
        negatives = []
        for other_sign in all_signs:
            if other_sign == sign:
                continue
            other_trajs = val_traj_by_sign.get(other_sign, [])[:5]
            negatives.extend(mahalanobis_score(t, template) for t in other_trajs)

        if not positives or not negatives:
            out[sign] = {
                "threshold": float("inf"),
                "positives": positives,
                "negatives_sampled": negatives,
                "achieved_precision": 0.0,
                "achieved_recall": 0.0,
                "n_positives": len(positives),
                "n_negatives": len(negatives),
                "notes": "insufficient val data",
            }
            continue

        # Sweep threshold over candidate values (sorted positives), pick
        # smallest threshold that achieves target_precision.
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
            # Fall back to highest-precision threshold reachable
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
            "positives": positives,
            "negatives_sampled": negatives,
            "achieved_precision": float(chosen_metrics[0]),
            "achieved_recall": float(chosen_metrics[1]),
            "n_positives": len(positives),
            "n_negatives": len(negatives),
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
    result = calibrate_thresholds(
        templates, args.val_trajectories_dir, target_precision=args.target_precision
    )
    # Strip full score lists from the saved file to keep it small
    summary = {
        sign: {
            "threshold": r["threshold"],
            "achieved_precision": r["achieved_precision"],
            "achieved_recall": r["achieved_recall"],
            "n_positives": r["n_positives"],
            "n_negatives": r["n_negatives"],
        }
        for sign, r in result.items()
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
