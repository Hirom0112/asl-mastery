"""Diagnostic analysis of a trained sign classifier.

Loads a classifier ckpt + its val trajectories, runs inference, produces:
  - Overall top-1 / top-3
  - Per-sign top-1 sorted by improvement vs baseline
  - Confusion matrix top entries (which sign is most-often-mistaken-for-which)
  - Per-source breakdown (sem_lex vs asl_citizen vs msasl vs lifeprint vs ytsearch)
  - Calibration: per-sign magnet ratio (predicted/actual count) — detects collapse

Usage:
    training/.venv/bin/python -m scripts.analyze_classifier \\
        --classifier-ckpt artifacts/ckpts/sign_classifier_v8_356d_tcn.pt \\
        --trajectories-dir data/trajectories_v8_pull \\
        --val-frac 0.20 --seed 42 \\
        --baseline-per-class artifacts/runs/sign_classifier_v0_per_class.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from training.detectors.sign_classifier import (
    SignClassifier, TransformerSignClassifier, count_parameters,
)
from training.detectors.sign_matcher import trajectory_from_frames


TIME_STEPS = 32


def _resample_to_T(feats: np.ndarray, T: int) -> np.ndarray:
    from training.detectors.fit_templates import _resample_trajectory
    return _resample_trajectory(feats, T)


def _split_clips(traj_root: Path, val_frac: float, seed: int) -> tuple[list[Path], list[Path]]:
    rnd = random.Random(seed)
    train, val = [], []
    for sign_dir in sorted(traj_root.iterdir()):
        if not sign_dir.is_dir():
            continue
        files = sorted(sign_dir.glob("*.json"))
        rnd.shuffle(files)
        n_val = max(1, int(len(files) * val_frac)) if len(files) >= 2 else 0
        val.extend(files[:n_val])
        train.extend(files[n_val:])
    return train, val


def _infer_arch_from_state_dict(state_dict: dict) -> str:
    """Return 'transformer' if the ckpt was trained as a transformer, else 'tcn'."""
    # Transformer has 'cls_token' and 'pos_embed' params
    if any("cls_token" in k for k in state_dict.keys()):
        return "transformer"
    return "tcn"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier-ckpt", type=Path, required=True)
    ap.add_argument("--trajectories-dir", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baseline-per-class", type=Path, default=None,
                    help="Optional per_class.json from an earlier run to diff against.")
    ap.add_argument("--top-confusions", type=int, default=15,
                    help="How many most-confused (true, predicted) pairs to surface.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    ckpt = torch.load(args.classifier_ckpt, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"]
    classes = ckpt["classes_in_use"]
    n_classes = len(classes)
    sign_to_idx = {s: i for i, s in enumerate(classes)}
    print(f"classes ({n_classes}): {classes[:8]}... {classes[-3:]}")

    arch = _infer_arch_from_state_dict(state_dict)
    print(f"arch detected from state_dict: {arch}")

    # Auto-detect feature dim from the first val trajectory
    _, val_paths = _split_clips(args.trajectories_dir, args.val_frac, args.seed)
    if not val_paths:
        raise SystemExit("no val trajectories found")
    sample = json.loads(val_paths[0].read_text())
    sample_feats = trajectory_from_frames(sample["frames"], T=TIME_STEPS)
    feat_dim = int(sample_feats.shape[1])
    print(f"detected feature dim: {feat_dim} ({'with embed (Phase 4.6)' if feat_dim == 356 else 'kpts-only baseline'})")

    # Build model with detected dim + arch
    if arch == "transformer":
        # Infer transformer config from state_dict shapes
        d_model = state_dict["input_proj.weight"].shape[0]
        n_layers = sum(1 for k in state_dict if k.startswith("encoder.layers.") and ".self_attn.in_proj_weight" in k)
        # nhead is harder to infer; try 4 and 8
        try:
            model = TransformerSignClassifier(num_features=feat_dim, num_classes=n_classes,
                                              d_model=d_model, num_layers=n_layers, nhead=8)
            model.load_state_dict(state_dict)
        except Exception:
            model = TransformerSignClassifier(num_features=feat_dim, num_classes=n_classes,
                                              d_model=d_model, num_layers=n_layers, nhead=4)
            model.load_state_dict(state_dict)
    else:
        model = SignClassifier(num_features=feat_dim, num_classes=n_classes,
                               hidden=256, num_blocks=5)
        model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print(f"classifier params: {count_parameters(model):,}")

    # Score every val trajectory
    print(f"\nscoring {len(val_paths)} val trajectories…")
    per_class_correct = np.zeros(n_classes, dtype=np.int64)
    per_class_total = np.zeros(n_classes, dtype=np.int64)
    per_class_predicted = np.zeros(n_classes, dtype=np.int64)
    confusion: dict[tuple[int, int], int] = defaultdict(int)
    per_source_correct: dict[str, int] = defaultdict(int)
    per_source_total: dict[str, int] = defaultdict(int)
    top1_total = top3_total = total = 0

    batch_feats = []
    batch_labels = []
    batch_sources = []

    def _flush():
        nonlocal top1_total, top3_total, total
        if not batch_feats:
            return
        x = torch.from_numpy(np.stack(batch_feats, 0)).float().to(device)
        with torch.no_grad():
            logits = model(x)
        top3 = logits.topk(3, dim=1).indices.cpu().numpy()
        preds = top3[:, 0]
        for pred, top3_row, label, src in zip(preds, top3, batch_labels, batch_sources):
            per_class_total[label] += 1
            per_class_predicted[pred] += 1
            per_source_total[src] += 1
            if pred == label:
                per_class_correct[label] += 1
                per_source_correct[src] += 1
                top1_total += 1
            else:
                confusion[(int(label), int(pred))] += 1
            if label in top3_row:
                top3_total += 1
            total += 1
        batch_feats.clear()
        batch_labels.clear()
        batch_sources.clear()

    BATCH = 128
    for jp in val_paths:
        try:
            traj = json.loads(jp.read_text())
        except Exception:
            continue
        sign = traj.get("sign_id")
        if sign not in sign_to_idx:
            continue
        source = traj.get("source", "unknown")
        # source isn't always in trajectory; infer from clip_path if missing
        if source == "unknown" and traj.get("clip_path"):
            cp = traj["clip_path"]
            if "sem_lex" in cp: source = "sem_lex"
            elif "asl_citizen" in cp: source = "asl_citizen"
            elif "msasl" in cp: source = "msasl"
            elif "lifeprint" in cp: source = "lifeprint"
            elif "ytsearch" in cp: source = "ytsearch"
            elif "v2-yt" in cp or "asl_clips" in cp: source = "project_clean"
            elif "wlasl" in cp: source = "wlasl"
        feats = trajectory_from_frames(traj["frames"], T=TIME_STEPS)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        # Trajectories with NO detected hands have 100D feats; upcast to 356D
        # by inserting 128 zeros for each slot's embedding portion (preserves
        # the same layout train_classifier uses).
        if feats.shape[1] == 100 and feat_dim == 356:
            T_, _ = feats.shape
            up = np.zeros((T_, 356), dtype=np.float32)
            up[:, 0:42] = feats[:, 0:42]                # slot 0 kpts
            # up[:, 42:170] = 0                          # slot 0 embed (missing)
            up[:, 170:212] = feats[:, 42:84]             # slot 1 kpts
            # up[:, 212:340] = 0                          # slot 1 embed (missing)
            up[:, 340:356] = feats[:, 84:100]            # pose
            feats = up
        batch_feats.append(feats.astype(np.float32))
        batch_labels.append(sign_to_idx[sign])
        batch_sources.append(source)
        if len(batch_feats) >= BATCH:
            _flush()
    _flush()

    overall_top1 = top1_total / max(total, 1)
    overall_top3 = top3_total / max(total, 1)
    print(f"\n=== OVERALL ===  N={total}  top-1={overall_top1 * 100:.2f}%  top-3={overall_top3 * 100:.2f}%")

    # Per-source breakdown
    print(f"\n=== PER SOURCE ===")
    for src in sorted(per_source_total, key=per_source_total.get, reverse=True):
        n = per_source_total[src]
        c = per_source_correct[src]
        print(f"  {src:<14}  N={n:>4}  top-1={c / max(n, 1) * 100:5.2f}%")

    # Per-sign top-1 (sorted by accuracy)
    per_class_rows = []
    for i, cls in enumerate(classes):
        n = int(per_class_total[i])
        c = int(per_class_correct[i])
        p = int(per_class_predicted[i])
        per_class_rows.append({
            "sign": cls, "n_val": n, "n_correct": c,
            "top1_recall": c / n if n > 0 else 0.0,
            "n_predicted": p,
            "magnet_ratio": p / n if n > 0 else (float("inf") if p > 0 else 0.0),
        })

    # Diff against baseline if provided
    if args.baseline_per_class and args.baseline_per_class.exists():
        baseline = json.loads(args.baseline_per_class.read_text())
        base_recall = {r["sign"]: r["top1_recall"] for r in baseline}
        for r in per_class_rows:
            r["baseline_top1"] = base_recall.get(r["sign"], None)
            r["delta"] = (r["top1_recall"] - r["baseline_top1"]) if r["baseline_top1"] is not None else None
        movers = [r for r in per_class_rows if r.get("delta") is not None]
        movers.sort(key=lambda r: r["delta"], reverse=True)
        print(f"\n=== BIGGEST GAINERS vs baseline (top 15) ===")
        for r in movers[:15]:
            print(f"  {r['sign']:<12}  {r['baseline_top1'] * 100:5.1f}% → {r['top1_recall'] * 100:5.1f}%  "
                  f"({r['delta'] * 100:+.1f}pp, N={r['n_val']})")
        print(f"\n=== BIGGEST LOSERS vs baseline (top 10) ===")
        for r in movers[-10:][::-1]:
            print(f"  {r['sign']:<12}  {r['baseline_top1'] * 100:5.1f}% → {r['top1_recall'] * 100:5.1f}%  "
                  f"({r['delta'] * 100:+.1f}pp, N={r['n_val']})")
    else:
        per_class_rows.sort(key=lambda r: -r["top1_recall"])
        print(f"\n=== TOP 15 SIGNS ===")
        for r in per_class_rows[:15]:
            print(f"  {r['sign']:<12}  {r['top1_recall'] * 100:5.1f}%  (N={r['n_val']})")
        print(f"\n=== BOTTOM 15 SIGNS ===")
        for r in per_class_rows[-15:]:
            print(f"  {r['sign']:<12}  {r['top1_recall'] * 100:5.1f}%  (N={r['n_val']})")

    # Confusion matrix top entries
    print(f"\n=== TOP {args.top_confusions} CONFUSIONS (true → predicted, count) ===")
    confusions_sorted = sorted(confusion.items(), key=lambda kv: -kv[1])
    for (true_idx, pred_idx), n in confusions_sorted[:args.top_confusions]:
        print(f"  {classes[true_idx]:<12} → {classes[pred_idx]:<12}  {n}× misclassified")

    # Magnet-ratio collapse detection
    collapsed = [r for r in per_class_rows if r["magnet_ratio"] > 2.5]
    if collapsed:
        print(f"\n=== MAGNET SIGNS (predicted ≥2.5× actual — classifier defaulting here) ===")
        for r in sorted(collapsed, key=lambda r: -r["magnet_ratio"])[:10]:
            print(f"  {r['sign']:<12}  predicted={r['n_predicted']}  actual={r['n_val']}  ratio={r['magnet_ratio']:.2f}×")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
