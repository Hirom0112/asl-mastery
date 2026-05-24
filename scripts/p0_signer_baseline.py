"""P0 — honest, leak-free baseline (FREE / local).

Fixes the experimental hygiene the two senior-DS reviews flagged (2026-05-23):
  1. Recovers signer_id by joining trajectory JSONs -> unified manifest on
     clip_path (the trajectory extract dropped signer_id; the manifest keeps it).
  2. Builds a SIGNER-DISJOINT split (hold out whole signers, not clips). The
     manifest's own `split` field leaks 31/41 sem_lex val signers into train.
  3. NO --max-per-sign oversampling. Natural distribution.
  4. Sem-Lex-only, 100D (embeddings stripped) — the clean anchor the skeptic
     demanded. Reports top-1 / top-3 / top-5 + a magnet-collapse check
     (# distinct predicted classes).

Usage:
    training/.venv/bin/python -m scripts.p0_signer_baseline \
        --traj-root data/trajectories_v8_pull/trajectories_v8 \
        --manifest  data/labeled_frames/unified_clip_manifest_modal_v4.json \
        --vocab     dataset/slice1b_vocabulary.json \
        --sources   sem_lex \
        --run-dir   runs/p0_semlex_100d_signerdisjoint \
        --epochs 80
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from training.detectors.sign_classifier import SignClassifier, count_parameters
from training.detectors.sign_matcher import trajectory_from_frames
from training.detectors.fit_templates import _mirror_features, _resample_trajectory
from training.detectors.train_classifier import _load_vocab, _time_warp

TIME_STEPS = 32


def build_clip_to_signer(manifest_path: Path) -> dict[str, dict]:
    raw = json.loads(manifest_path.read_text())
    recs = raw if isinstance(raw, list) else next(
        v for v in raw.values() if isinstance(v, list))
    out = {}
    for r in recs:
        cp = r.get("clip_path")
        if cp:
            out[cp] = {"signer_id": r.get("signer_id"), "source": r.get("source")}
    return out


def gather(traj_root: Path, clip_meta: dict[str, dict], sources: set[str],
           strip_embeddings: bool, norm: str = "body"):
    """Return list of (sign, signer_key, source, feats_TF). signer_key is
    'src:signer_id' when present, else 'src:clip:<stem>' (clip-disjoint fallback)."""
    samples = []
    no_signer = Counter()
    not_in_manifest = 0
    for sign_dir in sorted(traj_root.iterdir()):
        if not sign_dir.is_dir():
            continue
        sign = sign_dir.name
        for j in sign_dir.glob("*.json"):
            try:
                t = json.loads(j.read_text())
            except Exception:
                continue
            cp = t.get("clip_path")
            meta = clip_meta.get(cp)
            if meta is None:
                not_in_manifest += 1
                continue
            src = meta.get("source")
            if sources and src not in sources:
                continue
            frames = t.get("frames", [])
            if len(frames) < 2:
                continue
            if strip_embeddings:
                for f in frames:
                    for h in (f.get("hands") or []):
                        h.pop("embedding", None)
            feats = trajectory_from_frames(frames, TIME_STEPS, norm=norm)
            if not np.isfinite(feats).any():
                continue
            sid = meta.get("signer_id")
            if sid is None:
                no_signer[src] += 1
                signer_key = f"{src}:clip:{j.stem}"
            else:
                signer_key = f"{src}:{sid}"
            samples.append((sign, signer_key, src, feats.astype(np.float32)))
    return samples, no_signer, not_in_manifest


def signer_disjoint_split(samples, val_frac: float, seed: int):
    import random
    rnd = random.Random(seed)
    signers = sorted({s[1] for s in samples})
    rnd.shuffle(signers)
    n_val = max(1, int(round(len(signers) * val_frac)))
    val_signers = set(signers[:n_val])
    train = [s for s in samples if s[1] not in val_signers]
    val = [s for s in samples if s[1] in val_signers]
    return train, val, val_signers, set(signers) - val_signers


class DS(Dataset):
    def __init__(self, rows, sign_to_idx, augment=False, jitter_std=5.0):
        self.x = [r[3] for r in rows]
        self.y = [sign_to_idx[r[0]] for r in rows]
        self.augment = augment
        # Additive coord jitter std. The v1 body-normalized features carry a
        # wide dynamic range so 5.0 is mild; the v2 hand-relative handshape is
        # hand-scaled (~unit magnitude), so the caller passes a smaller std to
        # keep the relative perturbation comparable across the A/B.
        self.jitter_std = jitter_std

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i].copy()
        if self.augment:
            scale = np.random.uniform(0.9, 1.1)
            jitter = np.random.normal(0.0, self.jitter_std, size=(1, x.shape[1]))
            mask = np.isfinite(x)
            x = np.where(mask, x * scale + jitter, x).astype(np.float32)
            if np.random.random() < 0.5:
                x = _mirror_features(x)
            if np.random.random() < 0.5:
                x = _time_warp(x, 0.15)
        return torch.from_numpy(x.astype(np.float32)), self.y[i]


def evaluate(model, loader, device, num_classes, collect_confusion=False):
    model.eval()
    total = correct1 = correct3 = correct5 = 0
    predicted = Counter()
    # confusion[(true_idx, pred_idx)] = count; per_class_total[true_idx] = count
    confusion: Counter = Counter()
    per_class_total: Counter = Counter()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            k = min(5, num_classes)
            top = logits.topk(k, dim=1).indices
            p1 = top[:, 0]
            correct1 += (p1 == y).sum().item()
            correct3 += (top[:, :min(3, k)] == y.unsqueeze(1)).any(1).sum().item()
            correct5 += (top == y.unsqueeze(1)).any(1).sum().item()
            total += y.size(0)
            yc = y.cpu().numpy()
            p1c = p1.cpu().numpy()
            for pi in p1c:
                predicted[int(pi)] += 1
            if collect_confusion:
                for ti, pi in zip(yc, p1c):
                    confusion[(int(ti), int(pi))] += 1
                    per_class_total[int(ti)] += 1
    out = {
        "top1": correct1 / max(total, 1),
        "top3": correct3 / max(total, 1),
        "top5": correct5 / max(total, 1),
        "n": total,
        "distinct_predicted": len(predicted),
    }
    if collect_confusion:
        out["confusion"] = confusion
        out["per_class_total"] = per_class_total
    return out


def print_confusion_submatrix(eval_out: dict, confusion_signs: list[str],
                              idx_to_sign: dict[int, str],
                              sign_to_idx: dict[str, int]) -> dict:
    """Print per-class top-1 + a confusion sub-matrix restricted to
    `confusion_signs` (rows = true sign, cols = predicted). Predicted columns
    are restricted to the same set; an 'other' column captures predictions
    that fell outside the set. Returns a JSON-serializable summary."""
    confusion: Counter = eval_out.get("confusion", Counter())
    per_class_total: Counter = eval_out.get("per_class_total", Counter())
    present = [s for s in confusion_signs if s in sign_to_idx]
    missing = [s for s in confusion_signs if s not in sign_to_idx]
    if missing:
        print(f"[P0]   (confusion-signs not in classes, skipped: {missing})")
    if not present:
        print("[P0]   no requested confusion-signs are present; skipping sub-matrix")
        return {"requested": confusion_signs, "present": [], "rows": {}}

    idxs = [sign_to_idx[s] for s in present]
    idx_set = set(idxs)
    print("\n[P0] === CONFUSION SUB-MATRIX (rows=true, cols=pred) ===")
    print(f"[P0]   signs: {present}")
    rows_summary: dict = {}
    for s in present:
        ti = sign_to_idx[s]
        tot = per_class_total.get(ti, 0)
        diag = confusion.get((ti, ti), 0)
        top1 = diag / tot if tot else 0.0
        cells = []
        row_counts: dict[str, int] = {}
        for s2 in present:
            pj = sign_to_idx[s2]
            c = confusion.get((ti, pj), 0)
            cells.append(f"{s2}={c}")
            row_counts[s2] = c
        other = tot - sum(confusion.get((ti, pj), 0) for pj in idx_set)
        row_counts["__other__"] = other
        print(f"[P0]   {s:>16}  n={tot:<4} top1={top1:.3f}  | "
              + "  ".join(cells) + f"  other={other}")
        rows_summary[s] = {"n": tot, "top1": top1, "cells": row_counts}
    return {"requested": confusion_signs, "present": present, "rows": rows_summary}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--sources", default="sem_lex",
                    help="comma-sep source filter; empty = all sources")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--feat-dim", type=int, default=100, choices=[100, 108, 356])
    ap.add_argument("--norm", default="hand", choices=["body", "hand"],
                    help="Feature normalization: body (v1, 100/356D) or hand "
                         "(v2, 108D hand-relative handshape). Default body.")
    ap.add_argument("--confusion-signs", default="",
                    help="Comma-separated sign names; prints per-class top-1 + "
                         "a confusion sub-matrix over them after eval (e.g. the "
                         "handshape-confusable set). Empty = overall only.")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--ablate", default="none",
                    choices=["none", "hands_only", "pose_only"],
                    help="Zero feature blocks to isolate signal source. "
                         "100D layout: hands[0:84] pose[84:100].")
    ap.add_argument("--block-norm", action="store_true",
                    help="Use BlockLayerNorm (per kpt/embed/pose block) instead "
                         "of global LayerNorm. The encoder-fairness fix.")
    args = ap.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    sources = {s for s in args.sources.split(",") if s} if args.sources else set()

    # v2 hand normalization is 108D and embedding-free; keep feat_dim coherent.
    if args.norm == "hand":
        if args.feat_dim not in (108,):
            print(f"[P0] norm=hand → forcing feat_dim 108 (was {args.feat_dim})")
            args.feat_dim = 108
    strip_emb = args.feat_dim == 100

    print(f"[P0] sources={sources or 'ALL'}  feat_dim={args.feat_dim}  norm={args.norm} "
          f"(strip_embeddings={strip_emb})")
    clip_meta = build_clip_to_signer(args.manifest)
    print(f"[P0] manifest clips: {len(clip_meta)}")

    print("[P0] gathering trajectories + joining signer_id ...")
    samples, no_signer, not_in_manifest = gather(
        args.traj_root, clip_meta, sources, strip_emb, norm=args.norm)
    print(f"[P0] usable clips: {len(samples)}  "
          f"(not_in_manifest={not_in_manifest}, no_signer_by_src={dict(no_signer)})")

    if args.ablate != "none":
        if args.feat_dim == 108:
            raise SystemExit(
                "[P0] --ablate is not defined for the 108D (v2/hand) layout; the "
                "hardcoded 100D hands[0:84]/pose[84:100] ranges don't apply. "
                "Re-run with --ablate none (default).")
        # 100D layout: hands[0:84] pose[84:100]; 356D: pose[340:356], hand-kpts
        # at [0:42]+[170:212] (embeds untouched here).
        for (_, _, _, feats) in samples:
            if args.feat_dim == 100:
                if args.ablate == "hands_only":
                    feats[:, 84:100] = 0.0
                elif args.ablate == "pose_only":
                    feats[:, 0:84] = 0.0
            else:  # 356D
                if args.ablate == "hands_only":
                    feats[:, 340:356] = 0.0
                elif args.ablate == "pose_only":
                    feats[:, 0:42] = 0.0
                    feats[:, 170:212] = 0.0
        print(f"[P0] ABLATION={args.ablate}: zeroed complementary feature block")

    train, val, val_signers, train_signers = signer_disjoint_split(
        samples, args.val_frac, args.seed)
    overlap = val_signers & train_signers  # 0 by construction
    print(f"[P0] SIGNER-DISJOINT split: {len(train)} train / {len(val)} val clips")
    print(f"[P0]   signers: {len(train_signers)} train / {len(val_signers)} val "
          f"| overlap = {len(overlap)} (must be 0)")
    print(f"[P0]   val sign coverage: "
          f"{len({r[0] for r in val})}/{len({r[0] for r in samples})}")

    vocab = _load_vocab(args.vocab)
    present = sorted({r[0] for r in samples} & set(vocab))
    sign_to_idx = {s: i for i, s in enumerate(present)}
    num_classes = len(present)
    train = [r for r in train if r[0] in sign_to_idx]
    val = [r for r in val if r[0] in sign_to_idx]
    idx_to_sign = {i: s for s, i in sign_to_idx.items()}
    print(f"[P0] classes: {num_classes}")
    # Print available sign names so the parent can pick handshape-confusable
    # ones to pass via --confusion-signs.
    print(f"[P0] available signs ({num_classes}): {present}")

    confusion_signs = [s for s in args.confusion_signs.split(",") if s]

    tr_counts = Counter(r[0] for r in train)
    print(f"[P0] train per-class range: {min(tr_counts.values())}.."
          f"{max(tr_counts.values())} (NO max-per-sign, natural dist)")

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[P0] device={device}")

    # v1 body coords have a wide range (5.0 jitter is mild); v2 handshape is
    # hand-scaled (~unit), so use a much smaller additive jitter there.
    jitter_std = 0.05 if args.norm == "hand" else 5.0
    tl = DataLoader(DS(train, sign_to_idx, augment=not args.no_augment,
                       jitter_std=jitter_std),
                    batch_size=args.batch_size, shuffle=True, drop_last=True)
    vl = DataLoader(DS(val, sign_to_idx, augment=False),
                    batch_size=args.batch_size, shuffle=False)

    norm_blocks = None
    if args.block_norm:
        if args.feat_dim == 356:
            norm_blocks = [42, 128, 42, 128, 16]
        elif args.feat_dim == 108:
            # v2 per slot: handshape 42 + location 2 + orientation 2 = 46; pose 16
            norm_blocks = [46, 46, 16]
        else:
            norm_blocks = [42, 42, 16]
    model = SignClassifier(num_features=args.feat_dim, num_classes=num_classes,
                           hidden=256, num_blocks=5,
                           norm_blocks=norm_blocks).to(device)
    print(f"[P0] params: {count_parameters(model):,}  "
          f"norm={'block' if norm_blocks else 'global'}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    history = []
    best = {"top1": -1.0}
    patience = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot = 0.0
        nb = 0
        for x, y in tl:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        m = evaluate(model, vl, device, num_classes)
        row = {"epoch": ep, "train_loss": tot / max(nb, 1), **m,
               "sec": time.time() - t0}
        history.append(row)
        (args.run_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"  ep {ep:03d}/{args.epochs}  loss={row['train_loss']:.3f}  "
              f"top1={m['top1']:.3f} top3={m['top3']:.3f} top5={m['top5']:.3f}  "
              f"distinct_pred={m['distinct_predicted']}/{num_classes}  {row['sec']:.1f}s")
        if m["top1"] > best["top1"] + 1e-4:
            best = {**m, "epoch": ep}; patience = 0
        else:
            patience += 1
            if patience >= 12:
                print(f"  early stop (best top1={best['top1']:.3f} @ ep {best['epoch']})")
                break

    # `best` may carry the per-epoch Counters if a future edit collects them;
    # the per-epoch evaluate() here does NOT, so best is JSON-clean.
    confusion_summary = None
    if confusion_signs:
        final = evaluate(model, vl, device, num_classes, collect_confusion=True)
        confusion_summary = print_confusion_submatrix(
            final, confusion_signs, idx_to_sign, sign_to_idx)

    summary = {
        "phase": "P0",
        "sources": sorted(sources) or "ALL",
        "feat_dim": args.feat_dim,
        "norm": args.norm,
        "split": "signer-disjoint",
        "signer_overlap": len(overlap),
        "n_train": len(train), "n_val": len(val), "n_classes": num_classes,
        "best": best,
    }
    if confusion_summary is not None:
        summary["confusion_signs"] = confusion_summary
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[P0] === HONEST BASELINE ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
