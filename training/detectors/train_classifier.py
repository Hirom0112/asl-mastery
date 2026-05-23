"""Train the SignClassifier head on extracted trajectories.

Same train/val split as templates_v7 (5468 / 1332 clips, 70 signs).
The classifier replaces the Mahalanobis matcher's role.

Usage (local):
    python -m training.detectors.train_classifier \
        --train-dir data/trajectories_v6_train \
        --val-dir data/trajectories_v6_val \
        --vocab-json dataset/slice1b_vocabulary.json \
        --run-dir runs/sign_classifier_v0 \
        --epochs 80 --batch-size 256 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from training.detectors.sign_classifier import (
    SignClassifier,
    TransformerSignClassifier,
    count_parameters,
)
from training.detectors.sign_matcher import trajectory_from_frames
from training.detectors.fit_templates import _mirror_features, _resample_trajectory

TIME_STEPS = 32


def _time_warp(traj_TF: np.ndarray, max_stretch: float = 0.15) -> np.ndarray:
    """Stretch/compress along time axis by ±max_stretch (default 15%).

    Pick a random scale s ∈ [1-max_stretch, 1+max_stretch], remap the T
    indices to virtual indices in [0, T*s], then re-resample to T frames."""
    T, F = traj_TF.shape
    s = 1.0 + np.random.uniform(-max_stretch, max_stretch)
    new_T = max(2, int(round(T * s)))
    src_idx = np.linspace(0, T - 1, new_T)
    lo = np.floor(src_idx).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (src_idx - lo).reshape(-1, 1)
    warped = traj_TF[lo] * (1 - w) + traj_TF[hi] * w
    return _resample_trajectory(warped.astype(np.float32), T)


def _load_vocab(vocab_path: Path) -> list[str]:
    raw = json.loads(vocab_path.read_text())
    if isinstance(raw, list):
        signs = raw
    elif isinstance(raw, dict):
        if "kept_signs" in raw and isinstance(raw["kept_signs"], list):
            signs = raw["kept_signs"]
        elif "signs" in raw:
            signs = raw["signs"]
        else:
            signs = sorted(raw.keys())
    else:
        raise ValueError(f"unsupported vocab format: {type(raw)}")
    if signs and isinstance(signs[0], dict):
        signs = [
            s.get("sign_id") or s.get("id") or s.get("sign") or s.get("name")
            for s in signs
        ]
    return sorted(s for s in signs if s)


class TrajectoryDataset(Dataset):
    """Loads all (T,F) feature tensors + labels into RAM upfront. Tiny."""

    def __init__(self, root: Path, sign_to_idx: dict[str, int],
                 augment: bool = False, augment_jitter_px: float = 5.0,
                 augment_scale: tuple[float, float] = (0.9, 1.1),
                 augment_mirror_p: float = 0.5,
                 augment_timewarp_p: float = 0.5,
                 augment_timewarp_max: float = 0.15):
        self.augment = augment
        self.jitter_px = augment_jitter_px
        self.scale_lo, self.scale_hi = augment_scale
        self.mirror_p = augment_mirror_p
        self.timewarp_p = augment_timewarp_p
        self.timewarp_max = augment_timewarp_max
        self.features: list[np.ndarray] = []
        self.labels: list[int] = []
        skipped = 0

        # Gather all (sign, json_path) tuples first — this also has volume RTT
        # per glob, but it's 80 small dirs vs ~13K small files.
        all_jobs: list[tuple[str, Path]] = []
        for sign_dir in sorted(root.iterdir()):
            if not sign_dir.is_dir():
                continue
            sign = sign_dir.name
            if sign not in sign_to_idx:
                continue
            for j in sign_dir.glob("*.json"):
                all_jobs.append((sign, j))

        # Parallel read + parse via thread pool. Each file is a single read +
        # JSON decode; threads release the GIL during the actual read. This
        # turns ~13K × 50ms-RTT sequential = 11min into ~13K / 32 × 50ms = 21s.
        from concurrent.futures import ThreadPoolExecutor

        def _load_one(args):
            sign, j = args
            try:
                traj = json.loads(j.read_text())
            except Exception:
                return None
            frames = traj.get("frames", [])
            if len(frames) < 2:
                return None
            feats = trajectory_from_frames(frames, TIME_STEPS)
            if not np.isfinite(feats).any():
                return None
            return (sign_to_idx[sign], feats.astype(np.float32))

        with ThreadPoolExecutor(max_workers=32) as ex:
            for res in ex.map(_load_one, all_jobs):
                if res is None:
                    skipped += 1
                    continue
                label, feats = res
                self.labels.append(label)
                self.features.append(feats)
        print(f"  loaded {len(self.features)} clips from {root} "
              f"(skipped {skipped})")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        x = self.features[idx].copy()
        if self.augment:
            # Per-clip global scale + xy jitter on the finite entries
            scale = np.random.uniform(self.scale_lo, self.scale_hi)
            jitter = np.random.normal(0.0, self.jitter_px, size=(1, x.shape[1]))
            mask = np.isfinite(x)
            x = np.where(mask, x * scale + jitter, x).astype(np.float32)
            # Horizontal mirror (also swaps hand slots — proper symmetry)
            if np.random.random() < self.mirror_p:
                x = _mirror_features(x)
            # Time-warp ±15%
            if np.random.random() < self.timewarp_p:
                x = _time_warp(x, max_stretch=self.timewarp_max)
        return torch.from_numpy(x.astype(np.float32)), self.labels[idx]


def evaluate(model: nn.Module, loader: DataLoader, device: str,
             num_classes: int) -> dict:
    model.eval()
    total = 0
    correct = 0
    correct_top3 = 0
    per_class_correct = np.zeros(num_classes, dtype=np.int64)
    per_class_total = np.zeros(num_classes, dtype=np.int64)
    per_class_predicted = np.zeros(num_classes, dtype=np.int64)
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss_sum += crit(logits, y).item()
            top3 = logits.topk(3, dim=1).indices
            pred = top3[:, 0]
            correct += (pred == y).sum().item()
            correct_top3 += (top3 == y.unsqueeze(1)).any(dim=1).sum().item()
            total += y.size(0)
            for yi, pi in zip(y.cpu().numpy(), pred.cpu().numpy()):
                per_class_total[yi] += 1
                per_class_predicted[pi] += 1
                if yi == pi:
                    per_class_correct[yi] += 1
    return {
        "loss": loss_sum / max(total, 1),
        "top1": correct / max(total, 1),
        "top3": correct_top3 / max(total, 1),
        "n": total,
        "per_class_correct": per_class_correct,
        "per_class_total": per_class_total,
        "per_class_predicted": per_class_predicted,
    }


def train(
    train_dir: Path,
    val_dir: Path,
    vocab_json: Path,
    run_dir: Path,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
    use_bf16: bool = True,
    early_stop_patience: int = 8,
    early_stop_delta: float = 0.001,
    augment: bool = True,
    balanced_sampling: str = "none",
    max_per_sign: int = 0,
    zero_embeddings: bool = False,
    arch: str = "tcn",
    transformer_d_model: int = 192,
    transformer_layers: int = 3,
    transformer_heads: int = 4,
    transformer_ff: int = 512,
    device: Optional[str] = None,
) -> dict:
    """balanced_sampling: 'none' | 'sqrt' | 'inverse'.
    - 'none' (default, baseline): plain shuffle.
    - 'sqrt' (soft cap, Phase 4.10): weights = 1/sqrt(freq). Well-established
      middle ground in imbalanced learning. Doesn't drop data.
    - 'inverse' (hard cap): weights = 1/freq. Aggressive; prior comment in
      this file noted class-weighting hurt val accuracy — that was loss-side
      not batch-composition-side, but caveat applies.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir.mkdir(parents=True, exist_ok=True)
    vocab = _load_vocab(vocab_json)
    sign_to_idx = {s: i for i, s in enumerate(vocab)}
    idx_to_sign = vocab
    print(f"vocab size: {len(vocab)}")
    print("loading train…")
    train_ds = TrajectoryDataset(train_dir, sign_to_idx, augment=augment)
    print("loading val…")
    val_ds = TrajectoryDataset(val_dir, sign_to_idx, augment=False)

    # Bidirectional per-sign target: cap rich classes DOWN to max_per_sign,
    # oversample thin classes UP to max_per_sign via index replication. Each
    # duplicate gets a fresh augmentation roll at __getitem__ time, so the
    # effective dataset is `max_per_sign × num_classes` augmented examples
    # per epoch, perfectly class-balanced.
    if max_per_sign > 0:
        import random as _random
        from collections import defaultdict as _dd
        rnd = _random.Random(42)
        by_class: dict[int, list[int]] = _dd(list)
        for idx, lbl in enumerate(train_ds.labels):
            by_class[lbl].append(idx)
        keep_idxs: list[int] = []
        oversampled = 0
        for lbl, idxs in by_class.items():
            rnd.shuffle(idxs)
            if len(idxs) >= max_per_sign:
                # Subsample (cap down to max_per_sign)
                keep_idxs.extend(idxs[:max_per_sign])
            else:
                # Oversample with replacement (pad up to max_per_sign)
                # Each repeated index will get a different augmentation at
                # __getitem__ time, producing distinct augmented samples.
                keep_idxs.extend(idxs)
                shortfall = max_per_sign - len(idxs)
                pad = [rnd.choice(idxs) for _ in range(shortfall)]
                keep_idxs.extend(pad)
                oversampled += 1
        rnd.shuffle(keep_idxs)
        n_before = len(train_ds.labels)
        train_ds.features = [train_ds.features[i] for i in keep_idxs]
        train_ds.labels = [train_ds.labels[i] for i in keep_idxs]
        n_after = len(train_ds.labels)
        n_classes_now = len({l for l in train_ds.labels})
        print(f"  balanced train to exactly {max_per_sign}/sign: {n_before} → {n_after} "
              f"({n_classes_now} classes, {oversampled} signs oversampled with aug)")

    # Optional: zero out the 128D embedding portions of each frame to test
    # encoder vs no-encoder downstream impact. Layout (per fit_templates.py):
    #   slot0 kpts[0:42] embed[42:170] slot1 kpts[170:212] embed[212:340] pose[340:356]
    # Only applies when feat dim == 356.
    if zero_embeddings:
        zeroed = 0
        for i, feats in enumerate(train_ds.features):
            if feats.shape[1] == 356:
                feats[:, 42:170] = 0.0
                feats[:, 212:340] = 0.0
                zeroed += 1
        for i, feats in enumerate(val_ds.features):
            if feats.shape[1] == 356:
                feats[:, 42:170] = 0.0
                feats[:, 212:340] = 0.0
        print(f"  zero_embeddings=True: zeroed 256D embed portion in {zeroed} train clips "
              f"(model will see skeleton+pose only)")
    # Restrict the model to the classes actually present in data
    present_idxs = sorted({l for l in train_ds.labels} | {l for l in val_ds.labels})
    print(f"classes present in data: {len(present_idxs)} / {len(vocab)}")
    # Remap labels to dense range
    remap = {old: new for new, old in enumerate(present_idxs)}
    train_ds.labels = [remap[l] for l in train_ds.labels]
    val_ds.labels = [remap[l] for l in val_ds.labels]
    num_classes = len(present_idxs)
    classes_in_use = [idx_to_sign[i] for i in present_idxs]
    (run_dir / "classes.json").write_text(json.dumps(classes_in_use, indent=2))

    # v2 default: plain CE + label smoothing (no class weighting). v1's
    # class-weight attempt hurt val accuracy on natural distributions.
    # v3 option: WeightedRandomSampler (batch composition, not loss weighting).
    from collections import Counter
    import numpy as np
    class_counts = Counter(train_ds.labels)
    print(f"  class frequency range: {min(class_counts.values())}..{max(class_counts.values())} "
          f"(balanced_sampling={balanced_sampling})")

    if balanced_sampling in ("sqrt", "inverse"):
        freqs = np.array([class_counts.get(i, 1) for i in range(num_classes)],
                         dtype=np.float64)
        if balanced_sampling == "sqrt":
            class_weight = 1.0 / np.sqrt(freqs)
        else:
            class_weight = 1.0 / freqs
        per_sample_weight = class_weight[np.array(train_ds.labels)]
        sampler = WeightedRandomSampler(
            weights=per_sample_weight.tolist(),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, drop_last=True,
                                  pin_memory=device == "cuda")
        print(f"  using WeightedRandomSampler ({balanced_sampling}); "
              f"effective weight range: {class_weight.min():.4f}..{class_weight.max():.4f}")
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True,
                                  pin_memory=device == "cuda")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers,
                            pin_memory=device == "cuda")

    # Auto-detect feature dim from the data (100 baseline, 356 with handshape embed)
    if train_ds.features:
        feat_dim = int(train_ds.features[0].shape[1])
    else:
        feat_dim = 100
    print(f"detected feature dim: {feat_dim}")

    if arch == "transformer":
        model = TransformerSignClassifier(
            num_features=feat_dim,
            num_classes=num_classes,
            d_model=transformer_d_model,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_ff,
        ).to(device)
    elif arch == "tcn":
        model = SignClassifier(num_features=feat_dim, num_classes=num_classes,
                               hidden=256, num_blocks=5).to(device)
    else:
        raise ValueError(f"unknown arch: {arch!r} (expected 'tcn' or 'transformer')")
    print(f"arch={arch}  classifier params: {count_parameters(model):,}")
    optim = torch.optim.AdamW(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    amp_dtype = torch.bfloat16 if (use_bf16 and device == "cuda") else None

    history: list[dict] = []
    best_top1 = -1.0
    best_epoch = -1
    patience = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(x)
                    loss = crit(logits, y)
            else:
                logits = model(x)
                loss = crit(logits, y)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"non-finite loss at epoch={epoch} batch={n_batches}")
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            train_loss += loss.item()
            n_batches += 1
        sched.step()
        train_loss /= max(n_batches, 1)
        elapsed = time.time() - t0

        val = evaluate(model, val_loader, device, num_classes)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_top1": val["top1"],
            "val_top3": val["top3"],
            "lr": optim.param_groups[0]["lr"],
            "epoch_sec": elapsed,
        }
        history.append(row)
        print(f"epoch {epoch:03d}/{epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val['loss']:.4f}  "
              f"top1={val['top1']:.3f}  top3={val['top3']:.3f}  "
              f"{elapsed:.1f}s")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        improved = val["top1"] > best_top1 + early_stop_delta
        if improved:
            best_top1 = val["top1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({
                "epoch": epoch,
                "state_dict": best_state,
                "val_top1": val["top1"],
                "val_top3": val["top3"],
                "classes_in_use": classes_in_use,
            }, run_dir / "best.pt")
            patience = 0
        else:
            patience += 1
            if early_stop_patience > 0 and patience >= early_stop_patience:
                print(f"  early stop: no top1 improvement ≥ {early_stop_delta} "
                      f"for {early_stop_patience} epochs (best epoch {best_epoch}, top1={best_top1:.3f})")
                break

    # Final last.pt
    torch.save({
        "epoch": history[-1]["epoch"],
        "state_dict": model.state_dict(),
        "classes_in_use": classes_in_use,
    }, run_dir / "last.pt")

    # Reload best for final eval
    if best_state is not None:
        model.load_state_dict(best_state)
        final = evaluate(model, val_loader, device, num_classes)
        # Magnet ratio + per-class top-1 on the best model
        pcc = final["per_class_correct"]
        pct = final["per_class_total"]
        pcp = final["per_class_predicted"]
        per_class = []
        for i, cls in enumerate(classes_in_use):
            per_class.append({
                "sign": cls,
                "n_val": int(pct[i]),
                "n_correct": int(pcc[i]),
                "top1_recall": float(pcc[i] / pct[i]) if pct[i] > 0 else 0.0,
                "n_predicted": int(pcp[i]),
                "magnet_ratio": float(pcp[i] / pct[i]) if pct[i] > 0 else (float("inf") if pcp[i] > 0 else 0.0),
            })
        (run_dir / "per_class.json").write_text(json.dumps(per_class, indent=2))

    return {
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_top1": best_top1,
        "n_classes": num_classes,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=Path, required=True)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--vocab-json", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--early-stop-patience", type=int, default=8)
    ap.add_argument("--balanced-sampling", default="none",
                    choices=["none", "sqrt", "inverse"],
                    help="Soft rebalancing via WeightedRandomSampler. "
                         "sqrt = 1/sqrt(freq) per Phase 4.10.")
    ap.add_argument("--max-per-sign", type=int, default=0,
                    help="Per-class hard cap on train clips (val unaffected). "
                         "0 = no cap. Use when rich classes magnet-collapse.")
    ap.add_argument("--zero-embeddings", action="store_true",
                    help="Zero out 128D handshape-encoder embedding portions of "
                         "each frame (when present). Diagnostic A/B for encoder "
                         "contribution: trains on skeleton+pose only.")
    ap.add_argument("--arch", default="tcn", choices=["tcn", "transformer"],
                    help="Classifier backbone. transformer = Phase 4.7 head.")
    ap.add_argument("--transformer-d-model", type=int, default=192)
    ap.add_argument("--transformer-layers", type=int, default=3)
    ap.add_argument("--transformer-heads", type=int, default=4)
    ap.add_argument("--transformer-ff", type=int, default=512)
    args = ap.parse_args()

    result = train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        vocab_json=args.vocab_json,
        run_dir=args.run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        use_bf16=not args.no_bf16,
        augment=not args.no_augment,
        early_stop_patience=args.early_stop_patience,
        balanced_sampling=args.balanced_sampling,
        max_per_sign=args.max_per_sign,
        zero_embeddings=args.zero_embeddings,
        arch=args.arch,
        transformer_d_model=args.transformer_d_model,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_ff=args.transformer_ff,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
