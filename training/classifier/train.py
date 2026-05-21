"""Training entry point for the v3.x small 3D CNN.

Per docs/MODEL.md §2 and docs/ARCHITECTURE.md §2.5 step 5. Adapted
from the BiLSTM-on-keypoints trainer that shipped under the now-
superseded ADR 0006: the model is `SmallR2Plus1D` from
`training/classifier/cnn.py` and the dataset is
`VideoClipDataset` from `training/classifier/dataset_video.py`. The
training shape is `(B, T=16, H, W, 3)` raw RGB.

Records every run's git commit, dataset version hash, random seed,
and full hyperparameter config. Training time per run rises from
the minutes-per-run figure that served the BiLSTM to **hours** under
Path B (per docs/MODEL.md §2 and docs/ROADMAP.md Phase 4).

Usage:

    python -m training.classifier.train \\
        --manifest dataset/clean/v3/dataset_v3_manifest.json \\
        --output runs/v3-001/ \\
        --epochs 60 \\
        --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.classifier.augment import (  # noqa: E402
    load_background_bank,
    make_train_augment,
    make_val_transform,
)
from training.classifier.cnn import SmallR2Plus1D, count_parameters  # noqa: E402
from training.classifier.dataset_video import (  # noqa: E402
    VideoClipDataset,
    make_weighted_sampler,
)

log = logging.getLogger("train")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _flippable_lookup(manifest_path: Path) -> dict[str, bool]:
    """Per-sign `flippable` flags from the cleaning-pipeline manifest if
    present; falls back to all-False (conservative: never flip)."""
    try:
        with manifest_path.open() as f:
            m = json.load(f)
    except Exception:
        return {}
    # The manifest can carry a `flippable` flag per record; promote it
    # to a sign-level dict.
    out: dict[str, bool] = {}
    for r in m.get("records", []):
        sid = r.get("sign_id")
        if sid and "flippable" in r:
            out[sid] = bool(r["flippable"])
    return out


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(args.seed)

    flippable = _flippable_lookup(args.manifest)
    background_bank = load_background_bank(args.background_bank) if args.background_bank else []

    train_augment = make_train_augment(
        input_height=args.input_size,
        input_width=args.input_size,
        background_bank=background_bank,
        apply_bg_swap_prob=args.bg_swap_prob,
        flippable_lookup=flippable,
    )
    val_transform = make_val_transform(input_height=args.input_size, input_width=args.input_size)

    train_ds = VideoClipDataset(
        args.manifest,
        split="train",
        augment=train_augment,  # signature (frames, sign_id) — dataset passes both
        input_height=args.input_size,
        input_width=args.input_size,
    )
    val_ds = VideoClipDataset(
        args.manifest,
        split="val",
        augment=val_transform,
        input_height=args.input_size,
        input_width=args.input_size,
    )

    num_classes = len(train_ds.classes)
    log.info("classes: %d", num_classes)
    log.info("train clips: %d  val clips: %d", len(train_ds), len(val_ds))

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallR2Plus1D(
        num_classes=num_classes,
        input_height=args.input_size,
        input_width=args.input_size,
    ).to(device)
    log.info("model: SmallR2Plus1D  params: %d", count_parameters(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    run_meta: dict[str, Any] = {
        "args": vars(args),
        "git_sha": _git_sha(),
        "param_count": count_parameters(model),
        "device": str(device),
        "num_classes": num_classes,
        "classes": train_ds.classes,
        "manifest": str(args.manifest),
        "input_height": args.input_size,
        "input_width": args.input_size,
        "model_architecture": "small_r2plus1d",
    }
    with (output_dir / "run.json").open("w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    best_val_acc = 0.0
    best_path = output_dir / "best.pt"
    patience = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * x.size(0)
            total += x.size(0)

        scheduler.step()
        train_loss = total_loss / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        log.info(
            "epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "classes": train_ds.classes,
                    "model_architecture": "small_r2plus1d",
                    "input_height": args.input_size,
                    "input_width": args.input_size,
                    "val_acc": val_acc,
                },
                best_path,
            )
            log.info("new best val_acc=%.4f saved → %s", val_acc, best_path)
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                log.info("early stop at epoch %d (patience=%d)", epoch, patience)
                break

    log.info("done. best val_acc=%.4f", best_val_acc)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--background-bank",
        type=Path,
        default=None,
        help="Directory of replacement background images for MOG2 background swap (classical CV per ADR 0005). Empty / missing dir = no-op.",
    )
    parser.add_argument("--bg-swap-prob", type=float, default=0.5)
    args = parser.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
