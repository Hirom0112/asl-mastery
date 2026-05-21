"""Training entry point for the landmark classifier.

Per docs/MODEL.md §2 and docs/ARCHITECTURE.md §2.5 step 5.

Records every run's git commit, dataset version hash, MediaPipe
version (as recorded in the manifest), random seed, and full
hyperparameter config. Per docs/ROADMAP.md Phase 4, training time
per run is on the order of minutes under the landmark-based
architecture (ADR 0006).

Usage:

    python -m training.classifier.train \\
        --manifest dataset/clean/v1/dataset_v1_manifest.json \\
        --model bilstm \\
        --epochs 60 \\
        --output runs/v1-001/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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
from training.classifier.dataset import KeypointDataset, make_weighted_sampler  # noqa: E402
from training.classifier.init import init_classifier_weights  # noqa: E402
from training.classifier.model import (  # noqa: E402
    BiLSTMClassifier,
    TransformerClassifier,
    count_parameters,
)
from training.keypoints import TOTAL_COORDS  # noqa: E402

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


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "bilstm":
        return BiLSTMClassifier(input_dim=TOTAL_COORDS, num_classes=num_classes)
    if name == "transformer":
        return TransformerClassifier(input_dim=TOTAL_COORDS, num_classes=num_classes)
    raise ValueError(f"unknown model: {name}")


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

    train_ds = KeypointDataset(args.manifest, split="train", augment_training=True)
    val_ds = KeypointDataset(args.manifest, split="val", augment_training=False)

    num_classes = len(train_ds.classes)
    log.info("classes: %d", num_classes)
    log.info("train clips: %d  val clips: %d", len(train_ds), len(val_ds))

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, num_classes).to(device)
    init_classifier_weights(model)
    log.info("model: %s  params: %d", args.model, count_parameters(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    run_meta: dict[str, Any] = {
        "args": vars(args),
        "git_sha": _git_sha(),
        "param_count": count_parameters(model),
        "device": str(device),
        "num_classes": num_classes,
        "classes": train_ds.classes,
        "manifest": str(args.manifest),
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
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
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
                    "model_name": args.model,
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
    parser.add_argument("--model", choices=["bilstm", "transformer"], default="bilstm")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    args = parser.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
