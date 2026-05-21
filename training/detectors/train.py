"""Training loop for the from-scratch hand detector.

Local invocation:
    python -m training.detectors.train \
        --train-manifest data/labeled_frames/hand_bbox/train.json \
        --val-manifest   data/labeled_frames/hand_bbox/val.json \
        --run-id hand_det_v0_smoke \
        --epochs 2 \
        --batch-size 8

Modal invocation is wired in training/modal_app.py::train_hand_detector
(scaffolded alongside this file).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.detectors.augment import default_train_augment
from training.detectors.dataset import HandBboxDataset, load_manifest
from training.detectors.hand_detector import HandDetector, count_parameters
from training.detectors.losses import HandDetectorLoss


def _collate(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    stacked = {k: torch.stack([t[k] for t in targets], dim=0) for k in targets[0]}
    return images, stacked


def train(
    train_manifest: Path,
    val_manifest: Path,
    run_dir: Path,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    device: str | None = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_manifest(train_manifest)
    val_records = load_manifest(val_manifest)
    print(f"train frames: {len(train_records)}  val frames: {len(val_records)}")

    train_ds = HandBboxDataset(train_records, augment=default_train_augment)
    val_ds = HandBboxDataset(val_records, augment=None)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )

    model = HandDetector().to(device)
    print(f"hand detector params: {count_parameters(model):,}")
    loss_fn = HandDetectorLoss().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    history = []
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_total = 0.0
        n_batches = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
            outputs = model(images)
            losses = loss_fn(outputs, targets)
            optim.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            train_total += losses["total"].item()
            n_batches += 1
        sched.step()
        train_loss = train_total / max(n_batches, 1)

        model.eval()
        val_total = 0.0
        n_val = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
                outputs = model(images)
                losses = loss_fn(outputs, targets)
                val_total += losses["total"].item()
                n_val += 1
        val_loss = val_total / max(n_val, 1)

        elapsed = time.time() - t0
        msg = (
            f"epoch {epoch:03d}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"lr={sched.get_last_lr()[0]:.2e}  {elapsed:.1f}s"
        )
        print(msg)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "elapsed_s": elapsed}
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch}, run_dir / "best.pt")

    torch.save({"model": model.state_dict(), "epoch": epochs}, run_dir / "last.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {"best_val_loss": best_val, "epochs": epochs, "run_dir": str(run_dir)}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, required=True)
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_dir = args.runs_dir / args.run_id
    result = train(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        run_dir=run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
    )
    print(result)
