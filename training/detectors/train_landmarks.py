"""Training loop for the from-scratch hand landmark regressor.

Local:
    python -m training.detectors.train_landmarks \
        --train-manifest data/labeled_frames/hand_keypoints/external_train.json \
        --val-manifest   data/labeled_frames/hand_keypoints/external_val.json \
        --run-id hand_landmarks_v0_smoke --epochs 2 --batch-size 16

Modal:
    modal run training/modal_app.py::train_hand_landmarks ...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.detectors.hand_landmarks import HandLandmarkRegressor, count_parameters
from training.detectors.landmarks_augment import default_train_augment
from training.detectors.landmarks_dataset import HandLandmarkDataset, load_manifest


def _collate(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    coords = torch.stack([t["coords"] for t in targets], 0)
    vis = torch.stack([t["visibility"] for t in targets], 0)
    return images, {"coords": coords, "visibility": vis}


def _losses(out, target):
    # L1 on visible coords only
    vis = target["visibility"].unsqueeze(-1)  # (B, 21, 1)
    coord_diff = (out["coords"] - target["coords"]).abs()
    n_vis = vis.sum().clamp(min=1.0) * 2  # (x, y)
    coord_loss = (coord_diff * vis).sum() / n_vis

    if "visibility" in out:
        vis_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out["visibility"], target["visibility"]
        )
    else:
        vis_loss = torch.tensor(0.0, device=coord_loss.device)
    total = coord_loss + 0.1 * vis_loss
    return {"total": total, "coord": coord_loss.detach(), "visibility": vis_loss.detach()}


def train(
    train_manifest: Path,
    val_manifest: Path,
    run_dir: Path,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    device: str | None = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]

    train_items = load_manifest(train_manifest)
    val_items = load_manifest(val_manifest)

    train_ds = HandLandmarkDataset(train_items, repo_root=repo_root, augment=default_train_augment)
    val_ds = HandLandmarkDataset(val_items, repo_root=repo_root, augment=None)
    print(f"train hands: {len(train_ds)}  val hands: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=_collate)

    model = HandLandmarkRegressor().to(device)
    print(f"params: {count_parameters(model):,}")
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    history = []
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        ttot, n = 0.0, 0
        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
            out = model(imgs)
            losses = _losses(out, targets)
            optim.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            ttot += losses["total"].item()
            n += 1
        sched.step()
        train_loss = ttot / max(n, 1)

        model.eval()
        vtot, nv = 0.0, 0
        per_kp_err = torch.zeros(21)
        per_kp_n = torch.zeros(21)
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
                out = model(imgs)
                losses = _losses(out, targets)
                vtot += losses["total"].item()
                nv += 1
                # per-keypoint pixel error at 224x224
                err = ((out["coords"] - targets["coords"]) ** 2).sum(-1).sqrt() * 224.0  # (B, 21)
                vis = targets["visibility"]
                per_kp_err += (err.cpu() * vis.cpu()).sum(0)
                per_kp_n += vis.cpu().sum(0)
        val_loss = vtot / max(nv, 1)
        mean_pkpe = (per_kp_err / per_kp_n.clamp(min=1)).mean().item()
        elapsed = time.time() - t0
        print(f"epoch {epoch:03d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"mean_keypoint_px_err={mean_pkpe:.2f}  lr={sched.get_last_lr()[0]:.2e}  {elapsed:.1f}s")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "mean_keypoint_px_err": mean_pkpe, "elapsed_s": elapsed})

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
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(train(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        run_dir=args.runs_dir / args.run_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
    ))
