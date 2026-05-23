"""Train the HandshapeEncoder via NT-Xent contrastive learning.

Phase 4.6 Stage 2. Reads the crop index produced by scripts/dump_hand_crops.py
and trains a 11M-param ResNet-like encoder to produce 128D L2-normalized
embeddings. Two augmented views per crop per step.

Split policy: by clip_id (all crops of one source clip stay in train or val),
so the encoder can't trivially memorize via near-duplicate frames.

Usage (local):
    training/.venv/bin/python -m training.detectors.train_handshape \\
        --crops-root data/hand_crops \\
        --run-dir runs/handshape_v0 \\
        --epochs 60 --batch-size 1024

Modal: see training/modal_app.py::train_handshape_encoder.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.io as tvio
import torchvision.transforms.v2 as T2
from torch.utils.data import DataLoader, Dataset

from training.detectors.handshape_encoder import (
    HandshapeEncoder,
    count_parameters,
    nt_xent_loss,
)


def _build_augment(image_size: int = 224) -> T2.Compose:
    """Two-view augmentation for handshape contrastive. Hand-handedness-
    preserving (NO hflip), camera-variation-friendly."""
    return T2.Compose([
        T2.RandomResizedCrop(image_size, scale=(0.8, 1.0), antialias=True),
        T2.RandomRotation(degrees=15),
        T2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T2.RandomGrayscale(p=0.1),
        T2.ToDtype(torch.float32, scale=True),
    ])


class HandCropDataset(Dataset):
    """Returns RAW uint8 crops (no augmentation) — augmentation runs ON THE GPU
    in the training loop to keep CPU workers fast.

    Cache: all 182K crops pre-decoded into a single uint8 tensor at init
    (~27 GB). `share_memory_()` is intentionally NOT called because Modal's
    /dev/shm is too small (~64 MB default) and SIGBUSes workers. Instead we
    rely on Linux copy-on-write fork: each worker inherits a read-only view
    of the cache pages, no duplication unless modified (we never modify).
    """

    INPUT_SIZE = 224

    def __init__(self, records: list[dict],
                 cache_in_memory: bool = True, n_cache_workers: int = 16):
        self.records = records
        self.cache_in_memory = cache_in_memory
        self._cache: torch.Tensor | None = None
        if cache_in_memory:
            self._build_cache(n_cache_workers)

    def _build_cache(self, n_workers: int) -> None:
        """Decode + resize every crop ONCE into a main-process uint8 tensor.

        NOT shared via share_memory_() — that uses /dev/shm which is small in
        Modal containers and causes SIGBUS on worker access. With CoW fork
        (Linux default DataLoader behavior), read-only access in workers
        doesn't duplicate pages, so memory cost is roughly N × 3 × S² bytes
        in the main process only.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        N = len(self.records)
        S = self.INPUT_SIZE
        self._cache = torch.empty((N, 3, S, S), dtype=torch.uint8)
        print(f"  building in-memory crop cache (CoW, no shm): {N} crops × 3 × {S} × {S} uint8 "
              f"= {self._cache.numel() / 1e9:.1f} GB")

        def _load(i: int) -> tuple[int, torch.Tensor]:
            img = tvio.read_image(self.records[i]["crop_path"], mode=tvio.ImageReadMode.RGB)
            _, h, w = img.shape
            if (h, w) != (S, S):
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0).float(), size=(S, S),
                    mode="bilinear", align_corners=False,
                ).squeeze(0).clamp(0, 255).to(torch.uint8)
            return i, img

        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_load, i) for i in range(N)]
            for fut in as_completed(futures):
                i, img = fut.result()
                self._cache[i].copy_(img)
                done += 1
                if done % 20000 == 0:
                    rate = done / (time.time() - t0)
                    print(f"    cache: {done}/{N}  ({rate:.0f} crops/s)")
        print(f"  cache built in {time.time() - t0:.0f}s ({N / (time.time() - t0):.0f} crops/s)")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        """Return raw uint8 crop (3, 224, 224). Augmentation happens on GPU."""
        if self._cache is not None:
            return self._cache[idx]
        return tvio.read_image(self.records[idx]["crop_path"], mode=tvio.ImageReadMode.RGB)


def _build_gpu_augment(image_size: int = 224) -> T2.Compose:
    """GPU-side augmentation. Applied per-batch AFTER move-to-device. Same
    transforms as before; torchvision.v2 supports GPU tensors directly."""
    return T2.Compose([
        T2.RandomResizedCrop(image_size, scale=(0.8, 1.0), antialias=True),
        T2.RandomRotation(degrees=15),
        T2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T2.RandomGrayscale(p=0.1),
        T2.ToDtype(torch.float32, scale=True),
    ])


def _split_records(records: list[dict], val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    by_clip: dict[str, list[dict]] = {}
    for r in records:
        by_clip.setdefault(r["clip_id"], []).append(r)
    clip_ids = sorted(by_clip.keys())
    rnd = random.Random(seed)
    rnd.shuffle(clip_ids)
    n_val = max(1, int(len(clip_ids) * val_frac))
    val_clips = set(clip_ids[:n_val])
    train, val = [], []
    for cid, recs in by_clip.items():
        (val if cid in val_clips else train).extend(recs)
    return train, val


def train(
    crops_root: Path,
    run_dir: Path,
    epochs: int = 60,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 0.1,
    num_workers: int = 12,
    val_frac: float = 0.05,
    seed: int = 42,
    use_bf16: bool = True,
    use_compile: bool = True,
    use_channels_last: bool = True,
    early_stop_patience: int = 8,
    device: Optional[str] = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir.mkdir(parents=True, exist_ok=True)

    idx = json.loads((crops_root / "index.json").read_text())
    records = idx["records"]
    print(f"loaded {len(records)} crop records ({len({r['clip_id'] for r in records})} unique clips)")

    train_recs, val_recs = _split_records(records, val_frac, seed)
    print(f"split: train={len(train_recs)}  val={len(val_recs)}")

    gpu_augment = _build_gpu_augment(HandshapeEncoder.INPUT_SIZE)
    train_ds = HandCropDataset(train_recs, cache_in_memory=True)
    val_ds = HandCropDataset(val_recs, cache_in_memory=True)
    # num_workers > 0 + CoW fork: workers inherit the cache pages read-only,
    # so we get parallel slice+pin without /dev/shm pressure.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0), prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=True,
        num_workers=min(4, num_workers), pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    model = HandshapeEncoder().to(device)
    if use_channels_last and device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    print(f"encoder params: {count_parameters(model):,}")

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    amp_dtype = torch.bfloat16 if (use_bf16 and device == "cuda") else None

    if use_compile and device == "cuda":
        try:
            # Use 'default' mode (not 'reduce-overhead') — reduce-overhead enables
            # CUDAGraphs which conflicts with our L2-normalize forward + NT-Xent
            # sim matrix (the buffer is reused before NT-Xent reads it). 'default'
            # still gets us ~20-30% speedup without that hazard.
            model = torch.compile(model, mode="default")
        except Exception as e:  # noqa: BLE001
            print(f"torch.compile failed ({e!r}); continuing without compile")

    history: list[dict] = []
    best_val = float("inf")
    best_epoch = -1
    patience = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n_steps = 0
        for raw in train_loader:
            # raw: (B, 3, 224, 224) uint8 — apply augment + dtype-convert on GPU
            raw = raw.to(device, non_blocking=True)
            v1 = gpu_augment(raw)
            v2 = gpu_augment(raw)
            if use_channels_last and device == "cuda":
                v1 = v1.contiguous(memory_format=torch.channels_last)
                v2 = v2.contiguous(memory_format=torch.channels_last)
            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    z1 = model(v1)
                    z2 = model(v2)
                    loss = nt_xent_loss(z1, z2, temperature=temperature)
            else:
                z1 = model(v1); z2 = model(v2)
                loss = nt_xent_loss(z1, z2, temperature=temperature)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            train_loss += loss.item()
            n_steps += 1
        sched.step()
        train_loss /= max(n_steps, 1)
        train_elapsed = time.time() - t0

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for raw in val_loader:
                raw = raw.to(device, non_blocking=True)
                v1 = gpu_augment(raw)
                v2 = gpu_augment(raw)
                if use_channels_last and device == "cuda":
                    v1 = v1.contiguous(memory_format=torch.channels_last)
                    v2 = v2.contiguous(memory_format=torch.channels_last)
                if amp_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        z1 = model(v1); z2 = model(v2)
                        loss = nt_xent_loss(z1, z2, temperature=temperature)
                else:
                    z1 = model(v1); z2 = model(v2)
                    loss = nt_xent_loss(z1, z2, temperature=temperature)
                val_loss += loss.item()
                n_val += 1
        val_loss /= max(n_val, 1)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "lr": optim.param_groups[0]["lr"], "epoch_sec": train_elapsed})
        print(f"epoch {epoch:03d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"lr={optim.param_groups[0]['lr']:.2e}  {train_elapsed:.1f}s")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        if val_loss < best_val - 0.001:
            best_val = val_loss
            best_epoch = epoch
            # Save uncompiled state_dict for portability
            inner = model._orig_mod if hasattr(model, "_orig_mod") else model
            best_state = {k: v.detach().cpu().clone() for k, v in inner.state_dict().items()}
            torch.save({
                "epoch": epoch,
                "state_dict": best_state,
                "val_loss": val_loss,
                "config": {
                    "embed_dim": HandshapeEncoder.EMBED_DIM,
                    "input_size": HandshapeEncoder.INPUT_SIZE,
                    "temperature": temperature,
                },
            }, run_dir / "best.pt")
            patience = 0
        else:
            patience += 1
            if early_stop_patience > 0 and patience >= early_stop_patience:
                print(f"early stop: no val improvement for {early_stop_patience} epochs "
                      f"(best epoch {best_epoch}, val_loss={best_val:.4f})")
                break

    return {
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "n_train": len(train_recs),
        "n_val": len(val_recs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops-root", type=Path, default=Path("data/hand_crops"))
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-channels-last", action="store_true")
    ap.add_argument("--early-stop-patience", type=int, default=8)
    args = ap.parse_args()

    res = train(
        crops_root=args.crops_root,
        run_dir=args.run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        num_workers=args.num_workers,
        val_frac=args.val_frac,
        seed=args.seed,
        use_bf16=not args.no_bf16,
        use_compile=not args.no_compile,
        use_channels_last=not args.no_channels_last,
        early_stop_patience=args.early_stop_patience,
    )
    print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
