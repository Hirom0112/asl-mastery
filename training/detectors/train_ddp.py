"""DDP multi-GPU fine-tuning for the hand detector (Option B: e.g. 8× H100).

RESUME/FINE-TUNE path only — not from-scratch. One process per GPU via
torch.multiprocessing.spawn inside a single Modal container (gpu="H100:8").
Reuses the proven single-GPU pieces: HandDetector, HandDetectorLoss,
HandBboxDataset (packed memmap), the GPU-aug chain, and _detection_counts.

Design decisions flagged for review (see the adversarial review notes):
  - Data is localized to /tmp ONCE by the entrypoint (single process) BEFORE
    spawn, so the 8 workers just mmap the same local file. No rank-0/barrier
    copy dance, no 8× copy.
  - Global batch = per_gpu_batch × world_size. Default 64×8 = 512, matching
    the single-GPU run, so the LR schedule needs no rescaling.
  - SyncBatchNorm: the detector is BatchNorm-heavy; at 64/GPU, per-GPU BN
    stats are noisy, so we convert to SyncBatchNorm.
  - Validation runs on rank 0 only over the full (small, 3219) val set, using
    the underlying module (not the DDP wrapper) to avoid collective expectations.
  - torch.compile is OFF by default under DDP (compile×DDP is a known bug
    surface; the model is small so the gain is modest).
  - Only rank 0 saves checkpoints / writes history; dist.barrier() fences it.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from training.detectors.hand_detector import HandDetector, count_parameters
from training.detectors.losses import HandDetectorLoss
from training.detectors.dataset import HandBboxDataset, load_manifest
from training.detectors.train import _collate_gpu_aug, _detection_counts
from training.detectors.gpu_augment import gpu_augment_batch, render_targets_gpu


def _ddp_setup(rank: int, world_size: int, master_port: str = "29512") -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = master_port
    # 30-min collective timeout so a dead rank fails the job instead of
    # hanging 8 H100s at ~$32/hr until the container timeout.
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        timeout=__import__("datetime").timedelta(minutes=30),
    )
    torch.cuda.set_device(rank)


def _ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _worker(rank: int, world_size: int, cfg: dict) -> None:
    try:
        _ddp_setup(rank, world_size, cfg["master_port"])
        device = f"cuda:{rank}"
        is_main = rank == 0

        train_records = load_manifest(Path(cfg["train_manifest"]))
        val_records = load_manifest(Path(cfg["val_manifest"]))
        train_ds = HandBboxDataset(train_records, gpu_aug_mode=True,
                                   packed_path=Path(cfg["packed_train_local"]))
        val_ds = HandBboxDataset(val_records, gpu_aug_mode=True,
                                 packed_path=Path(cfg["packed_val_local"]))

        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                     shuffle=True, drop_last=True)
        nw = cfg["num_workers"]
        loader = DataLoader(
            train_ds, batch_size=cfg["per_gpu_batch"], sampler=sampler,
            num_workers=nw, collate_fn=_collate_gpu_aug, pin_memory=True,
            persistent_workers=nw > 0, drop_last=True,
            **({"prefetch_factor": 4} if nw > 0 else {}),
        )

        model = HandDetector().to(device)
        if cfg.get("resume_from"):
            ckpt = torch.load(cfg["resume_from"], map_location=device,
                              weights_only=False)
            sd = ckpt.get("model", ckpt)
            model.load_state_dict(sd)
            if is_main:
                print(f"[rank0] resumed from {cfg['resume_from']} "
                      f"(epoch {ckpt.get('epoch','?')}, "
                      f"val {ckpt.get('best_val_loss','?')})")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank])
        loss_fn = HandDetectorLoss().to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"])
        warm = cfg["warmup_epochs"]
        if warm > 0:
            w = torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, total_iters=warm)
            c = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(cfg["epochs"] - warm, 1), eta_min=1e-5)
            sched = torch.optim.lr_scheduler.SequentialLR(opt, [w, c], [warm])
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

        run_dir = Path(cfg["run_dir"])
        if is_main:
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"[rank0] DDP world_size={world_size}  per_gpu_batch="
                  f"{cfg['per_gpu_batch']}  global_batch="
                  f"{cfg['per_gpu_batch']*world_size}  params={count_parameters(model):,}")

        history = []
        best_val = float("inf")
        amp = torch.bfloat16
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            sampler.set_epoch(epoch)  # reshuffle differently each epoch
            t0 = time.time()
            tot, nb = 0.0, 0
            for images, bboxes, mask in loader:
                images = images.to(device, non_blocking=True)
                bboxes = bboxes.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                images, bboxes = gpu_augment_batch(images, bboxes)
                targets = render_targets_gpu(bboxes, mask, input_size=320, stride=4)
                with torch.autocast(device_type="cuda", dtype=amp):
                    out = model(images)
                    losses = loss_fn(out, targets)
                if not torch.isfinite(losses["total"]):
                    raise RuntimeError(f"non-finite loss rank{rank} ep{epoch}")
                opt.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += losses["total"].item(); nb += 1
            sched.step()

            # Validation + checkpointing on rank 0 only.
            if is_main:
                m = model.module
                m.eval()
                vt = 0.0; nvb = 0
                det = {k: 0.0 for k in ("recall_hits", "gt_total",
                                        "matched_peaks", "peak_total", "conf_at_gt_sum")}
                val_loader = DataLoader(val_ds, batch_size=cfg["per_gpu_batch"],
                                        shuffle=False, num_workers=nw,
                                        collate_fn=_collate_gpu_aug)
                with torch.no_grad():
                    for images, bboxes, mask in val_loader:
                        images = images.to(device); bboxes = bboxes.to(device); mask = mask.to(device)
                        targets = render_targets_gpu(bboxes, mask, 320, 4)
                        with torch.autocast(device_type="cuda", dtype=amp):
                            out = m(images)
                            l = loss_fn(out, targets)
                        vt += l["total"].item(); nvb += 1
                        c = _detection_counts(out["heatmap"], targets["center_mask"])
                        for k in det:
                            det[k] += c[k]
                val_loss = vt / max(nvb, 1)
                rec = det["recall_hits"] / max(det["gt_total"], 1.0)
                prec = det["matched_peaks"] / max(det["peak_total"], 1.0)
                conf = det["conf_at_gt_sum"] / max(det["gt_total"], 1.0)
                dt = time.time() - t0
                print(f"epoch {epoch:03d}/{cfg['epochs']}  train_loss={tot/max(nb,1):.4f}  "
                      f"val_loss={val_loss:.4f}  det_recall={rec:.3f} det_prec={prec:.3f} "
                      f"conf@gt={conf:.3f}  {dt:.1f}s", flush=True)
                history.append({"epoch": epoch, "train_loss": tot/max(nb,1),
                                "val_loss": val_loss, "det_recall": rec,
                                "det_precision": prec, "conf_at_gt": conf, "sec": dt})
                (run_dir / "history.json").write_text(json.dumps(history, indent=2))
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({"model": m.state_dict(), "epoch": epoch,
                                "best_val_loss": best_val}, run_dir / "best.pt")
            dist.barrier()  # keep ranks in lockstep across the rank-0 val phase

        if is_main:
            torch.save({"model": model.module.state_dict(),
                        "epoch": cfg["epochs"]}, run_dir / "last.pt")
            print(f"[rank0] done. best_val_loss={best_val:.4f}", flush=True)
    finally:
        _ddp_cleanup()


def launch(cfg: dict, world_size: int) -> None:
    """Spawn `world_size` workers (call from the Modal entrypoint, after the
    packed data has been localized to /tmp)."""
    mp.spawn(_worker, args=(world_size, cfg), nprocs=world_size, join=True)
