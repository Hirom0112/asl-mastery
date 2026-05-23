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
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.detectors.augment import default_train_augment
from training.detectors.dataset import HandBboxDataset, load_manifest
from training.detectors.hand_detector import HandDetector, count_parameters
from training.detectors.losses import HandDetectorLoss


def _collate(batch):
    """Default collate for the CPU-aug path: (image, target_dict)."""
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    stacked = {k: torch.stack([t[k] for t in targets], dim=0) for k in targets[0]}
    return images, stacked


def _collate_gpu_aug(batch):
    """Collate for the GPU-aug path. Each sample is (image, bbox_padded,
    bbox_mask) — no targets yet, the loop renders them on GPU."""
    images, bboxes, masks = zip(*batch)
    images = torch.stack(images, dim=0)
    bboxes = torch.stack(bboxes, dim=0)
    masks = torch.stack(masks, dim=0)
    return images, bboxes, masks


@torch.no_grad()
def _detection_counts(pred_logits: torch.Tensor, center_mask: torch.Tensor,
                      threshold: float = 0.2, radius: int = 2) -> dict:
    """Frame-level detection quality from the CenterNet heatmap, computed
    against the GT center_mask (1 at true hand centers).

    Returns raw counts so the caller can aggregate a global recall/precision
    across the whole val set (not a mean-of-batch-means). Also returns the
    summed predicted confidence AT GT centers — the direct calibration
    number: the decode comment notes 'max heatmap prob ~0.27 on real ASL
    frames', so watching this climb is how we know retraining fixed the
    under-confidence that drops 40% of frames.
    """
    import torch.nn.functional as F
    prob = torch.sigmoid(pred_logits.float())            # (B,1,H,W)
    pooled = F.max_pool2d(prob, 3, stride=1, padding=1)
    is_peak = (prob == pooled) & (prob >= threshold)      # local maxima
    k = 2 * radius + 1
    # A GT center is "recalled" if any peak lies within `radius` cells.
    peak_near = F.max_pool2d(is_peak.float(), k, stride=1, padding=radius) > 0
    gt_near = F.max_pool2d(center_mask.float(), k, stride=1, padding=radius) > 0
    gt = center_mask > 0.5
    return {
        "recall_hits": float((gt & peak_near).sum().item()),
        "gt_total": float(gt.sum().item()),
        "matched_peaks": float((is_peak & gt_near).sum().item()),
        "peak_total": float(is_peak.sum().item()),
        "conf_at_gt_sum": float((prob * gt.float()).sum().item()),
    }


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
    resume_from: Path | None = None,
    cache_in_memory: bool = False,
    use_bf16: bool = False,
    use_compile: bool = False,
    use_channels_last: bool = False,
    warmup_epochs: int = 0,
    use_ema: bool = False,
    ema_decay: float = 0.999,
    early_stop_patience: int = 0,
    early_stop_delta: float = 0.005,
    gpu_aug: bool = False,
    packed_train_path: Path | None = None,
    packed_val_path: Path | None = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_manifest(train_manifest)
    val_records = load_manifest(val_manifest)
    print(f"train frames: {len(train_records)}  val frames: {len(val_records)}")

    if cache_in_memory:
        print(f"building in-memory image cache (this is a one-time cost)...")
    train_ds = HandBboxDataset(
        train_records,
        augment=None if gpu_aug else default_train_augment,
        cache_in_memory=cache_in_memory and packed_train_path is None,
        gpu_aug_mode=gpu_aug,
        packed_path=packed_train_path,
    )
    val_ds = HandBboxDataset(
        val_records, augment=None,
        cache_in_memory=cache_in_memory and packed_val_path is None,
        gpu_aug_mode=gpu_aug,
        packed_path=packed_val_path,
    )

    # The RAM cache is now ONE shared-memory uint8 mega-tensor (see
    # HandBboxDataset._build_cache). Workers fork after share_memory_(),
    # so they all read into the same buffer — no per-worker re-cache,
    # zero-copy. We can finally use the workers we paid for. The earlier
    # `eff_workers=0 if cache_in_memory else num_workers` clamp turned
    # `cache_in_memory=True` into a single-threaded pipeline and was the
    # root cause of the 32 ms/sample plateau diagnosed at epoch 30.
    eff_workers = num_workers
    if cache_in_memory and os.environ.get("CACHE_NO_SHM") == "1" and eff_workers > 0:
        print(f"CACHE_NO_SHM=1 + cache_in_memory=True → "
              f"forcing num_workers from {eff_workers} to 0 "
              f"(no shared memory means workers would COW the cache)")
        eff_workers = 0
    collate = _collate_gpu_aug if gpu_aug else _collate
    loader_kwargs = dict(
        num_workers=eff_workers,
        collate_fn=collate,
        pin_memory=device == "cuda",
        persistent_workers=eff_workers > 0,
    )
    if eff_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        **loader_kwargs,
    )

    model = HandDetector().to(device)
    if use_channels_last and device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    print(f"hand detector params: {count_parameters(model):,}")
    loss_fn = HandDetectorLoss().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Warmup + cosine: linear warmup for warmup_epochs, then cosine to 1e-5.
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-5)
        sched = torch.optim.lr_scheduler.SequentialLR(
            optim, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    if use_compile and device == "cuda":
        print("torch.compile() enabled")
        model = torch.compile(model)

    start_epoch = 1
    history = []
    best_val = float("inf")
    cum_epoch_base = 0  # cumulative epoch counter for cross-resume bookkeeping

    if resume_from is not None and resume_from.exists():
        print(f"resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        # torch.compile() wraps the model in an OptimizedModule whose keys are
        # prefixed with `_orig_mod.`; the checkpoint has bare keys. Load into
        # the underlying module so resume works with or without compile.
        (getattr(model, "_orig_mod", model)).load_state_dict(ckpt["model"])
        # The previous run's best.pt may not include optim/sched state. We
        # accept that loss; the cosine schedule is rebuilt from epoch=1
        # because we're starting a fresh run with new hyperparams (bigger
        # batch / different GPU), so re-baselining the schedule is intended.
        # `start_epoch` controls the *display* counter only — total epochs
        # to run is still `epochs`. Cumulative epoch # across resumes is
        # tracked separately and persisted in history.json so we never
        # lose visibility again (this was a real problem at the 30-epoch
        # plateau audit: history was per-run, not cumulative).
        prev_epoch = int(ckpt.get("epoch", 0))
        cum_epoch_base = int(ckpt.get("cum_epoch", prev_epoch))
        print(f"  loaded model weights from epoch {prev_epoch} (cumulative {cum_epoch_base})")
        # NOTE: do NOT inherit the prior best_val. On a fine-tune (new aug /
        # added negatives / changed objective), the new run's val may never
        # beat the old baseline, so best.pt would never save (the
        # "no best.pt produced" crash). Keep best_val=inf so best.pt always
        # reflects the best epoch of THIS run.
        # Carry prior history forward if it lives next to the checkpoint
        prior_history_path = resume_from.parent / "history.json"
        if prior_history_path.exists():
            try:
                history = json.loads(prior_history_path.read_text())
                print(f"  carried {len(history)} prior history entries")
            except Exception:
                history = []

    # Optional EMA of model weights — small, steady lift on noisy val curves
    # near plateau (YOLOX recipe). Decay 0.999 is standard.
    ema_model = None
    if use_ema:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        # AveragedModel handles the .to(device) and the channels_last format
        # via copy of the underlying tensors.
        ema_base = getattr(model, "_orig_mod", model)
        ema_model = AveragedModel(
            ema_base, multi_avg_fn=get_ema_multi_avg_fn(ema_decay)
        ).to(device)
        if use_channels_last and device == "cuda":
            ema_model = ema_model.to(memory_format=torch.channels_last)
        print(f"EMA enabled with decay={ema_decay}")

    # Plateau-based early stop. Disabled when patience=0.
    no_improve_count = 0
    es_threshold = early_stop_delta
    es_patience = early_stop_patience
    early_stopped_at = None

    amp_dtype = torch.bfloat16 if use_bf16 else None

    # When gpu_aug is on, lazy-import to avoid CPU-side dependency in
    # workers and to keep the module untouched when the flag is off.
    if gpu_aug:
        from training.detectors.gpu_augment import (
            gpu_augment_batch, render_targets_gpu,
        )
    else:
        gpu_augment_batch = None
        render_targets_gpu = None

    def _prep_batch(batch, train_mode: bool):
        """Move to device, optionally apply GPU aug + render targets.
        Returns (images, targets_dict) in both code paths.
        """
        if gpu_aug:
            images, bboxes, mask = batch
            images = images.to(device, non_blocking=True)
            bboxes = bboxes.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            if train_mode:
                images, bboxes = gpu_augment_batch(images, bboxes)
            if use_channels_last and device == "cuda":
                images = images.to(memory_format=torch.channels_last)
            targets = render_targets_gpu(bboxes, mask, input_size=320, stride=4)
        else:
            images, targets = batch
            images = images.to(device, non_blocking=True)
            if use_channels_last and device == "cuda":
                images = images.to(memory_format=torch.channels_last)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        return images, targets

    heartbeat_every = int(os.environ.get("HEARTBEAT_EVERY", "50"))
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.time()
        train_total = 0.0
        n_batches = 0
        last_hb_t = time.time()
        last_hb_batches = 0
        try:
            est_batches = len(train_loader)
        except TypeError:
            est_batches = None
        for batch in train_loader:
            images, targets = _prep_batch(batch, train_mode=True)
            if amp_dtype is not None and device == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(images)
                    losses = loss_fn(outputs, targets)
            else:
                outputs = model(images)
                losses = loss_fn(outputs, targets)
            # Abort-on-NaN: any non-finite loss means hours of wasted GPU
            # compute if we don't catch it now. Better to crash loudly than
            # to discover the NaN at epoch 80 (see session 16 deep-dive).
            if not torch.isfinite(losses["total"]).item():
                hm = losses.get("heatmap")
                sz = losses.get("size")
                raise RuntimeError(
                    f"non-finite loss at epoch={epoch} batch={n_batches}: "
                    f"total={losses['total'].item()} "
                    f"heatmap={hm.item() if hm is not None else 'n/a'} "
                    f"size={sz.item() if sz is not None else 'n/a'}. "
                    "Most common cause: bf16 + numerically unsafe loss "
                    "stanza. See losses.py for autocast-disable patterns."
                )
            optim.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            if ema_model is not None:
                ema_model.update_parameters(getattr(model, "_orig_mod", model))
            train_total += losses["total"].item()
            n_batches += 1
            if heartbeat_every > 0 and n_batches % heartbeat_every == 0:
                now = time.time()
                dt = max(now - last_hb_t, 1e-6)
                bps = (n_batches - last_hb_batches) / dt
                tot_str = f"/{est_batches}" if est_batches else ""
                eta_s = ((est_batches - n_batches) / bps) if (est_batches and bps > 0) else None
                eta_str = f"  eta_epoch={eta_s/60:.1f}m" if eta_s is not None else ""
                print(
                    f"  e{epoch} b{n_batches}{tot_str}  "
                    f"loss={losses['total'].item():.4f}  "
                    f"{bps:.2f} batch/s{eta_str}",
                    flush=True,
                )
                last_hb_t = now
                last_hb_batches = n_batches
        sched.step()
        train_loss = train_total / max(n_batches, 1)

        # Pick the eval model: EMA if enabled, else the live model.
        eval_model = ema_model if ema_model is not None else model
        eval_model.eval()
        val_total = 0.0
        val_total_ema = 0.0
        n_val = 0
        det = {"recall_hits": 0.0, "gt_total": 0.0, "matched_peaks": 0.0,
               "peak_total": 0.0, "conf_at_gt_sum": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                images, targets = _prep_batch(batch, train_mode=False)
                if amp_dtype is not None and device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        outputs = eval_model(images)
                        losses = loss_fn(outputs, targets)
                else:
                    outputs = eval_model(images)
                    losses = loss_fn(outputs, targets)
                val_total += losses["total"].item()
                n_val += 1
                c = _detection_counts(outputs["heatmap"], targets["center_mask"])
                for k_ in det:
                    det[k_] += c[k_]
        val_loss = val_total / max(n_val, 1)
        # Global detection metrics (the P3 gate): recall = fraction of true
        # hand centers found; conf_at_gt = mean predicted confidence where a
        # hand truly is (calibration; baseline ~0.27 → target >0.5).
        det_recall = det["recall_hits"] / max(det["gt_total"], 1.0)
        det_prec = det["matched_peaks"] / max(det["peak_total"], 1.0)
        det_conf = det["conf_at_gt_sum"] / max(det["gt_total"], 1.0)

        elapsed = time.time() - t0
        cum_epoch = cum_epoch_base + epoch
        msg = (
            f"epoch {epoch:03d}/{epochs} (cum {cum_epoch})  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"det_recall={det_recall:.3f} det_prec={det_prec:.3f} "
            f"conf@gt={det_conf:.3f}  "
            f"lr={sched.get_last_lr()[0]:.2e}  {elapsed:.1f}s"
            + ("  [EMA-eval]" if ema_model is not None else "")
        )
        print(msg)
        history.append({
            "epoch": epoch,
            "cum_epoch": cum_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "det_recall": det_recall,
            "det_precision": det_prec,
            "conf_at_gt": det_conf,
            "elapsed_s": elapsed,
            "ema": ema_model is not None,
        })
        # Write history every epoch so a crash doesn't lose the trajectory.
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        # Save checkpoints. For EMA runs, we save BOTH the live model
        # ("best.pt") and the EMA-averaged copy ("best_ema.pt") so
        # downstream consumers can pick. `cum_epoch` is recorded so
        # future resumes know the true cumulative count.
        save_model = getattr(model, "_orig_mod", model)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": save_model.state_dict(),
                "epoch": epoch,
                "cum_epoch": cum_epoch,
                "best_val_loss": best_val,
            }, run_dir / "best.pt")
            if ema_model is not None:
                torch.save({
                    "model": ema_model.module.state_dict(),
                    "epoch": epoch,
                    "cum_epoch": cum_epoch,
                    "best_val_loss": best_val,
                    "ema_decay": ema_decay,
                }, run_dir / "best_ema.pt")

        # Plateau-based early stop. Counts epochs that fail to improve
        # by >= es_threshold. Resets on a real gain. Skipped when
        # patience=0 (the legacy behavior).
        if es_patience > 0:
            if val_loss < best_val + es_threshold - 1e-9:
                # `best_val` was updated above iff val_loss improved at all;
                # require ≥ threshold improvement vs the running best to
                # count as a non-plateau step.
                if val_loss <= best_val:
                    # actual best — improvement counted via best_val update path
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            if no_improve_count >= es_patience:
                print(f"  early stop: no improvement ≥ {es_threshold} "
                      f"for {es_patience} epochs")
                early_stopped_at = epoch
                break

    save_model = getattr(model, "_orig_mod", model)
    final_cum = cum_epoch_base + (early_stopped_at or epochs)
    torch.save({
        "model": save_model.state_dict(),
        "epoch": early_stopped_at or epochs,
        "cum_epoch": final_cum,
        "best_val_loss": best_val,
    }, run_dir / "last.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {
        "best_val_loss": best_val,
        "epochs": early_stopped_at or epochs,
        "cum_epoch": final_cum,
        "early_stopped_at": early_stopped_at,
        "run_dir": str(run_dir),
    }


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
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Load model weights from a previous best.pt and continue training. "
                        "Optimizer + LR schedule are re-baselined; the goal is to "
                        "continue learning with new hyperparameters (e.g. bigger batch).")
    p.add_argument("--cache-in-memory", action="store_true",
                   help="Pre-decode all images into RAM at startup. Removes "
                        "disk I/O as the per-step bottleneck.")
    p.add_argument("--bf16", action="store_true",
                   help="bf16 autocast (H100/A100). ~1.5-2x speedup.")
    p.add_argument("--compile", dest="use_compile", action="store_true",
                   help="torch.compile() the model. ~1.2-1.5x extra speedup.")
    p.add_argument("--channels-last", action="store_true",
                   help="channels-last memory format. ~1.2-1.5x extra on H100.")
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="Linear LR warmup epochs before cosine annealing.")
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
        resume_from=args.resume_from,
        cache_in_memory=args.cache_in_memory,
        use_bf16=args.bf16,
        use_compile=args.use_compile,
        use_channels_last=args.channels_last,
        warmup_epochs=args.warmup_epochs,
    )
    print(result)
