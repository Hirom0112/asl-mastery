"""Linear probe of the trained HandshapeEncoder.

Trains a single linear layer on top of FROZEN encoder embeddings to predict
sign_id. Top-1 accuracy on a held-out clip-disjoint split tells us whether
the encoder learned representations useful for downstream sign classification.

This is the missing quantitative signal that NT-Xent val_loss can't provide.
Run on the saved encoder ckpt BEFORE spending $0.80 on the v4 trajectory extract.

Decision gate:
  random chance ≈ 1/79 ≈ 1.3%
  < 5% top-1   → encoder is broken; debug before extract
  5-15%        → marginal; v4 retrain may not move the 11% ceiling
  15-25%       → real signal; proceed to extract+retrain with confidence
  25%+         → strong; the encoder is the architectural unlock we needed

Usage:
    training/.venv/bin/python -m scripts.linear_probe_encoder \\
        --encoder-ckpt artifacts/ckpts/handshape_v0d_best.pt \\
        --crops-root data/hand_crops \\
        --n-per-sign 50 \\
        --epochs 30
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.io as tvio
from torch.utils.data import DataLoader, Dataset

from training.detectors.handshape_encoder import HandshapeEncoder, count_parameters


class ProbeDataset(Dataset):
    """Returns (uint8_crop_224, sign_idx). Used for both train and val.
    Augmentation deliberately MINIMAL — we want to measure embedding quality,
    not augmentation-robustness."""

    def __init__(self, records: list[dict], sign_to_idx: dict[str, int]):
        self.records = [r for r in records if r["sign_id"] in sign_to_idx]
        self.sign_to_idx = sign_to_idx

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        img = tvio.read_image(r["crop_path"], mode=tvio.ImageReadMode.RGB)  # uint8
        return img, self.sign_to_idx[r["sign_id"]]


def _split_by_clip(records: list[dict], val_frac: float, seed: int,
                   n_per_sign: int) -> tuple[list[dict], list[dict]]:
    """Subsample to n_per_sign crops per sign, then clip-disjoint train/val split."""
    by_sign = defaultdict(list)
    for r in records:
        by_sign[r["sign_id"]].append(r)
    rnd = random.Random(seed)
    out_train: list[dict] = []
    out_val: list[dict] = []
    for sign, recs in by_sign.items():
        # Group by clip_id so all crops of one clip stay together
        by_clip = defaultdict(list)
        for r in recs:
            by_clip[r["clip_id"]].append(r)
        clips = list(by_clip.keys())
        rnd.shuffle(clips)
        n_val = max(1, int(len(clips) * val_frac))
        val_clips = set(clips[:n_val])
        # Cap at n_per_sign by picking up to that many crops, train first
        train_recs = []
        val_recs = []
        for cid, crops in by_clip.items():
            (val_recs if cid in val_clips else train_recs).extend(crops)
        rnd.shuffle(train_recs)
        rnd.shuffle(val_recs)
        out_train.extend(train_recs[:n_per_sign])
        out_val.extend(val_recs[: max(5, n_per_sign // 5)])
    return out_train, out_val


@torch.no_grad()
def _embed_all(encoder: HandshapeEncoder, loader: DataLoader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed all crops in the loader once. Returns (N, D) embeddings and (N,) labels."""
    encoder.eval()
    all_z, all_y = [], []
    for img, y in loader:
        img = img.to(device, non_blocking=True).float() / 255.0
        z = encoder.features(img)  # use pre-projection features (512D), richer signal
        all_z.append(z.cpu())
        all_y.append(y)
    return torch.cat(all_z, 0), torch.cat(all_y, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-ckpt", type=Path, required=True)
    ap.add_argument("--crops-root", type=Path, default=Path("data/hand_crops"))
    ap.add_argument("--n-per-sign", type=int, default=50)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip-pool", action="store_true",
                    help="Mean-pool embeddings across all crops of one clip before "
                         "training the linear head. Tests whether the encoder gives "
                         "useful signal AT THE CLIP LEVEL (which is what the "
                         "downstream TCN actually consumes via temporal pooling).")
    ap.add_argument("--n-crops-per-clip", type=int, default=5,
                    help="Used only with --clip-pool: how many crops per clip to "
                         "average (lower = faster probe, higher = better pooling).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    # Load encoder
    encoder = HandshapeEncoder().to(device).eval()
    ckpt = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["state_dict"])
    print(f"loaded encoder from {args.encoder_ckpt}; params: {count_parameters(encoder):,}")

    # Load + subsample crops
    idx_path = args.crops_root / "index.json"
    records = json.loads(idx_path.read_text())["records"]
    signs = sorted({r["sign_id"] for r in records})
    sign_to_idx = {s: i for i, s in enumerate(signs)}
    n_classes = len(signs)
    chance = 1.0 / n_classes
    print(f"signs: {n_classes}  (chance top-1 = {chance * 100:.2f}%)")

    if args.clip_pool:
        # Different sampling: pick N_PER_SIGN distinct clips per sign, then
        # cap each clip at N_CROPS_PER_CLIP. Embeddings will be pooled by clip.
        by_sign_clip: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in records:
            by_sign_clip[(r["sign_id"], r["clip_id"])].append(r)
        # Group: sign → list of (clip_id, list_of_crops)
        by_sign: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
        for (sign, clip_id), crops in by_sign_clip.items():
            by_sign[sign].append((clip_id, crops))
        rnd = random.Random(args.seed)
        out_train: list[dict] = []
        out_val: list[dict] = []
        for sign, clip_groups in by_sign.items():
            rnd.shuffle(clip_groups)
            n_val_clips = max(1, int(len(clip_groups) * args.val_frac))
            val_clips = clip_groups[:n_val_clips]
            train_clips = clip_groups[n_val_clips:n_val_clips + args.n_per_sign]
            for cid, crops in train_clips:
                sub = crops[:args.n_crops_per_clip]
                for r in sub:
                    out_train.append(r)
            for cid, crops in val_clips[:max(5, args.n_per_sign // 5)]:
                sub = crops[:args.n_crops_per_clip]
                for r in sub:
                    out_val.append(r)
        train_recs, val_recs = out_train, out_val
        n_train_clips = len({r["clip_id"] for r in train_recs})
        n_val_clips = len({r["clip_id"] for r in val_recs})
        print(f"CLIP-POOL mode")
        print(f"split: train={len(train_recs)} crops ({n_train_clips} clips)  "
              f"val={len(val_recs)} crops ({n_val_clips} clips)")
    else:
        train_recs, val_recs = _split_by_clip(records, args.val_frac, args.seed, args.n_per_sign)
        print(f"split: train={len(train_recs)}  val={len(val_recs)}")
    print(f"train per-sign sample: {Counter(r['sign_id'] for r in train_recs).most_common(5)}")

    train_ds = ProbeDataset(train_recs, sign_to_idx)
    val_ds = ProbeDataset(val_recs, sign_to_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device == "cuda"))

    print("\nembedding train set (frozen encoder, single pass)...")
    t0 = time.time()
    Z_train, Y_train = _embed_all(encoder, train_loader, device)
    print(f"  train embed shape: {Z_train.shape} ({time.time() - t0:.1f}s)")
    t0 = time.time()
    Z_val, Y_val = _embed_all(encoder, val_loader, device)
    print(f"  val embed shape:   {Z_val.shape} ({time.time() - t0:.1f}s)")

    # Clip-pool: average per-clip embeddings before training the linear head.
    # This is the realistic ask of the encoder for our downstream classifier.
    if args.clip_pool:
        def _pool(Z: torch.Tensor, Y: torch.Tensor, recs: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
            assert Z.size(0) == len(recs)
            by_clip: dict[str, list[int]] = defaultdict(list)
            for i, r in enumerate(recs):
                by_clip[r["clip_id"]].append(i)
            pooled_Z, pooled_Y = [], []
            for cid, idxs in by_clip.items():
                pooled_Z.append(Z[idxs].mean(dim=0))
                pooled_Y.append(Y[idxs[0]])  # all crops in a clip share sign_id
            return torch.stack(pooled_Z), torch.stack(pooled_Y)
        Z_train, Y_train = _pool(Z_train, Y_train, train_recs)
        Z_val, Y_val = _pool(Z_val, Y_val, val_recs)
        print(f"  after clip-pool: train={Z_train.shape}, val={Z_val.shape}")

    # L2-normalize the features (linear probe convention)
    Z_train = F.normalize(Z_train, dim=-1).to(device)
    Z_val = F.normalize(Z_val, dim=-1).to(device)
    Y_train = Y_train.to(device)
    Y_val = Y_val.to(device)

    # Train a single linear layer
    head = nn.Linear(Z_train.size(1), n_classes).to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    bs = 512
    best_top1 = 0.0
    best_top3 = 0.0
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(Z_train.size(0), device=device)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, Z_train.size(0), bs):
            idxs = perm[i:i + bs]
            logits = head(Z_train[idxs])
            loss = F.cross_entropy(logits, Y_train[idxs])
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n_batches += 1
        sched.step()

        # Eval
        head.eval()
        with torch.no_grad():
            logits = head(Z_val)
            preds = logits.argmax(-1)
            top1 = (preds == Y_val).float().mean().item()
            top3_preds = logits.topk(3, -1).indices
            top3 = (top3_preds == Y_val.unsqueeze(-1)).any(-1).float().mean().item()
        if top1 > best_top1:
            best_top1 = top1
            best_top3 = top3
            best_epoch = epoch
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  e{epoch:03d}  train_loss={total_loss / n_batches:.3f}  "
                  f"val_top1={top1 * 100:.2f}%  val_top3={top3 * 100:.2f}%  "
                  f"(best e{best_epoch}: top1={best_top1 * 100:.2f}%)")

    print(f"\n=== LINEAR PROBE RESULT ===")
    print(f"best val top-1: {best_top1 * 100:.2f}%  (chance {chance * 100:.2f}%)")
    print(f"best val top-3: {best_top3 * 100:.2f}%")
    print(f"best epoch: {best_epoch}")
    print()
    if best_top1 < 0.05:
        verdict = "BROKEN — debug encoder before extract"
    elif best_top1 < 0.15:
        verdict = "MARGINAL — extract may not move ceiling much"
    elif best_top1 < 0.25:
        verdict = "GOOD — proceed with confidence"
    else:
        verdict = "STRONG — encoder is the architectural unlock"
    print(f"verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
