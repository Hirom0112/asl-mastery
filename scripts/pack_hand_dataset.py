"""P3 speed optimization — pre-decode a hand_bbox manifest into a uint8 memmap.

The hand detector is a 2.3M-param model; the training bottleneck is decoding
61k–181k JPEGs from the Modal volume every epoch. This packs them ONCE into a
single (N,3,320,320) uint8 memmap + a bbox sidecar, so every subsequent run
(P3a tuning, P3b, ablations) memmaps it with zero decode and zero RAM.

Writes via np.memmap (not np.save) so we never hold the full ~19–55 GB array
in RAM. Read back by HandBboxDataset(packed_path=<meta.json>).

Run as a cheap CPU job (local if images are present, else a Modal CPU
function pointed at the volume manifest). One-time cost ~10–20 min.

Usage:
    python -m scripts.pack_hand_dataset \
        --manifest data/labeled_frames/hand_bbox/train_with_negatives.json \
        --out-base data/labeled_frames/hand_bbox/packed/train \
        --workers 16
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torchvision.io as tvio

from training.detectors.dataset import load_manifest

INPUT_SIZE = 320


def pack(manifest: Path, out_base: Path, workers: int = 16) -> dict:
    records = load_manifest(manifest)
    n = len(records)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    dat_path = out_base.with_suffix(".dat")
    meta_path = out_base.with_suffix(".meta.json")

    shape = (n, 3, INPUT_SIZE, INPUT_SIZE)
    mm = np.memmap(dat_path, dtype=np.uint8, mode="w+", shape=shape)
    bboxes_out: list[list[list[float]]] = [None] * n  # type: ignore

    def _load(i: int):
        rec = records[i]
        try:
            img = tvio.read_image(str(rec.image_path), mode=tvio.ImageReadMode.RGB)
        except Exception:
            # Unreadable image → black frame, no boxes (treated as negative).
            return i, np.zeros((3, INPUT_SIZE, INPUT_SIZE), np.uint8), []
        _, ah, aw = img.shape
        src_w = rec.width if rec.width else aw
        src_h = rec.height if rec.height else ah
        resized = torch.nn.functional.interpolate(
            img.unsqueeze(0).float(), size=(INPUT_SIZE, INPUT_SIZE),
            mode="bilinear", align_corners=False,
        ).squeeze(0).clamp(0, 255).to(torch.uint8).numpy()
        sx = INPUT_SIZE / src_w
        sy = INPUT_SIZE / src_h
        scaled = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in rec.bboxes]
        return i, resized, scaled

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_load, i) for i in range(n)]):
            i, arr, boxes = fut.result()
            mm[i] = arr
            bboxes_out[i] = boxes
            done += 1
            if done % 5000 == 0:
                print(f"  packed {done}/{n} ({done / (time.time() - t0):.0f} img/s)")
    mm.flush()
    del mm

    meta = {
        "dat": dat_path.name,
        "dtype": "uint8",
        "shape": list(shape),
        "bboxes": bboxes_out,
        "source_manifest": str(manifest),
    }
    meta_path.write_text(json.dumps(meta))
    gb = np.prod(shape) / 1e9
    print(f"packed {n} images in {time.time() - t0:.1f}s → {dat_path} "
          f"({gb:.1f} GB) + {meta_path}")
    return {"n": n, "dat": str(dat_path), "meta": str(meta_path), "gb": gb}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-base", type=Path, required=True,
                    help="Path prefix; writes <base>.dat + <base>.meta.json")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    print(json.dumps(pack(args.manifest, args.out_base, args.workers), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
