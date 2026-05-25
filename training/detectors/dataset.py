"""Dataset class for hand-bbox labeled frames.

Reads a flat JSON manifest produced by /labeling (CVAT or LabelStudio
export normalized to a project-internal schema), loads image + bboxes,
renders the CenterNet-style training targets (Gaussian heatmap +
size map + binary center mask) at the model's output resolution.

No torchvision.models. Only torchvision.transforms / .io for image
decode and tensor conversion.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torchvision.io as tvio
from torch.utils.data import Dataset


@dataclass
class FrameRecord:
    """One labeled frame in the hand-bbox dataset.

    Coordinates are in INPUT-IMAGE pixel space (the image as it sits
    on disk). Augmentation + resize to 320 happens in __getitem__.
    """

    image_path: Path
    width: int
    height: int
    # Each bbox: [x_min, y_min, x_max, y_max] in input-image pixels.
    bboxes: list[list[float]]


def load_manifest(manifest_path: Path) -> list[FrameRecord]:
    """Project-internal manifest schema:

    {
      "version": 1,
      "task": "hand_bbox",
      "items": [
        {
          "image_path": "data/labeled_frames/hand_bbox/wlasl_00001_f034.jpg",
          "width": 1280,
          "height": 720,
          "bboxes": [[412.0, 198.5, 562.3, 348.1], [...]]
        },
        ...
      ]
    }

    All paths in items[].image_path are interpreted relative to the
    repository root.
    """
    raw = json.loads(manifest_path.read_text())
    assert raw["version"] == 1, raw["version"]
    # Same bbox schema is used for hand_bbox and face_bbox (single-class
    # detector trained on either via FaceDetector/HandDetector — same
    # architecture, different labels).
    assert raw["task"] in {"hand_bbox", "face_bbox"}, raw["task"]

    repo_root = manifest_path.resolve().parent
    while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
        repo_root = repo_root.parent

    records: list[FrameRecord] = []
    n_clipped = 0
    n_skipped_no_dims = 0
    for item in raw["items"]:
        w = int(item["width"])
        h = int(item["height"])
        # CRITICAL: the Modal-built CMU manifest has width=0/height=0 for
        # all CMU records (upstream bug in build_manifests). The dataset
        # `__getitem__` falls back to actual image dims when rec.width=0,
        # so older training ran fine. We must do the same here — skipping
        # bbox-validation when manifest dims are missing — otherwise
        # the OOB clip turns into a record-dropping bug.
        if w <= 0 or h <= 0:
            n_skipped_no_dims += 1
            records.append(
                FrameRecord(
                    image_path=repo_root / item["image_path"],
                    width=w,
                    height=h,
                    bboxes=[[float(x) for x in b] for b in item["bboxes"]],
                )
            )
            continue
        cleaned = []
        for b in item["bboxes"]:
            x0, y0, x1, y1 = (float(x) for x in b)
            cx0 = max(0.0, min(x0, float(w)))
            cy0 = max(0.0, min(y0, float(h)))
            cx1 = max(0.0, min(x1, float(w)))
            cy1 = max(0.0, min(y1, float(h)))
            if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
                n_clipped += 1
            if cx1 - cx0 < 1.0 or cy1 - cy0 < 1.0:
                continue  # post-clip degenerate — drop this bbox
            cleaned.append([cx0, cy0, cx1, cy1])
        if not cleaned and item["bboxes"]:
            # Had boxes but all clipped to degenerate → drop.
            # An item that was ALREADY empty (item["bboxes"] == []) is an
            # intentional hard NEGATIVE (P3) and must be kept.
            continue
        records.append(
            FrameRecord(
                image_path=repo_root / item["image_path"],
                width=w,
                height=h,
                bboxes=cleaned,
            )
        )
    if n_clipped:
        print(f"  load_records: clipped {n_clipped} out-of-bounds bboxes")
    if n_skipped_no_dims:
        print(f"  load_records: {n_skipped_no_dims} records have manifest "
              f"dims=0 (CMU upstream bug) — OOB clip deferred to image-load fallback")
    return records


def _gaussian_radius(w: float, h: float, min_overlap: float = 0.7) -> int:
    """CenterNet heuristic for Gaussian sigma per bbox size."""
    a1, b1 = 1.0, h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - math.sqrt(max(b1 * b1 - 4 * a1 * c1, 0.0))) / (2 * a1)

    a2, b2 = 4.0, 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    r2 = (b2 - math.sqrt(max(b2 * b2 - 4 * a2 * c2, 0.0))) / (2 * a2)

    a3, b3 = 4 * min_overlap, -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(max(b3 * b3 - 4 * a3 * c3, 0.0))) / (2 * a3)

    return max(1, int(min(r1, r2, r3)))


def _draw_gaussian(heatmap: np.ndarray, center: tuple[int, int], radius: int) -> None:
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    grid = np.arange(diameter) - radius
    gx, gy = np.meshgrid(grid, grid)
    g = np.exp(-(gx * gx + gy * gy) / (2 * sigma * sigma))

    cx, cy = center
    h, w = heatmap.shape
    left, right = min(cx, radius), min(w - cx, radius + 1)
    top, bottom = min(cy, radius), min(h - cy, radius + 1)
    if left + right <= 0 or top + bottom <= 0:
        return

    masked_heatmap = heatmap[cy - top : cy + bottom, cx - left : cx + right]
    masked_g = g[radius - top : radius + bottom, radius - left : radius + right]
    np.maximum(masked_heatmap, masked_g, out=masked_heatmap)


class HandBboxDataset(Dataset):
    """Returns (image, target_dict).

    image: float32 tensor (3, INPUT_SIZE, INPUT_SIZE) in [0, 1].
    target_dict: dict with
        - heatmap:     (1, OUT_SIZE, OUT_SIZE)  float32 Gaussian map
        - size:        (2, OUT_SIZE, OUT_SIZE)  float32, nonzero at centers
        - center_mask: (1, OUT_SIZE, OUT_SIZE)  float32 binary

    When `cache_in_memory=True`, all images are decoded + pre-resized to
    INPUT_SIZE × INPUT_SIZE uint8 tensors at __init__. For ~14k CMU
    samples this is ~4.3 GB host RAM — easily fits on a Modal H100 box
    (which has hundreds of GB) and turns disk I/O from the training
    bottleneck into a one-time cost.
    """

    INPUT_SIZE = 320
    STRIDE = 4
    OUT_SIZE = INPUT_SIZE // STRIDE  # 80

    def __init__(
        self,
        records: list[FrameRecord],
        augment: Callable[[torch.Tensor, list[list[float]]], tuple[torch.Tensor, list[list[float]]]] | None = None,
        cache_in_memory: bool = False,
        gpu_aug_mode: bool = False,
        packed_path: Path | None = None,
    ) -> None:
        self.records = records
        self.augment = augment
        # FAST PATH: a pre-decoded uint8 memmap built once by
        # scripts/pack_hand_dataset.py. Workers each mmap the SAME file
        # (OS page cache shared, copy-on-write safe) — no per-run JPEG
        # decode and, crucially, NO tensor.share_memory_() (which SIGBUSes
        # in Modal's small /dev/shm). Images must be packed in records order.
        self._mmap = None
        if packed_path is not None:
            meta = json.loads(Path(packed_path).read_text())
            dat = Path(packed_path).parent / meta["dat"]
            self._mmap = np.memmap(dat, dtype=meta["dtype"], mode="r",
                                   shape=tuple(meta["shape"]))
            self._cached_bboxes = [[list(b) for b in bb] for bb in meta["bboxes"]]
            print(f"  packed dataset: {meta['shape'][0]} images mmap'd from "
                  f"{dat} ({np.prod(meta['shape']) / 1e9:.1f} GB on disk, 0 RAM)")
        # When True, __getitem__ returns the bare resized image + padded
        # bboxes + mask, with NO CPU-side augmentation or target render.
        # The train loop does both on GPU via training/detectors/gpu_augment.py.
        # Eliminates the 6 ms/sample CPU bottleneck.
        self.gpu_aug_mode = gpu_aug_mode
        # When cache_in_memory=True, _cache is a single (N, 3, H, W) uint8
        # tensor in shared memory so DataLoader workers can index into it
        # zero-copy after fork. Earlier list[Tensor] form blocked that and
        # forced eff_workers=0, which destroyed throughput (see ADR notes
        # in train.py — the "32 ms/sample" plateau diagnosis).
        # NOTE: prefer packed_path (memmap) over cache_in_memory — same
        # zero-decode speed without the share_memory_() SIGBUS risk.
        self._cache: torch.Tensor | None = None
        if self._mmap is None:
            self._cached_bboxes = None
            if cache_in_memory:
                self._build_cache()

    def __len__(self) -> int:
        return len(self.records)

    def _build_cache(self, max_workers: int = 16) -> None:
        """Pre-decode + resize every image to a single shared-memory
        uint8 tensor of shape (N, 3, INPUT_SIZE, INPUT_SIZE).

        ~17.9 GB for 61k images — fits easily on Modal A100/H100 boxes
        (>100 GB host RAM). Calling `share_memory_()` makes the buffer
        zero-copy-readable by DataLoader workers across fork/spawn.

        Threaded — tvio.read_image releases the GIL during disk + JPEG
        decode, so a ThreadPoolExecutor saturates network I/O on a Modal
        volume. Bboxes are pre-scaled to INPUT_SIZE space; augs + target
        render still run per __getitem__.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n = len(self.records)
        input_size = self.INPUT_SIZE
        # Pre-allocate the shared mega-tensor. share_memory_() must be
        # called BEFORE workers fork — DataLoader does the fork lazily
        # on the first iteration, so doing it here in main is safe.
        self._cache = torch.empty(
            (n, 3, input_size, input_size), dtype=torch.uint8
        )
        if os.environ.get("CACHE_NO_SHM") == "1":
            print("CACHE_NO_SHM=1 → skipping share_memory_() "
                  "(cache in main-process heap; requires num_workers=0)")
        else:
            self._cache.share_memory_()
        self._cached_bboxes = [None] * n
        records = self.records

        def _load(i: int) -> tuple[int, torch.Tensor, list[list[float]]]:
            rec = records[i]
            img = tvio.read_image(str(rec.image_path),
                                  mode=tvio.ImageReadMode.RGB)
            _, ah, aw = img.shape
            src_w = rec.width if rec.width else aw
            src_h = rec.height if rec.height else ah
            resized = torch.nn.functional.interpolate(
                img.unsqueeze(0).float(),
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).clamp(0, 255).to(torch.uint8)
            sx = input_size / src_w
            sy = input_size / src_h
            scaled = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
                      for b in rec.bboxes]
            return i, resized, scaled

        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_load, i) for i in range(n)]
            for fut in as_completed(futures):
                i, img, boxes = fut.result()
                self._cache[i].copy_(img)
                self._cached_bboxes[i] = boxes
                done += 1
                if done % 2000 == 0:
                    print(f"  cache: {done}/{n}  ({done / (time.time() - t0):.0f} img/s)")
        print(f"  cache built: {n} images in {time.time() - t0:.1f}s "
              f"({self._cache.numel() / 1e9:.1f} GB shared)")

    def __getitem__(self, idx: int):
        # In gpu_aug_mode we keep the image as uint8 and let the train loop do
        # the /255 float-cast ON the GPU — a 4× smaller host→device transfer
        # (the landmark-trainer pipeline win). Otherwise cast to float here.
        to_float = not self.gpu_aug_mode
        if self._mmap is not None:
            # Copy the (3,H,W) uint8 slice out of the read-only mmap. .copy()
            # detaches from the mmap page so downstream in-place ops are safe.
            arr = np.array(self._mmap[idx])  # writable copy off the mmap page
            img = torch.from_numpy(arr)
            if to_float:
                img = img.float() / 255.0
            bboxes = [b[:] for b in self._cached_bboxes[idx]]
        elif self._cache is not None:
            img = self._cache[idx]
            img = img.float() / 255.0 if to_float else img.clone()
            bboxes = [b[:] for b in self._cached_bboxes[idx]]
        else:
            rec = self.records[idx]
            img = tvio.read_image(str(rec.image_path), mode=tvio.ImageReadMode.RGB).float() / 255.0
            actual_h, actual_w = img.shape[1], img.shape[2]
            src_w = rec.width if rec.width else actual_w
            src_h = rec.height if rec.height else actual_h
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=(self.INPUT_SIZE, self.INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            sx = self.INPUT_SIZE / src_w
            sy = self.INPUT_SIZE / src_h
            bboxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in rec.bboxes]
            if not to_float:  # JPEG fallback path under gpu_aug_mode → back to uint8
                img = (img * 255.0).round().clamp(0, 255).to(torch.uint8)

        if self.gpu_aug_mode:
            # No aug, no target render — train loop does both on GPU.
            from training.detectors.gpu_augment import pad_bboxes
            bbox_padded, bbox_mask = pad_bboxes(bboxes)
            return img, bbox_padded, bbox_mask

        if self.augment is not None:
            img, bboxes = self.augment(img, bboxes)

        target = self._render_targets(bboxes)
        return img, target

    def _render_targets(self, bboxes: list[list[float]]) -> dict[str, torch.Tensor]:
        H = W = self.OUT_SIZE
        heatmap = np.zeros((1, H, W), dtype=np.float32)
        size = np.zeros((2, H, W), dtype=np.float32)
        center_mask = np.zeros((1, H, W), dtype=np.float32)

        for x0, y0, x1, y1 in bboxes:
            bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
            cx = (x0 + x1) / 2 / self.STRIDE
            cy = (y0 + y1) / 2 / self.STRIDE
            ix, iy = int(cx), int(cy)
            if not (0 <= ix < W and 0 <= iy < H):
                continue
            radius = _gaussian_radius(bw / self.STRIDE, bh / self.STRIDE)
            _draw_gaussian(heatmap[0], (ix, iy), radius)
            size[0, iy, ix] = bw
            size[1, iy, ix] = bh
            center_mask[0, iy, ix] = 1.0

        return {
            "heatmap": torch.from_numpy(heatmap),
            "size": torch.from_numpy(size),
            "center_mask": torch.from_numpy(center_mask),
        }
