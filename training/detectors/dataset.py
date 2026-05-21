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
    assert raw["task"] == "hand_bbox", raw["task"]

    repo_root = manifest_path.resolve().parent
    while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
        repo_root = repo_root.parent

    records: list[FrameRecord] = []
    for item in raw["items"]:
        records.append(
            FrameRecord(
                image_path=repo_root / item["image_path"],
                width=int(item["width"]),
                height=int(item["height"]),
                bboxes=[[float(x) for x in b] for b in item["bboxes"]],
            )
        )
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
    """

    INPUT_SIZE = 320
    STRIDE = 4
    OUT_SIZE = INPUT_SIZE // STRIDE  # 80

    def __init__(
        self,
        records: list[FrameRecord],
        augment: Callable[[torch.Tensor, list[list[float]]], tuple[torch.Tensor, list[list[float]]]] | None = None,
    ) -> None:
        self.records = records
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rec = self.records[idx]
        img = tvio.read_image(str(rec.image_path), mode=tvio.ImageReadMode.RGB).float() / 255.0
        # img: (3, H, W) where H, W are rec.height, rec.width.

        # Resize to square INPUT_SIZE (letterboxing left out for slice-1
        # simplicity; webcam aspect ratio is configured to 1:1 at capture).
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(self.INPUT_SIZE, self.INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Scale bboxes from source resolution to INPUT_SIZE space.
        sx = self.INPUT_SIZE / rec.width
        sy = self.INPUT_SIZE / rec.height
        bboxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in rec.bboxes]

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
