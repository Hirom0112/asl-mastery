"""Dataset for hand keypoint regression training.

Consumes the hand_keypoints v1 manifest produced by
training/detectors/normalize_external.py:

  {
    "version": 1,
    "task": "hand_keypoints",
    "items": [
      {
        "image_path": "data/external/freihand/...jpg",
        "width": 224, "height": 224,
        "hands": [{
          "bbox": [x0, y0, x1, y1],     # in image pixel coords
          "keypoints": [[x, y, v], ...] # length 21, in image pixel coords
        }]
      },
      ...
    ]
  }

For each (image, hand) pair the dataset crops the bbox (with padding),
resizes to 224×224, and returns coords normalized to [0, 1] of the crop
plus a visibility vector.

Image-width/height fields in the manifest may be 0 (some loaders skip
PIL). In that case the dataset reads it from disk at __init__ via PIL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torchvision.io as tvio
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


def _ensure_size(item: dict, path: Path) -> tuple[int, int]:
    if item.get("width") and item.get("height"):
        return int(item["width"]), int(item["height"])
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        return 0, 0


def load_manifest(manifest_path: Path) -> list[dict]:
    raw = json.loads(manifest_path.read_text())
    assert raw["version"] == 1, raw["version"]
    assert raw["task"] in ("hand_keypoints", "pose_keypoints", "face_keypoints"), raw["task"]
    return raw["items"]


class HandLandmarkDataset(Dataset):
    INPUT_SIZE = 224

    def __init__(
        self,
        items: list[dict],
        repo_root: Path,
        crop_pad_frac: float = 0.20,
        augment: Callable | None = None,
        num_keypoints: int = 21,
        instance_key: str = "hands",
        input_size: int | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.crop_pad_frac = crop_pad_frac
        self.augment = augment
        self.num_keypoints = num_keypoints
        self.instance_key = instance_key
        if input_size is not None:
            self.INPUT_SIZE = input_size
        # Expand to one entry per hand. We deliberately skip per-image
        # existence checks AND per-image PIL size-reads here — both are
        # prohibitively slow on a Modal volume with 60K+ items. The
        # dataset reader uses the actual decoded image shape in
        # __getitem__ and torchvision.io will raise a clear error on a
        # genuinely-missing image, surfaced via DataLoader.
        self.entries: list[dict] = []
        for it in items:
            raw_path = it["image_path"]
            img_path = Path(raw_path) if Path(raw_path).is_absolute() \
                else (self.repo_root / raw_path).resolve()
            for hand in it.get(self.instance_key, []):
                bbox = hand.get("bbox")
                kps = hand.get("keypoints")
                if not bbox or not kps or len(kps) != self.num_keypoints:
                    continue
                entry = {
                    "image_path": img_path,
                    "bbox": bbox,
                    "keypoints": kps,
                }
                # v3 — 3D landmarks. Depth is camera-axis (root-relative,
                # scale-normalized) and therefore invariant to the in-plane
                # augmentations (hflip mirrors x only; rotate/scale/translate
                # are about the optical axis), so z rides through untouched.
                kz = hand.get("keypoints_z")
                if kz is not None and len(kz) == self.num_keypoints:
                    entry["keypoints_z"] = kz
                self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        e = self.entries[idx]
        img = tvio.read_image(str(e["image_path"]), mode=tvio.ImageReadMode.RGB).float() / 255.0
        _, H, W = img.shape

        x0, y0, x1, y1 = e["bbox"]
        bw, bh = x1 - x0, y1 - y0
        pad = self.crop_pad_frac * max(bw, bh)
        cx0 = max(0, int(x0 - pad))
        cy0 = max(0, int(y0 - pad))
        cx1 = min(W, int(x1 + pad))
        cy1 = min(H, int(y1 + pad))
        if cx1 <= cx0 or cy1 <= cy0:
            cx0, cy0, cx1, cy1 = 0, 0, W, H

        crop = img[:, cy0:cy1, cx0:cx1]
        resized = torch.nn.functional.interpolate(
            crop.unsqueeze(0),
            size=(self.INPUT_SIZE, self.INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Transform keypoints into crop-normalized [0, 1] space
        sx = self.INPUT_SIZE / max(cx1 - cx0, 1)
        sy = self.INPUT_SIZE / max(cy1 - cy0, 1)
        kp_t = []
        vis_t = []
        for x, y, v in e["keypoints"]:
            nx = (x - cx0) * sx / self.INPUT_SIZE
            ny = (y - cy0) * sy / self.INPUT_SIZE
            kp_t.append([nx, ny])
            vis_t.append(1.0 if v > 0 else 0.0)
        coords = torch.tensor(kp_t, dtype=torch.float32)
        vis = torch.tensor(vis_t, dtype=torch.float32)

        if self.augment is not None:
            resized, coords, vis = self.augment(resized, coords, vis)

        target = {"coords": coords, "visibility": vis}
        if "keypoints_z" in e:
            # z is aug-invariant (see __init__), so it's attached AFTER augment
            # untouched. has_depth lets the loss mask 2D-only sources.
            target["depth"] = torch.tensor(e["keypoints_z"], dtype=torch.float32)
            target["has_depth"] = torch.tensor(1.0, dtype=torch.float32)
        return resized, target


class PackedLandmarkDataset(Dataset):
    """Reads pre-cropped 224² hand crops from a uint8 memmap (built by
    modal_app.pack_landmark_dataset) — no per-epoch JPEG decode. Same output
    contract as HandLandmarkDataset so the augmentation pipeline is drop-in:
    returns (img 3×S×S float[0,1], {coords (21,2)[0,1], visibility (21,)})."""

    def __init__(self, meta_path: str | Path, augment: Callable | None = None,
                 indices: list[int] | None = None) -> None:
        meta_path = str(meta_path)
        assert meta_path.endswith(".meta.json"), meta_path
        base = meta_path[: -len(".meta.json")]
        meta = json.loads(Path(meta_path).read_text())
        self.INPUT_SIZE = meta["input_size"]
        self.mm = np.memmap(base + ".dat", dtype=np.uint8, mode="r",
                            shape=tuple(meta["shape"]))
        self.kps = np.load(base + ".kps.npy")
        self.vis = np.load(base + ".vis.npy")
        self.augment = augment
        src = meta["source"]
        # keep only successfully-packed crops (empty source = decode failed)
        idxs = indices if indices is not None else range(meta["n"])
        self.indices = [i for i in idxs if src[i]]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        # Return UINT8 (3,H,W) — 4× smaller host→device transfer than float;
        # the train/val loop converts to float[0,1] on the GPU. .copy() makes
        # it writable (silences the non-writable-memmap warning). This dataset
        # is the GPU-aug path, so no CPU augment here.
        img = torch.from_numpy(self.mm[i].copy())          # uint8
        coords = torch.from_numpy(self.kps[i].copy())
        vis = torch.from_numpy(self.vis[i].copy())
        return img, {"coords": coords, "visibility": vis}
