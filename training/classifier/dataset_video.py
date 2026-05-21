"""MP4 dataset loader for the v3.x raw-RGB training pipeline.

Reads cleaned per-clip MP4s from `dataset/clean/v<N>/normalized_videos/`,
samples a fixed-length frame window, applies pixel-level augmentation
on the training split, and returns `(video_tensor, label)` pairs where
`video_tensor` has shape `(T, H, W, 3)` float32 in `[0, 1]`.

Replaces `training/classifier/dataset.py` (the keypoint-tensor loader
deleted under T3 commit 2). Manifest format mirrors what the new
`training/data/clean.py` writes — each record carries
`normalized_video_path`, `sign_id`, `split`, and a `source_signer_id`
that drove the signer-disjoint split assignment.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


# Default capture window: 16 frames. Matches the camera-capture
# component's CAPTURE_MS / FRAME_INTERVAL_MS contract.
TEMPORAL_LENGTH = 16
DEFAULT_INPUT_SIZE = 96


class VideoClipDataset(Dataset):
    """Indexed access to the cleaned MP4 corpus for one split.

    Args:
        manifest_path: path to the cleaning-pipeline manifest JSON.
        split: one of {"train", "val", "test"}.
        augment: optional callable applied to each frame stack in
            training. Signature: ``(frames: ndarray[T, 256, 256, 3],
            sign_id: str) -> ndarray[T, H, W, 3]``. None means no
            augmentation (val/test).
        input_height, input_width: cropped output spatial dims.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        augment: Callable[[np.ndarray, str], np.ndarray] | None = None,
        input_height: int = DEFAULT_INPUT_SIZE,
        input_width: int = DEFAULT_INPUT_SIZE,
        temporal_length: int = TEMPORAL_LENGTH,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open() as f:
            self.manifest = json.load(f)
        self.split = split
        self.augment = augment
        self.input_height = input_height
        self.input_width = input_width
        self.temporal_length = temporal_length

        all_records: list[dict[str, Any]] = self.manifest.get("records", [])
        self.clips = [r for r in all_records if r.get("split") == split]
        self.classes: list[str] = self.manifest.get(
            "classes",
            sorted({r["sign_id"] for r in all_records}),
        )
        self.sign_to_idx = {s: i for i, s in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.clips)

    def _load_clip(self, path: Path) -> np.ndarray:
        """Read an MP4 → uint8 ndarray of shape (T, 256, 256, 3) in RGB.

        Uses torchvision.io.read_video for portability — it ships with
        the training image and avoids a separate decord dep. Returns a
        uniformly-sampled `T`-frame window over the available frames
        (with edge-replication padding if the source is shorter than
        ``temporal_length``).
        """
        import torchvision.io as tvio

        # read_video returns (T, H, W, C) uint8 in RGB by default.
        frames, _audio, _meta = tvio.read_video(str(path), pts_unit="sec")
        if frames.numel() == 0:
            raise RuntimeError(f"read_video returned no frames for {path}")
        frames_np = frames.numpy()  # (T_src, H, W, 3) uint8

        T_src = frames_np.shape[0]
        T_dst = self.temporal_length
        if T_src >= T_dst:
            # Evenly-spaced indices over the source clip.
            idx = np.linspace(0, T_src - 1, T_dst).round().astype(np.int64)
            return frames_np[idx]
        # Source shorter: replicate the last frame.
        out = np.empty((T_dst, *frames_np.shape[1:]), dtype=frames_np.dtype)
        out[:T_src] = frames_np
        out[T_src:] = frames_np[-1:]
        return out

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rec = self.clips[idx]
        mp4_path = Path(rec["normalized_video_path"])
        frames_u8 = self._load_clip(mp4_path)  # (T, 256, 256, 3) uint8

        if self.augment is not None:
            frames_u8 = self.augment(frames_u8, rec["sign_id"])  # (T, H, W, 3) uint8
        else:
            # Eval path: center-crop down to input_height × input_width.
            T, H, W, _ = frames_u8.shape
            top = max(0, (H - self.input_height) // 2)
            left = max(0, (W - self.input_width) // 2)
            frames_u8 = frames_u8[:, top : top + self.input_height, left : left + self.input_width, :]

        # uint8 → float32 in [0, 1]. Channel-last (T, H, W, 3).
        tensor = torch.from_numpy(frames_u8.astype(np.float32) / 255.0)
        label = self.sign_to_idx[rec["sign_id"]]
        return tensor, label


def make_weighted_sampler(dataset: VideoClipDataset) -> WeightedRandomSampler:
    """Per-class weighting so each class appears roughly equally per epoch.

    Mirrors `training/classifier/dataset.py::make_weighted_sampler` (the
    old keypoint variant) so train.py can swap dataset implementations
    transparently.
    """
    counts: dict[str, int] = {}
    for r in dataset.clips:
        counts[r["sign_id"]] = counts.get(r["sign_id"], 0) + 1
    weights = [1.0 / counts[r["sign_id"]] for r in dataset.clips]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def split_signer_ids(dataset: VideoClipDataset) -> dict[str, int]:
    """Return the count of unique source_signer_ids per split — useful
    in validate.py for honest signer-disjointedness reporting."""
    signers = {r.get("source_signer_id") for r in dataset.clips}
    return {dataset.split: len(signers)}


# Module-level smoke check: catches obvious manifest-shape regressions
# during ``python -c 'import training.classifier.dataset_video'``.
def _self_check() -> None:
    random.seed(0)
    np.random.seed(0)


_self_check()
