"""PyTorch dataset wrapping the cleaning-pipeline manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from training.classifier.augment import augment
from training.data.vocabulary import get_all


class KeypointDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        augment_training: bool = True,
    ) -> None:
        with manifest_path.open() as f:
            manifest = json.load(f)
        self.manifest = manifest
        self.clips = [c for c in manifest["clips"] if c["split"] == split]
        self.split = split
        self.augment_training = augment_training and split == "train"

        # sign_id → class index (alphabetical for determinism).
        kept_signs = sorted({c["sign_id"] for c in self.clips})
        self.classes = kept_signs
        self.sign_to_idx = {s: i for i, s in enumerate(kept_signs)}

        # Cache flippable flag per sign so augment() can skip it cheaply.
        self.flippable: dict[str, bool] = {item.sign_id: item.flippable for item in get_all()}

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        clip = self.clips[idx]
        x = np.load(clip["keypoint_path"]).astype(np.float32)  # (T, K)
        if self.augment_training:
            x = augment(x, flippable=self.flippable.get(clip["sign_id"], False))
        y = self.sign_to_idx[clip["sign_id"]]
        return torch.from_numpy(x), int(y)


def make_weighted_sampler(dataset: KeypointDataset) -> torch.utils.data.WeightedRandomSampler:
    """Per-class inverse-frequency weighted sampling for class balance."""
    from collections import Counter

    counts = Counter(c["sign_id"] for c in dataset.clips)
    weights = [1.0 / counts[c["sign_id"]] for c in dataset.clips]
    return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
