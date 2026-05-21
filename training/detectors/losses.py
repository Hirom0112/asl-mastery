"""Losses for the CenterNet-style hand detector.

Two terms per docs/VOCABULARY_TRAINER_ROADMAP.md Phase 1 Slice 1.2:
- Focal loss on the center-point heatmap (CornerNet / CenterNet variant).
- L1 loss on size regression, applied only at ground-truth center pixels.
"""

from __future__ import annotations

import torch
from torch import nn


class CenterNetFocalLoss(nn.Module):
    """Modified focal loss from CornerNet / CenterNet.

    Target heatmap is a Gaussian-rendered map per ground-truth center,
    not a binary mask. Negatives are weighted by (1 - target)^beta so
    near-center pixels do not get penalized as full negatives.

    Args:
        alpha: focal exponent on hard positives (default 2.0).
        beta:  exponent damping negatives near positives (default 4.0).
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred_logits: (B, 1, H, W)  unbounded
        # target:      (B, 1, H, W)  values in [0, 1], 1 at centers
        pred = torch.sigmoid(pred_logits).clamp(1e-4, 1 - 1e-4)

        pos_mask = target.eq(1).float()
        neg_mask = (1 - pos_mask)

        pos_loss = -((1 - pred) ** self.alpha) * torch.log(pred) * pos_mask
        neg_loss = (
            -((1 - target) ** self.beta)
            * (pred ** self.alpha)
            * torch.log(1 - pred)
            * neg_mask
        )

        n_pos = pos_mask.sum().clamp(min=1.0)
        return (pos_loss.sum() + neg_loss.sum()) / n_pos


class SizeL1Loss(nn.Module):
    """L1 loss on predicted (width, height) at ground-truth center pixels.

    Center pixels are a sparse subset of the feature map. The center
    mask is binary (1 at each ground-truth center, 0 elsewhere).
    """

    def forward(
        self,
        pred_size: torch.Tensor,    # (B, 2, H, W)
        target_size: torch.Tensor,  # (B, 2, H, W) zero outside centers
        center_mask: torch.Tensor,  # (B, 1, H, W) binary
    ) -> torch.Tensor:
        n = center_mask.sum().clamp(min=1.0)
        diff = torch.abs(pred_size - target_size) * center_mask
        return diff.sum() / n


class HandDetectorLoss(nn.Module):
    """Combined heatmap + size loss with a fixed weight."""

    def __init__(self, size_weight: float = 0.1) -> None:
        super().__init__()
        self.heatmap = CenterNetFocalLoss()
        self.size = SizeL1Loss()
        self.size_weight = size_weight

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        loss_hm = self.heatmap(outputs["heatmap"], targets["heatmap"])
        loss_sz = self.size(outputs["size"], targets["size"], targets["center_mask"])
        total = loss_hm + self.size_weight * loss_sz
        return {"total": total, "heatmap": loss_hm.detach(), "size": loss_sz.detach()}
