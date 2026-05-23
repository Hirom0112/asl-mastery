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
        #
        # CRITICAL bf16 fix: bf16 cannot represent (1 - 1e-4) as anything
        # other than 1.0 (bf16 has 7-bit mantissa → ~0.0078 resolution
        # near 1.0). So the clamp upper bound collapsed to 1.0, and at
        # random init logits range ~±26 → sigmoid → 1.0 → log(1-1.0) →
        # -inf → NaN on the very first batch. Force this whole stanza to
        # run in fp32 even under autocast.
        with torch.autocast(device_type=pred_logits.device.type, enabled=False):
            logits_f32 = pred_logits.float()
            target_f32 = target.float()
            pred = torch.sigmoid(logits_f32).clamp(1e-4, 1 - 1e-4)

            pos_mask = target_f32.eq(1).float()
            neg_mask = (1 - pos_mask)

            pos_loss = -((1 - pred) ** self.alpha) * torch.log(pred) * pos_mask
            neg_loss = (
                -((1 - target_f32) ** self.beta)
                * (pred ** self.alpha)
                * torch.log(1 - pred)
                * neg_mask
            )

            n_pos = pos_mask.sum().clamp(min=1.0)
            return (pos_loss.sum() + neg_loss.sum()) / n_pos


class SizeGIoULoss(nn.Module):
    """GIoU loss on predicted (width, height) at ground-truth center pixels.

    Both predicted and target boxes share the same center pixel (cx, cy),
    so the GIoU reduces to a function of (pred_w, pred_h, target_w, target_h)
    alone. This is a scale-invariant replacement for L1 over raw pixel
    sizes — universal +1-3 AP across dense detectors (Zheng et al. AAAI-20,
    arxiv 1911.08287).

    The whole loss is forced to fp32 because GIoU's IoU/union/enclose
    ratios trip the same bf16 underflow class that broke the focal heatmap.
    """

    def forward(
        self,
        pred_size: torch.Tensor,    # (B, 2, H, W)
        target_size: torch.Tensor,  # (B, 2, H, W) zero outside centers
        center_mask: torch.Tensor,  # (B, 1, H, W) binary
    ) -> torch.Tensor:
        with torch.autocast(device_type=pred_size.device.type, enabled=False):
            pred = pred_size.float().clamp(min=1e-3)  # positive sizes only
            target = target_size.float()
            pw, ph = pred[:, 0:1], pred[:, 1:2]
            tw, th = target[:, 0:1], target[:, 1:2]
            # Centered-box intersection / union / enclose
            iw = torch.min(pw, tw)
            ih = torch.min(ph, th)
            inter = iw * ih
            union = (pw * ph) + (tw * th) - inter
            iou = inter / union.clamp(min=1e-9)
            # Enclosing axis-aligned bbox around the two centered boxes
            cw = torch.max(pw, tw)
            ch = torch.max(ph, th)
            enclose = cw * ch
            giou = iou - (enclose - union) / enclose.clamp(min=1e-9)
            # GIoU is in (-1, 1]; 1-GIoU in [0, 2), well-behaved.
            loss = (1.0 - giou) * center_mask.float()
            n = center_mask.sum().clamp(min=1.0)
            return loss.sum() / n


class HandDetectorLoss(nn.Module):
    """Combined heatmap + size loss with a fixed weight.

    `size_weight=0.5` (was 0.1) so the size-regression branch gets actual
    gradient signal — at 0.1, heatmap loss dominated and size head
    converged to rough averages only. Diagnosed by the hyperparam audit
    agent for this plateau.
    """

    def __init__(self, size_weight: float = 0.5) -> None:
        super().__init__()
        self.heatmap = CenterNetFocalLoss()
        self.size = SizeGIoULoss()
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
