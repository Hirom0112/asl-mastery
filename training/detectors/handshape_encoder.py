"""HandshapeEncoder — small from-scratch CNN producing a 128D handshape
embedding from a 224×224 RGB hand crop.

Phase 4.6. Trained via contrastive learning (NT-Xent / SimCLR) on the
~182K hand crops dumped by scripts/dump_hand_crops.py. ADR-0012 strict
from-scratch — no pretrained weights, no foreign load_state_dict.

Architecture: ResNet-ish small backbone with progressively widening
channels, GAP, then a 2-layer MLP projection head with L2-normalized
output. Sized to ~5M params — small enough to train fast on H100 with
batch 1024+, large enough to capture handshape gestalt invariances.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn(in_c: int, out_c: int, k: int = 3, stride: int = 1, padding: int | None = None) -> nn.Sequential:
    p = (k // 2) if padding is None else padding
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, stride=stride, padding=p, bias=False),
        nn.BatchNorm2d(out_c),
    )


class ResBlock(nn.Module):
    """Pre-activation ResNet block. Channel-preserving or downsampling."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = _conv_bn(in_c, out_c, k=3, stride=stride)
        self.conv2 = _conv_bn(out_c, out_c, k=3, stride=1)
        self.act = nn.GELU()
        if in_c != out_c or stride != 1:
            self.shortcut = _conv_bn(in_c, out_c, k=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(y + self.shortcut(x))


class HandshapeEncoder(nn.Module):
    """224×224×3 → 128D L2-normalized embedding.

    Backbone stages with progressively wider channels (64→128→256→512),
    each stage starting with a stride-2 downsample. Output spatial after
    backbone: 7×7. GAP → 512D → MLP(512→256→128) → L2-norm.
    """

    INPUT_SIZE = 224
    EMBED_DIM = 128

    def __init__(self, embed_dim: int = 128, projection_hidden: int = 256):
        super().__init__()
        # Stem: 224 → 112 → 56
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Stages
        self.stage1 = nn.Sequential(
            ResBlock(32, 64, stride=1),
            ResBlock(64, 64, stride=1),
        )
        self.stage2 = nn.Sequential(
            ResBlock(64, 128, stride=2),  # 56 → 28
            ResBlock(128, 128, stride=1),
        )
        self.stage3 = nn.Sequential(
            ResBlock(128, 256, stride=2),  # 28 → 14
            ResBlock(256, 256, stride=1),
        )
        self.stage4 = nn.Sequential(
            ResBlock(256, 512, stride=2),  # 14 → 7
            ResBlock(512, 512, stride=1),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Projection head (SimCLR-style)
        self.proj = nn.Sequential(
            nn.Linear(512, projection_hidden),
            nn.GELU(),
            nn.Linear(projection_hidden, embed_dim),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-projection features (used at inference time for embeddings)."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).flatten(1)  # (B, 512)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalized projection embedding for contrastive loss.

        Clone after F.normalize to play nice with `torch.compile`'s CUDAGraphs
        path: the normalize+downstream NT-Xent sim matrix produces a tensor
        whose buffer would otherwise be overwritten by the next CUDAGraph step.
        """
        feats = self.features(x)
        z = self.proj(feats)
        return F.normalize(z, dim=-1).clone()


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss, SimCLR-style.

    z1, z2: (B, D) L2-normalized projection embeddings of two augmented views
    of the same B items. Positives are (z1[i], z2[i]); all other pairs are
    negatives.
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = z @ z.t() / temperature  # (2B, 2B), cosine sim / T (z is normalized)
    # Mask self-similarity (the diagonal)
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)
    # Targets: for row i in [0, B), positive is at i+B; for row i in [B, 2B),
    # positive is at i-B.
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
