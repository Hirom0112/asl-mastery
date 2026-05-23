"""Small TCN classifier head over (T, F) sign trajectories.

Replaces the Mahalanobis-template matcher (which the Session 17 magnet-
pathology diagnostic showed is structurally broken on this dataset).
Same input — the 32-step × 100-feature trajectory tensor produced by
trajectory_from_frames() — but outputs a softmax over the 70 signs
directly, trained with cross-entropy.

Why TCN over LSTM/Transformer at this size:
- ~1M params; trains in minutes on any modern GPU
- Causal-friendly + parallel-friendly (no recurrence)
- Dilated convs cover the 32-frame window in 4 layers
- No magnet pathology — cross-entropy directly punishes class imbalance
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int,
                 dropout: float = 0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.act = nn.GELU()
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # symmetric padding — same-length output
        y = self.conv1(x)[:, :, :x.size(2)] if self.pad else self.conv1(x)
        y = self.act(self.norm1(y))
        y = self.dropout(y)
        y = self.conv2(y)[:, :, :x.size(2)] if self.pad else self.conv2(y)
        y = self.act(self.norm2(y))
        y = self.dropout(y)
        return x + y  # residual


class BlockLayerNorm(nn.Module):
    """LayerNorm applied independently per feature block.

    The 356D feature vector concatenates blocks at wildly different scales
    (keypoints mean|·|≈2.1, the 128D L2-normed handshape embedding ≈0.06).
    A single LayerNorm over all 356 dims lets the keypoint variance dominate
    the shared statistics, scaling the embedding to ~noise. Normalizing each
    block to unit scale separately keeps every block contributing.
    """

    def __init__(self, block_sizes: list[int]):
        super().__init__()
        self.block_sizes = list(block_sizes)
        self.norms = nn.ModuleList(nn.LayerNorm(s) for s in self.block_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = torch.split(x, self.block_sizes, dim=-1)
        return torch.cat([n(p) for n, p in zip(self.norms, parts)], dim=-1)


class SignClassifier(nn.Module):
    """(B, T, F) → (B, num_classes) logits."""

    def __init__(
        self,
        num_features: int = 100,
        num_classes: int = 70,
        hidden: int = 192,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
        norm_blocks: list[int] | None = None,
    ):
        super().__init__()
        if norm_blocks is not None:
            assert sum(norm_blocks) == num_features, (
                f"norm_blocks {norm_blocks} sum {sum(norm_blocks)} != "
                f"num_features {num_features}")
            self.input_norm = BlockLayerNorm(norm_blocks)
        else:
            self.input_norm = nn.LayerNorm(num_features)
        self.input_proj = nn.Conv1d(num_features, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([
            TemporalBlock(hidden, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.input_norm(x)
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=2)  # global avg pool over time → (B, hidden)
        return self.head(x)


class TransformerSignClassifier(nn.Module):
    """(B, T, F) → (B, num_classes) via small transformer encoder + CLS token.

    Phase 4.7 — A/B comparison vs the TCN. Same input contract (32-step ×
    100-feature flattened trajectory) so we can swap freely. Two sizings
    are validated in parallel:
      - fair-comp (d_model=192, layers=3, heads=4) — ~1M params, matches TCN
      - upsized  (d_model=256, layers=4, heads=8) — ~2M params, capacity test
    """

    def __init__(
        self,
        num_features: int = 100,
        num_classes: int = 75,
        d_model: int = 192,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        max_len: int = 64,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(num_features)
        self.input_proj = nn.Linear(num_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.input_norm(x)
        x = self.input_proj(x)  # (B, T, d_model)
        b, t, _ = x.shape
        cls = self.cls_token.expand(b, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)
        x = x + self.pos_embed[:, : t + 1, :]
        x = self.encoder(x)
        return self.head(x[:, 0])  # CLS token → logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
