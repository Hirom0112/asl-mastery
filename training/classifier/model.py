"""Classifier architectures per docs/MODEL.md §1.

Two from-scratch architectures over MediaPipe keypoint sequences:
- BiLSTM (baseline, ~200K params)
- Transformer encoder (alternative, ~500K params; reserved if BiLSTM underperforms)

Neither loads pretrained weights. All linear/projection layers are
initialized in `init.py` via `torch.nn.init.kaiming_normal_`. LSTM
internals use PyTorch defaults. The "no-pretrained-pipeline" audit
surface is `training/classifier/init.py` + this file: no
`load_state_dict`, no external weight URLs.
"""

from __future__ import annotations

import math

import torch
from torch import nn


# Number of float coords occupied by both hands in the keypoint layout
# (must stay in lockstep with training/keypoints.py: 2 hands × 21 × 3 = 126).
_HAND_BLOCK_FLOATS = 126


def _frame_valid_mask(x: torch.Tensor) -> torch.Tensor:
    """Phase 9d.1 — per-frame validity mask derived from the keypoint tensor.

    A frame is "valid" if either hand block contributes a non-zero coord.
    Zero rows occur in two cases:
      1. MediaPipe missed both hands on that frame (recorded by ``clean.py``).
      2. The clip was padded with zeros to reach ``TEMPORAL_LENGTH``.

    Both cases are noise as far as the classifier is concerned and should
    be excluded from the temporal pool / attention. Returns a float
    tensor of shape ``(B, T)`` with values in ``{0., 1.}``.
    """
    hand_block = x[..., :_HAND_BLOCK_FLOATS]  # (B, T, 126)
    return (hand_block.abs().sum(dim=-1) > 0).to(x.dtype)


class BiLSTMClassifier(nn.Module):
    """2-layer bidirectional LSTM over keypoint sequences.

    Input:  (B, T, K)  float32 — (batch, temporal, keypoint floats)
    Output: (B, N)     float32 — logits over the N-class vocabulary
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.input_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, K)
        valid = _frame_valid_mask(x).unsqueeze(-1)  # (B, T, 1) — see 9d.1
        x = self.input_norm(x)
        out, _ = self.lstm(x)  # (B, T, 2H)
        # Masked mean-pool over time: zeroed (MediaPipe-miss / padded)
        # frames don't get a vote. If a clip is *all* missed (degenerate
        # edge case), fall back to the unmasked mean to keep the loss
        # finite — 9a.5 drops these in practice, but defense in depth.
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        all_missed = (valid.sum(dim=1) == 0).expand_as(valid_sum)
        masked = (out * valid).sum(dim=1) / valid_sum
        unmasked = out.mean(dim=1)
        pooled = torch.where(all_missed, unmasked, masked)
        logits = self.head(pooled)
        return logits


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017). No learned weights."""

    def __init__(self, d_model: int, max_len: int = 64) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    """Small Transformer encoder over keypoint sequences.

    Adopted if BiLSTM underperforms during Phase 4 iteration.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, K) → (B, T, d_model)
        # Phase 9d.1: derive a per-frame validity mask, then express it
        # as the encoder's src_key_padding_mask (True = ignore). The CLS
        # token is always attended (prepended as False).
        invalid = _frame_valid_mask(x) == 0  # (B, T) bool
        h = self.input_proj(x)
        cls = self.cls_token.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.pos_enc(h)
        cls_keep = torch.zeros(h.size(0), 1, dtype=torch.bool, device=x.device)
        key_padding_mask = torch.cat([cls_keep, invalid], dim=1)  # (B, 1+T)
        # If every keypoint frame is invalid the CLS token is the only
        # unmasked position — that still produces a finite logit because
        # the encoder layer's self-attention degenerates to identity on
        # CLS alone. Defense in depth against 9a.5 edge cases.
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        logits = self.head(h[:, 0])
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
