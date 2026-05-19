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
        x = self.input_norm(x)
        out, _ = self.lstm(x)  # (B, T, 2H)
        pooled = out.mean(dim=1)  # mean-pool over time
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
        h = self.input_proj(x)
        # Prepend CLS token
        cls = self.cls_token.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.pos_enc(h)
        h = self.encoder(h)
        logits = self.head(h[:, 0])  # CLS-token pool
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
