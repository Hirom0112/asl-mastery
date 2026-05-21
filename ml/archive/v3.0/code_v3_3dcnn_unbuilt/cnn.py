"""Small R(2+1)D-style 3D CNN, trained from scratch.

Per docs/MODEL.md §1 and ADR 0001 / ADR 0010. Replaces the BiLSTM /
Transformer architectures that shipped under the now-superseded
ADR 0006 (those files —  training/classifier/model.py and the
keypoint plumbing — are removed in T3).

Architecture:
- Stem: spatial (1×7×7) → temporal (3×1×1), 3 → 32 → 64 channels.
- Four res-stages with R(2+1)D blocks; channel ladder
  [64, 128, 256, 512]; spatial stride-2 at the start of stages 2–4.
- Global spatiotemporal average pool → linear classifier head.

Each R(2+1)D block factors a full 3D convolution into a 1×3×3
spatial conv composed with a 3×1×1 temporal conv (Tran et al. 2018,
"A Closer Look at Spatiotemporal Convolutions"). At our input scale
(`T=16`, `H=W=96`) the model lands at ~7.5M parameters, inside the
5–10M target band from docs/MODEL.md §1.

Initialization is Kaiming-normal for every conv/linear layer; biases
zero; BN scale 1, BN bias 0. There is no `load_state_dict` call. There
is no external classifier weight URL. The audit surface for the
no-pretrained-pipeline claim (docs/MODEL.md §7) is this file plus
the absence of any pretrained-weight import.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class R2Plus1DBlock(nn.Module):
    """One R(2+1)D residual block.

    in_ch → out_ch with optional spatial stride. Spatial conv runs
    first, then temporal, then residual add → ReLU. A 1×1×1 projection
    matches dimensions when in_ch != out_ch or stride != 1.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.spatial = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, 1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.temporal = nn.Conv3d(
            out_ch,
            out_ch,
            kernel_size=(3, 1, 1),
            stride=(1, 1, 1),
            padding=(1, 0, 0),
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.proj: nn.Module | None = None
        if in_ch != out_ch or stride != 1:
            self.proj = nn.Sequential(
                nn.Conv3d(
                    in_ch,
                    out_ch,
                    kernel_size=(1, 1, 1),
                    stride=(1, stride, stride),
                    bias=False,
                ),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.proj is None else self.proj(x)
        out = F.relu(self.bn1(self.spatial(x)), inplace=True)
        out = self.bn2(self.temporal(out))
        return F.relu(out + identity, inplace=True)


class SmallR2Plus1D(nn.Module):
    """End-to-end small R(2+1)D-style 3D CNN.

    Input convention from the camera capture path (lib/inference/classifier.ts):
    `(B, T, H, W, 3)` float32 in `[0, 1]`. This module permutes to PyTorch's
    canonical `(B, C, T, H, W)` at the entry; the permute carries through
    to ONNX export so the wire format stays channel-last and the browser
    side never has to know about internal layout.
    """

    def __init__(
        self,
        num_classes: int,
        input_height: int = 96,
        input_width: int = 96,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width

        self.stem = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(R2Plus1DBlock(64, 64), R2Plus1DBlock(64, 64))
        self.stage2 = nn.Sequential(R2Plus1DBlock(64, 128, stride=2), R2Plus1DBlock(128, 128))
        self.stage3 = nn.Sequential(R2Plus1DBlock(128, 256, stride=2), R2Plus1DBlock(256, 256))
        self.stage4 = nn.Sequential(R2Plus1DBlock(256, 512, stride=2), R2Plus1DBlock(512, 512))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(512, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-normal everywhere; BN scale 1; biases 0. No load_state_dict."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, 3) → permute to (B, C=3, T, H, W) for Conv3d.
        if x.dim() != 5:
            raise ValueError(f"expected (B, T, H, W, 3), got shape {tuple(x.shape)}")
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
