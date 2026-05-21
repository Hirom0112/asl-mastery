"""CenterNet-style from-scratch hand detector.

Architecture per docs/VOCABULARY_TRAINER_ROADMAP.md Phase 1 Slice 1.2:
- ~6 conv blocks with residual connections + batch norm (~1-3M params)
- CenterNet-style head: heatmap of hand centers (1 channel) + size
  regression (2 channels: width, height in pixels)
- Input: 320 x 320 RGB
- Output stride: 4 (heatmap is 80 x 80)

Loss is defined separately in losses.py. Training loop in train.py.
No torchvision.models. No pretrained init. Kaiming-normal for conv,
zeros for biases, ones for BN weights.
"""

from __future__ import annotations

import torch
from torch import nn


def _conv_bn_relu(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            _conv_bn_relu(in_ch, out_ch, stride=stride),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.body(x) + self.shortcut(x))


class HandDetector(nn.Module):
    """CenterNet-style anchor-free hand detector.

    Input:  (B, 3, 320, 320) RGB in [0, 1]
    Output: dict with
        - heatmap: (B, 1, 80, 80)   logits, no sigmoid applied
        - size:    (B, 2, 80, 80)   predicted (width, height) in pixels
                                    at input resolution
    """

    # Output stride from 320 input down to 80 feature map.
    STRIDE = 4

    def __init__(self) -> None:
        super().__init__()

        # Stem: 320 -> 160
        self.stem = nn.Sequential(
            _conv_bn_relu(3, 32, stride=2),
            _conv_bn_relu(32, 32, stride=1),
        )

        # Block 1: 160 -> 80
        self.block1 = nn.Sequential(
            _ResidualBlock(32, 64, stride=2),
            _ResidualBlock(64, 64),
        )

        # Block 2: 80 -> 40
        self.block2 = nn.Sequential(
            _ResidualBlock(64, 128, stride=2),
            _ResidualBlock(128, 128),
        )

        # Block 3: 40 -> 20
        self.block3 = nn.Sequential(
            _ResidualBlock(128, 192, stride=2),
            _ResidualBlock(192, 192),
        )

        # Upsample 20 -> 40 -> 80 with lateral skips
        self.up3to2 = nn.Sequential(
            nn.ConvTranspose2d(192, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.fuse2 = _conv_bn_relu(128, 128)

        self.up2to1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.fuse1 = _conv_bn_relu(64, 64)

        # Heads: small 3x3 then 1x1
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        self.size_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
        )

        self._init_from_scratch()

    def _init_from_scratch(self) -> None:
        # Every weight initialized here. No load_state_dict for foreign
        # weights anywhere downstream. Verified by the no-pretrained
        # audit in ADR 0012.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Heatmap bias init for focal-loss stability: predict low
        # foreground probability at start. Matches CenterNet practice.
        final_heatmap_conv = self.heatmap_head[-1]
        nn.init.constant_(final_heatmap_conv.bias, -2.19)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s = self.stem(x)        # (B, 32, 160, 160)
        c1 = self.block1(s)     # (B, 64, 80, 80)
        c2 = self.block2(c1)    # (B, 128, 40, 40)
        c3 = self.block3(c2)    # (B, 192, 20, 20)

        u2 = self.fuse2(self.up3to2(c3) + c2)   # (B, 128, 40, 40)
        u1 = self.fuse1(self.up2to1(u2) + c1)   # (B, 64, 80, 80)

        return {
            "heatmap": self.heatmap_head(u1),
            "size": self.size_head(u1),
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke check: model builds, forward pass shapes are right,
    # parameter count is in the 1-3M range targeted by the roadmap.
    model = HandDetector()
    n = count_parameters(model)
    x = torch.randn(2, 3, 320, 320)
    out = model(x)
    assert out["heatmap"].shape == (2, 1, 80, 80), out["heatmap"].shape
    assert out["size"].shape == (2, 2, 80, 80), out["size"].shape
    print(f"HandDetector OK. parameters={n:,} (~{n/1e6:.2f}M)")
