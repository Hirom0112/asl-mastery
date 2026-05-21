"""From-scratch hand landmark regressor (21 keypoints).

Architecture per docs/VOCABULARY_TRAINER_ROADMAP.md Phase 2 Slice 2.2:
- Input: 224 x 224 hand crop (RGB, [0, 1])
- 5 residual conv blocks (~1-2M params total)
- GAP + FC head → 42 outputs (21 keypoints × (x, y) in [0, 1] of crop)
- Optional visibility head → 21 outputs (sigmoid) per keypoint
- All weights initialized from scratch (Kaiming-normal). No
  torchvision.models, no load_state_dict reading foreign weights.

Loss is L1 on normalized coords + BCE on visibility (when supervised).
Training loop in train_hand_landmarks.py.
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


class HandLandmarkRegressor(nn.Module):
    """Direct 2D coordinate regression for 21 hand keypoints.

    Input:  (B, 3, 224, 224)
    Output: dict with
        - coords:     (B, 21, 2)  in [0, 1] of crop (sigmoid)
        - visibility: (B, 21)     logits (caller applies sigmoid)
    """

    INPUT_SIZE = 224
    NUM_KEYPOINTS = 21

    def __init__(self, predict_visibility: bool = True) -> None:
        super().__init__()
        self.predict_visibility = predict_visibility

        # Stem 224 -> 112
        self.stem = nn.Sequential(
            _conv_bn_relu(3, 32, stride=2),
            _conv_bn_relu(32, 32),
        )

        # 112 -> 56 -> 28 -> 14 -> 7
        self.stage1 = nn.Sequential(_ResidualBlock(32, 64, stride=2), _ResidualBlock(64, 64))
        self.stage2 = nn.Sequential(_ResidualBlock(64, 128, stride=2), _ResidualBlock(128, 128))
        self.stage3 = nn.Sequential(_ResidualBlock(128, 192, stride=2), _ResidualBlock(192, 192))
        self.stage4 = nn.Sequential(_ResidualBlock(192, 256, stride=2), _ResidualBlock(256, 256))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.coord_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.NUM_KEYPOINTS * 2),
        )
        if predict_visibility:
            self.vis_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, self.NUM_KEYPOINTS),
            )

        self._init_from_scratch()

    def _init_from_scratch(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x)))))
        pooled = self.gap(f).flatten(1)  # (B, 256)
        coords = torch.sigmoid(self.coord_head(pooled)).view(-1, self.NUM_KEYPOINTS, 2)
        out = {"coords": coords}
        if self.predict_visibility:
            out["visibility"] = self.vis_head(pooled)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = HandLandmarkRegressor()
    n = count_parameters(m)
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    assert out["coords"].shape == (2, 21, 2), out["coords"].shape
    assert out["visibility"].shape == (2, 21), out["visibility"].shape
    print(f"HandLandmarkRegressor OK. parameters={n:,} (~{n/1e6:.2f}M)")
