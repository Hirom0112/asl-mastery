"""From-scratch upper-body pose regressor (8 keypoints).

Same architecture pattern as HandLandmarkRegressor — small ResNet
backbone + coord regression head — parameterized for the 8-keypoint
upper-body topology defined in
training/detectors/external_loaders/__init__.py (nose, neck,
r_shoulder, l_shoulder, r_elbow, l_elbow, r_wrist, l_wrist).

Input: 256 x 256 person crop. Output: (B, 8, 2) in [0, 1] of crop +
optional visibility logits.
"""

from __future__ import annotations

import torch
from torch import nn

from training.detectors.hand_landmarks import _ResidualBlock, _conv_bn_relu


class PoseRegressor(nn.Module):
    INPUT_SIZE = 256
    NUM_KEYPOINTS = 8

    def __init__(self, predict_visibility: bool = True) -> None:
        super().__init__()
        self.predict_visibility = predict_visibility

        self.stem = nn.Sequential(
            _conv_bn_relu(3, 32, stride=2),
            _conv_bn_relu(32, 32),
        )
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
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.NUM_KEYPOINTS),
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
        pooled = self.gap(f).flatten(1)
        coords = torch.sigmoid(self.coord_head(pooled)).view(-1, self.NUM_KEYPOINTS, 2)
        out = {"coords": coords}
        if self.predict_visibility:
            out["visibility"] = self.vis_head(pooled)
        return out


if __name__ == "__main__":
    m = PoseRegressor()
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    out = m(torch.randn(2, 3, 256, 256))
    assert out["coords"].shape == (2, 8, 2)
    print(f"PoseRegressor OK. parameters={n:,} (~{n/1e6:.2f}M)")
