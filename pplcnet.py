"""
PPLCNet Backbone for Oyster Mushroom Detection
Paper: PP-LCNet: A Lightweight CPU Convolutional Neural Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    """Standard Conv → BN → ReLU6 block."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block (DW + PW)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu6(self.bn1(self.dw(x)))
        x = F.relu6(self.bn2(self.pw(x)))
        return x


class PPLCNet(nn.Module):
    """
    Lightweight PPLCNet backbone.

    Produces a (B, 512, H/32, W/32) feature map from an RGB input.
    """

    def __init__(self):
        super().__init__()
        self.stage1 = ConvBNReLU(3, 32, k=3, s=2, p=1)       # /2
        self.stage2 = DepthwiseSeparableConv(32, 64, stride=2)  # /4
        self.stage3 = DepthwiseSeparableConv(64, 128, stride=2) # /8
        self.stage4 = DepthwiseSeparableConv(128, 256, stride=2) # /16
        self.stage5 = DepthwiseSeparableConv(256, 512, stride=2) # /32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x


class DetectionHead(nn.Module):
    """
    Global-average-pool → FC detection head.

    Output: (B, num_classes + 5)  →  [class, cx, cy, w, h, conf]
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes + 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B, 512)
        return self.fc(x)                          # (B, num_classes + 5)


class PPLCNetDetector(nn.Module):
    """Full PPLCNet detector: backbone + detection head."""

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.backbone = PPLCNet()
        self.head = DetectionHead(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
