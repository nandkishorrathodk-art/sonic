"""Fast Multi-Scale Neural Backbone for the Visual Cortex.

Processes peripheral (global) and foveal (local detail) streams using
efficient inverted residual blocks with Squeeze-and-Excitation (SE) and
multi-scale spatial fusion.

Designed for sub-15ms inference on modern GPUs, while remaining efficient on CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Standard Conv2d + BatchNorm2d + GELU/SiLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SqueezeExcitation(nn.Module):
    """Channel attention to recalibrate spatial features."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1),
            nn.GELU(),
            nn.Conv2d(reduced, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class InvertedResidual(nn.Module):
    """Inverted residual block with depthwise separation and SE attention."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand_ratio: int = 2) -> None:
        super().__init__()
        self.use_res = (stride == 1 and in_ch == out_ch)
        hidden = int(round(in_ch * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_ch, hidden, kernel_size=1, padding=0))

        # Depthwise
        layers.append(ConvBNAct(hidden, hidden, kernel_size=3, stride=stride, padding=1, groups=hidden))
        layers.append(SqueezeExcitation(hidden))
        # Pointwise linear
        layers.append(ConvBNAct(hidden, out_ch, kernel_size=1, padding=0, act=False))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res:
            return x + self.block(x)
        return self.block(x)


class CorticalBackbone(nn.Module):
    """Dual-path feature extractor for peripheral and foveal vision streams."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        # Peripheral Stream: 256x256 -> 16x16
        self.stem = ConvBNAct(in_channels, 32, stride=2) # 128x128
        self.stage1 = InvertedResidual(32, 64, stride=2)  # 64x64
        self.stage2 = InvertedResidual(64, 96, stride=2)  # 32x32
        self.stage3 = InvertedResidual(96, feature_dim, stride=2) # 16x16

        # Foveal Stream: high-detail feature extractor (shares stage3 capacity)
        self.foveal_stem = ConvBNAct(in_channels, 32, stride=2)
        self.foveal_stage1 = InvertedResidual(32, 64, stride=2)
        self.foveal_stage2 = InvertedResidual(64, 96, stride=2)
        self.foveal_stage3 = InvertedResidual(96, feature_dim, stride=2)

        # Multi-scale Cross-Fusion: blends foveal details into peripheral map
        self.fusion = nn.Sequential(
            ConvBNAct(feature_dim * 2, feature_dim, kernel_size=1, padding=0),
            InvertedResidual(feature_dim, feature_dim, stride=1),
        )

    def forward(
        self,
        peripheral: torch.Tensor,
        foveal: torch.Tensor,
        fixation: tuple[float, float] = (0.5, 0.5),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (fused_features, peripheral_features, foveal_features).

        Output shapes:
        - fused_features: (B, feature_dim, 16, 16) - unified conscious visual field
        - peripheral_features: (B, feature_dim, 16, 16)
        - foveal_features: (B, feature_dim, 16, 16)
        """
        # 1. Forward Peripheral
        p = self.stem(peripheral)
        p = self.stage1(p)
        p = self.stage2(p)
        feat_p = self.stage3(p) # (B, dim, 16, 16)

        # 2. Forward Foveal
        f = self.foveal_stem(foveal)
        f = self.foveal_stage1(f)
        f = self.foveal_stage2(f)
        feat_f = self.foveal_stage3(f) # (B, dim, 16, 16)

        # 3. Dynamic Spatial Fusion
        # Foveal feature map is pooled and aligned to peripheral context
        feat_combined = torch.cat([feat_p, feat_f], dim=1)
        feat_fused = self.fusion(feat_combined)

        return feat_fused, feat_p, feat_f

