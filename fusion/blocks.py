"""Building blocks for the SAR-Optical fusion model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _norm(channels: int) -> nn.Module:
    return nn.GroupNorm(8, channels) if channels % 8 == 0 else nn.BatchNorm2d(channels)


class GatedConv2d(nn.Module):
    """Mask-aware gated convolution: the gate learns where content is valid."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel // 2
        self.feature = nn.Conv2d(in_ch, out_ch, kernel, stride, padding)
        self.gate = nn.Conv2d(in_ch, out_ch, kernel, stride, padding)
        self.norm = _norm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = F.silu(self.feature(x))
        gate = torch.sigmoid(self.gate(x))
        return self.norm(feat * gate)


class GatedBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(GatedConv2d(in_ch, out_ch), GatedConv2d(out_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _norm(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _norm(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class CrossModalFusionGate(nn.Module):
    """Fuse optical and SAR features.

    A spatial gate (conditioned on optical features, SAR features and the
    downsampled cloud mask) decides, per pixel and channel, how much to pull
    from the SAR-informed merged features versus the original optical features.
    In clouded pixels the gate opens toward SAR (which sees through cloud).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.merge = nn.Conv2d(channels * 2, channels, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, channels, 1),
            nn.Sigmoid(),
        )
        self.refine = ConvBlock(channels, channels)

    def forward(self, optical: torch.Tensor, sar: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.shape[-2:] != optical.shape[-2:]:
            mask = F.interpolate(mask, size=optical.shape[-2:], mode="bilinear", align_corners=False)
        merged = self.merge(torch.cat([optical, sar], dim=1))
        gate = self.gate(torch.cat([optical, sar, mask], dim=1))
        fused = optical * (1.0 - gate) + merged * gate
        return self.refine(fused)
