"""Reusable neural building blocks for the HALO framework."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch) if out_ch % 8 == 0 else nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch) if out_ch % 8 == 0 else nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    """Compact U-Net used by several HALO heads."""

    def __init__(self, in_ch: int, out_ch: int, base: int = 32, depth: int = 3) -> None:
        super().__init__()
        self.depth = depth
        self.inc = ConvBlock(in_ch, base)

        self.downs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        channels = base
        for _ in range(depth):
            self.downs.append(nn.MaxPool2d(2))
            self.down_convs.append(ConvBlock(channels, channels * 2))
            channels *= 2

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(nn.ConvTranspose2d(channels, channels // 2, 2, stride=2))
            self.up_convs.append(ConvBlock(channels, channels // 2))
            channels //= 2

        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        cur = skips[-1]
        for pool, conv in zip(self.downs, self.down_convs):
            cur = conv(pool(cur))
            skips.append(cur)

        cur = skips.pop()
        for up, conv in zip(self.ups, self.up_convs):
            cur = up(cur)
            skip = skips.pop()
            cur = conv(torch.cat([cur, skip], dim=1))
        return self.outc(cur)


class GatedConv2d(nn.Module):
    """Gated convolution (Yu et al.) - the gate learns where content is valid."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel // 2
        self.feature = nn.Conv2d(in_ch, out_ch, kernel, stride, padding)
        self.gate = nn.Conv2d(in_ch, out_ch, kernel, stride, padding)
        self.norm = nn.GroupNorm(8, out_ch) if out_ch % 8 == 0 else nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        gate = torch.sigmoid(self.gate(x))
        return self.norm(F.silu(feat) * gate)


def fractal_noise(
    batch: int,
    height: int,
    width: int,
    octaves: int = 5,
    persistence: float = 0.55,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate value-noise fractal fields in [0, 1] using summed upsampled octaves."""
    field = torch.zeros(batch, 1, height, width, device=device)
    amplitude = 1.0
    total = 0.0
    for octave in range(octaves):
        cells = 2 ** (octave + 1)
        grid = torch.rand(batch, 1, cells, cells, device=device, generator=generator)
        upsampled = F.interpolate(grid, size=(height, width), mode="bilinear", align_corners=False)
        field = field + amplitude * upsampled
        total += amplitude
        amplitude *= persistence
    field = field / max(total, 1e-6)
    field = (field - field.amin(dim=(2, 3), keepdim=True)) / (
        field.amax(dim=(2, 3), keepdim=True) - field.amin(dim=(2, 3), keepdim=True) + 1e-6
    )
    return field
