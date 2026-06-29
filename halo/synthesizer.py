"""Latent Terrain Synthesizer.

Generation is used *only* where the surface is physically unrecoverable
(opaque/thick cloud). The synthesizer is a gated-convolution encoder-decoder
that inpaints the hole, conditioned on:

  - the reclaimer's recovered image (valid surroundings),
  - a hole mask (1 = synthesize here),
  - SAR backscatter (cloud-penetrating structural anchor).

Gated convolutions let the network learn where information is valid, which is
ideal for irregular cloud holes.
"""

from __future__ import annotations

import torch
from torch import nn

from .blocks import GatedConv2d


class LatentTerrainSynthesizer(nn.Module):
    def __init__(self, optical_ch: int = 3, sar_ch: int = 2, base: int = 32) -> None:
        super().__init__()
        in_ch = optical_ch + 1 + sar_ch  # recovered + hole mask + SAR

        self.enc1 = GatedConv2d(in_ch, base)
        self.enc2 = GatedConv2d(base, base * 2, stride=2)
        self.enc3 = GatedConv2d(base * 2, base * 4, stride=2)

        self.bottleneck = nn.Sequential(
            GatedConv2d(base * 4, base * 4),
            GatedConv2d(base * 4, base * 4),
        )

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = GatedConv2d(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = GatedConv2d(base * 2, base)
        self.out = nn.Conv2d(base, optical_ch, 1)

    def forward(
        self,
        recovered: torch.Tensor,
        hole_mask: torch.Tensor,
        sar: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([recovered * (1.0 - hole_mask), hole_mask, sar], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))
