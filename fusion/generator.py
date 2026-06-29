"""Dual-Encoder Cross-Modal Fusion U-Net (DCMF-UNet).

A multi-modal generator for SAR-Optical cloud removal:

  - Optical encoder uses mask-aware gated convolutions.
  - SAR encoder runs in parallel (SAR penetrates cloud).
  - At every scale a CrossModalFusionGate merges the two streams, opening
    toward SAR where the cloud mask indicates missing optical data.
  - An SE bottleneck adds global context.
  - A U-Net decoder reconstructs the optical image from fused skips.

The final output preserves clear pixels exactly and only replaces the
masked (cloudy) pixels with generated content.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import ConvBlock, CrossModalFusionGate, GatedBlock, SqueezeExcite


class DCMFUNet(nn.Module):
    def __init__(
        self,
        optical_ch: int = 3,
        sar_ch: int = 2,
        base: int = 32,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.pool = nn.MaxPool2d(2)

        # Optical encoder (mask-aware): input optical + mask.
        self.opt_in = GatedBlock(optical_ch + 1, base)
        self.opt_downs = nn.ModuleList()
        ch = base
        for _ in range(depth):
            self.opt_downs.append(GatedBlock(ch, ch * 2))
            ch *= 2
        opt_bottleneck = ch

        # SAR encoder.
        self.sar_in = ConvBlock(sar_ch, base)
        self.sar_downs = nn.ModuleList()
        ch = base
        for _ in range(depth):
            self.sar_downs.append(ConvBlock(ch, ch * 2))
            ch *= 2

        # Fusion gates per scale (encoder levels + bottleneck).
        self.fusions = nn.ModuleList()
        ch = base
        for _ in range(depth + 1):
            self.fusions.append(CrossModalFusionGate(ch))
            ch *= 2

        self.bottleneck = nn.Sequential(
            ConvBlock(opt_bottleneck, opt_bottleneck),
            SqueezeExcite(opt_bottleneck),
        )

        # Decoder.
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        ch = opt_bottleneck
        for _ in range(depth):
            self.ups.append(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2))
            self.dec.append(GatedBlock(ch, ch // 2))
            ch //= 2

        self.head = nn.Conv2d(base, optical_ch, 1)

    def forward(self, optical: torch.Tensor, sar: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        original = optical
        # Encoders produce features per scale.
        o = self.opt_in(torch.cat([optical, mask], dim=1))
        s = self.sar_in(sar)
        opt_feats = [o]
        sar_feats = [s]
        for opt_down, sar_down in zip(self.opt_downs, self.sar_downs):
            o = opt_down(self.pool(opt_feats[-1]))
            s = sar_down(self.pool(sar_feats[-1]))
            opt_feats.append(o)
            sar_feats.append(s)

        # Fuse at every scale.
        fused = [
            fusion(of, sf, mask)
            for fusion, of, sf in zip(self.fusions, opt_feats, sar_feats)
        ]

        x = self.bottleneck(fused[-1])
        skips = fused[:-1]
        for up, dec in zip(self.ups, self.dec):
            x = up(x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))

        generated = torch.sigmoid(self.head(x))
        # Preserve clear pixels, replace only masked (cloudy) pixels.
        return original * (1.0 - mask) + generated * mask

    def generate(self, optical: torch.Tensor, sar: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return model fill for cloud pixels only (before clear-pixel preservation)."""
        o = self.opt_in(torch.cat([optical, mask], dim=1))
        s = self.sar_in(sar)
        opt_feats = [o]
        sar_feats = [s]
        for opt_down, sar_down in zip(self.opt_downs, self.sar_downs):
            o = opt_down(self.pool(opt_feats[-1]))
            s = sar_down(self.pool(sar_feats[-1]))
            opt_feats.append(o)
            sar_feats.append(s)
        fused = [fusion(of, sf, mask) for fusion, of, sf in zip(self.fusions, opt_feats, sar_feats)]
        x = self.bottleneck(fused[-1])
        skips = fused[:-1]
        for up, dec in zip(self.ups, self.dec):
            x = up(x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
        return torch.sigmoid(self.head(x))
