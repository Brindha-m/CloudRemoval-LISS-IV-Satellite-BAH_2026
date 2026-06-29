"""Conditional 70x70 PatchGAN discriminator.

It scores local realism of the optical image conditioned on the SAR input,
so the generator must produce content consistent with the radar structure.
"""

from __future__ import annotations

import torch
from torch import nn


class PatchDiscriminator(nn.Module):
    def __init__(self, optical_ch: int = 3, sar_ch: int = 2, base: int = 64) -> None:
        super().__init__()
        in_ch = optical_ch + sar_ch

        def block(i: int, o: int, stride: int, norm: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv2d(i, o, 4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_ch, base, stride=2, norm=False),
            *block(base, base * 2, stride=2),
            *block(base * 2, base * 4, stride=2),
            *block(base * 4, base * 8, stride=1),
            nn.Conv2d(base * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([optical, sar], dim=1))
