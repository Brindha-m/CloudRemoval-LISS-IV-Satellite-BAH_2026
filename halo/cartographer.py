"""Cloud Cartographer.

Rather than a binary cloud mask, this head estimates a *continuous opacity
field*: a transmittance map (tau in [0,1], 1 = fully clear) together with
cloud and shadow probabilities, and a per-channel airlight estimate.

These are the physical quantities the Radiometric Reclaimer needs to invert
the atmospheric scattering model.
"""

from __future__ import annotations

import torch
from torch import nn

from .blocks import TinyUNet


class CloudCartographer(nn.Module):
    def __init__(self, optical_ch: int = 3, sar_ch: int = 2, base: int = 32) -> None:
        super().__init__()
        self.backbone = TinyUNet(optical_ch + sar_ch, base, base=base, depth=3)
        self.tau_head = nn.Conv2d(base, 1, 1)
        self.cloud_head = nn.Conv2d(base, 1, 1)
        self.shadow_head = nn.Conv2d(base, 1, 1)
        # Airlight: one scalar per optical channel, pooled from global context.
        self.airlight_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base, optical_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(torch.cat([optical, sar], dim=1))
        tau = torch.sigmoid(self.tau_head(features))
        cloud_prob = torch.sigmoid(self.cloud_head(features))
        shadow_prob = torch.sigmoid(self.shadow_head(features))
        airlight = self.airlight_head(features)  # [B, C, 1, 1]
        return {
            "tau": tau,
            "cloud_prob": cloud_prob,
            "shadow_prob": shadow_prob,
            "airlight": airlight,
        }
