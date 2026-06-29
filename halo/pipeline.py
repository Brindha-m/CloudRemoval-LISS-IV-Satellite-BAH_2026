"""HALO end-to-end pipeline.

HALO = Hybrid Atmospheric-Latent Optical reconstruction.

Flow (deliberately different from a flat detect/fuse/reconstruct/postprocess
stack):

    observed optical + SAR
        -> Cloud Cartographer        (continuous opacity tau, airlight)
        -> Radiometric Reclaimer     (physics inversion for thin cloud/haze)
        -> Latent Terrain Synthesizer(generation only inside opaque holes)
        -> Harmonized blend          (opacity-feathered composite)

The blend is the novelty glue: thin regions keep physically-recovered pixels,
opaque regions take synthesized content, and a feathered tau weight removes
seams.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .cartographer import CloudCartographer
from .reclaimer import RadiometricReclaimer
from .synthesizer import LatentTerrainSynthesizer


def feather(mask: torch.Tensor, iterations: int = 2) -> torch.Tensor:
    soft = mask
    for _ in range(iterations):
        soft = F.avg_pool2d(soft, kernel_size=3, stride=1, padding=1)
    return soft.clamp(0.0, 1.0)


class HaloPipeline(nn.Module):
    def __init__(self, optical_ch: int = 3, sar_ch: int = 2, thick_hole_tau: float = 0.30) -> None:
        super().__init__()
        self.thick_hole_tau = thick_hole_tau
        self.cartographer = CloudCartographer(optical_ch, sar_ch)
        self.reclaimer = RadiometricReclaimer(optical_ch)
        self.synthesizer = LatentTerrainSynthesizer(optical_ch, sar_ch)

    def forward(self, observed: torch.Tensor, sar: torch.Tensor) -> dict[str, torch.Tensor]:
        maps = self.cartographer(observed, sar)
        tau = maps["tau"]
        airlight = maps["airlight"]

        recovered = self.reclaimer(observed, tau, airlight)

        hole = (tau < self.thick_hole_tau).float()
        hole_soft = feather(hole)
        synthesized = self.synthesizer(recovered, hole_soft, sar)

        output = recovered * (1.0 - hole_soft) + synthesized * hole_soft
        output = output.clamp(0.0, 1.0)

        return {
            "output": output,
            "recovered": recovered,
            "synthesized": synthesized,
            "tau": tau,
            "cloud_prob": maps["cloud_prob"],
            "shadow_prob": maps["shadow_prob"],
            "airlight": airlight,
            "hole": hole,
            "hole_soft": hole_soft,
        }
