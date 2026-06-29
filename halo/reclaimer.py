"""Radiometric Reclaimer.

Key idea: thin clouds and haze are *physically invertible*. Under the
atmospheric scattering model an observed pixel is

    I = J * tau + A * (1 - tau)

where J is the true surface reflectance, tau is transmittance, and A is the
airlight / cloud radiance. Where tau is not too small we can recover J
analytically instead of hallucinating it:

    J_hat = (I - A * (1 - tau)) / clamp(tau, eps)

A small residual CNN then cleans up inversion noise. This keeps thin-cloud
regions spectrally faithful and reserves generation only for opaque areas.
"""

from __future__ import annotations

import torch
from torch import nn

from .blocks import TinyUNet


class RadiometricReclaimer(nn.Module):
    def __init__(self, optical_ch: int = 3, base: int = 24, tau_floor: float = 0.15) -> None:
        super().__init__()
        self.tau_floor = tau_floor
        # Refinement sees the analytic estimate, the original, and tau.
        self.refine = TinyUNet(optical_ch * 2 + 1, optical_ch, base=base, depth=2)

    def analytic_inverse(
        self,
        observed: torch.Tensor,
        tau: torch.Tensor,
        airlight: torch.Tensor,
    ) -> torch.Tensor:
        tau_safe = tau.clamp(min=self.tau_floor)
        recovered = (observed - airlight * (1.0 - tau)) / tau_safe
        return recovered.clamp(0.0, 1.0)

    def forward(
        self,
        observed: torch.Tensor,
        tau: torch.Tensor,
        airlight: torch.Tensor,
    ) -> torch.Tensor:
        analytic = self.analytic_inverse(observed, tau, airlight)
        residual = self.refine(torch.cat([analytic, observed, tau], dim=1))
        return (analytic + residual).clamp(0.0, 1.0)
