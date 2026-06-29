"""Cloud Forge - the self-supervised data engine.

This removes the need for real paired cloudy/clear LISS-IV data. It:

  1. Generates procedural clean multispectral scenes (Green, Red, NIR) with
     vegetation-like spectral correlations, plus a matching SAR pair.
  2. Forges a continuous cloud opacity field (tau) and airlight.
  3. Composites a cloudy observation through the physics model
     I = J * tau + A * (1 - tau).

The same forge supplies ground-truth tau and hole masks, so every HALO module
is trained with exact supervision. Coverage and thickness are curriculum
controllable. At real-deployment time you swap the procedural clean scene for
real clear LISS-IV tiles - the forging logic is identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .blocks import fractal_noise


@dataclass
class ForgeConfig:
    coverage: float = 0.4          # fraction of scene affected by cloud
    max_thickness: float = 0.9     # how opaque the thickest cloud gets (1 - tau)
    thick_hole_tau: float = 0.30   # tau below this is treated as unrecoverable
    softness: float = 0.06         # transition softness of cloud edges


def generate_clean_scene(
    batch: int, height: int, width: int, device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (optical[B,3,H,W] as G/R/NIR, sar[B,2,H,W] as VV/VH)."""
    vegetation = fractal_noise(batch, height, width, octaves=5, device=device, generator=generator)
    moisture = fractal_noise(batch, height, width, octaves=4, device=device, generator=generator)

    nir = 0.25 + 0.55 * vegetation
    red = 0.18 + 0.20 * (1.0 - vegetation) + 0.05 * moisture
    green = 0.20 + 0.18 * (1.0 - vegetation) + 0.04 * moisture
    optical = torch.cat([green, red, nir], dim=1).clamp(0.0, 1.0)

    # SAR roughly tracks structural edges plus speckle.
    grad_x = torch.abs(vegetation[:, :, :, 1:] - vegetation[:, :, :, :-1]).mean()
    edges = fractal_noise(batch, height, width, octaves=6, device=device, generator=generator)
    speckle = torch.rand(batch, 2, height, width, device=device, generator=generator)
    structure = (0.5 * edges + 0.5 * vegetation)
    vv = (structure + 0.15 * speckle[:, 0:1] + 0.05 * grad_x).clamp(0.0, 1.0)
    vh = (0.7 * structure + 0.2 * speckle[:, 1:2]).clamp(0.0, 1.0)
    sar = torch.cat([vv, vh], dim=1)
    return optical, sar


def generate_opacity(
    batch: int, height: int, width: int, config: ForgeConfig,
    device: torch.device | str = "cpu", generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return transmittance tau[B,1,H,W] in [0,1] (1 = fully clear)."""
    field = fractal_noise(batch, height, width, octaves=4, persistence=0.6, device=device, generator=generator)
    threshold = 1.0 - config.coverage
    cloud_amount = torch.sigmoid((field - threshold) / max(config.softness, 1e-3))
    thickness = fractal_noise(batch, height, width, octaves=3, device=device, generator=generator)
    opacity = cloud_amount * (0.3 + 0.7 * thickness) * config.max_thickness
    tau = (1.0 - opacity).clamp(0.0, 1.0)
    return tau


def forge_batch(
    batch: int, height: int, width: int, config: ForgeConfig,
    device: torch.device | str = "cpu", generator: torch.Generator | None = None,
    clean_optical: torch.Tensor | None = None, sar: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if clean_optical is None or sar is None:
        clean_optical, sar = generate_clean_scene(batch, height, width, device, generator)

    optical_ch = clean_optical.shape[1]
    tau = generate_opacity(batch, height, width, config, device, generator)
    airlight = (0.75 + 0.25 * torch.rand(batch, optical_ch, 1, 1, device=device, generator=generator))

    cloudy = (clean_optical * tau + airlight * (1.0 - tau)).clamp(0.0, 1.0)
    thick_hole = (tau < config.thick_hole_tau).float()

    return {
        "cloudy": cloudy,
        "clean": clean_optical,
        "sar": sar,
        "tau": tau,
        "airlight": airlight,
        "thick_hole": thick_hole,
        "cloud_prob": (1.0 - tau),
    }


def curriculum_config(progress: float) -> ForgeConfig:
    """Ramp difficulty: small/thin clouds early, large/thick clouds later.

    progress in [0, 1].
    """
    progress = max(0.0, min(1.0, progress))
    return ForgeConfig(
        coverage=0.15 + 0.45 * progress,
        max_thickness=0.55 + 0.40 * progress,
        thick_hole_tau=0.30,
        softness=0.06,
    )
