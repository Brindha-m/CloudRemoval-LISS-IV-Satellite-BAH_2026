"""HALO training objectives.

Innovative pieces:
  - Opacity-weighted reconstruction: pixels are weighted by cloud opacity so
    the model focuses where the surface was actually lost.
  - Frequency-split loss: separates low-frequency radiometry from
    high-frequency texture so fine detail (vegetation/urban edges) is preserved.
  - Spectral Angle (SAM) + NDVI consistency: enforce physical spectral meaning.
  - Seam energy: penalises gradient discontinuities at hole boundaries.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _blur(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)


def opacity_weighted_l1(pred: torch.Tensor, target: torch.Tensor, opacity: torch.Tensor) -> torch.Tensor:
    weight = 1.0 + 4.0 * opacity
    return ((pred - target).abs() * weight).mean()


def frequency_split_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_low, target_low = _blur(pred), _blur(target)
    pred_high, target_high = pred - pred_low, target - target_low
    low = F.l1_loss(pred_low, target_low)
    high = F.l1_loss(pred_high, target_high)
    return low + 2.0 * high


def spectral_angle_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_flat = pred.flatten(2)
    target_flat = target.flatten(2)
    cosine = F.cosine_similarity(pred_flat, target_flat, dim=1).clamp(-0.999, 0.999)
    return torch.acos(cosine).mean()


def ndvi(image: torch.Tensor) -> torch.Tensor:
    red = image[:, 1:2]
    nir = image[:, 2:3]
    return (nir - red) / (nir + red + 1e-6)


def ndvi_consistency(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(ndvi(pred), ndvi(target))


def seam_energy(pred: torch.Tensor, target: torch.Tensor, hole_mask: torch.Tensor) -> torch.Tensor:
    """Penalise gradient mismatch, emphasised near hole boundaries."""
    boundary = (_blur(hole_mask) - hole_mask).abs()
    boundary = boundary / (boundary.amax() + 1e-6)
    weight = 1.0 + 5.0 * boundary

    def grads(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return gx, gy

    pgx, pgy = grads(pred)
    tgx, tgy = grads(target)
    wx = weight[:, :, :, 1:]
    wy = weight[:, :, 1:, :]
    return ((pgx - tgx).abs() * wx).mean() + ((pgy - tgy).abs() * wy).mean()


def cartographer_loss(
    pred_tau: torch.Tensor, gt_tau: torch.Tensor,
    pred_cloud: torch.Tensor, gt_cloud: torch.Tensor,
) -> torch.Tensor:
    return F.l1_loss(pred_tau, gt_tau) + F.binary_cross_entropy(pred_cloud, gt_cloud.clamp(0, 1))


def reclaimer_loss(recovered: torch.Tensor, clean: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    # Weight thin/medium regions (where physics should work) most.
    thin_weight = (tau > 0.3).float()
    base = (recovered - clean).abs() * (0.2 + thin_weight)
    return base.mean() + 0.5 * spectral_angle_loss(recovered, clean)


def final_loss(
    output: torch.Tensor, clean: torch.Tensor, opacity: torch.Tensor, hole_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    recon = opacity_weighted_l1(output, clean, opacity)
    freq = frequency_split_loss(output, clean)
    sam = spectral_angle_loss(output, clean)
    veg = ndvi_consistency(output, clean)
    seam = seam_energy(output, clean, hole_mask)
    total = recon + 0.5 * freq + 0.3 * sam + 0.3 * veg + 0.2 * seam
    return {"total": total, "recon": recon, "freq": freq, "sam": sam, "ndvi": veg, "seam": seam}
