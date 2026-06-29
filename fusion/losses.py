"""Loss functions for SAR-Optical fusion cloud removal.

L_total = w1*L1 + w2*perceptual + w3*adversarial + w4*spectral + w5*(1-SSIM)

The perceptual term uses VGG16 features when torchvision is available and is
otherwise skipped, so training still runs in minimal environments.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class LossWeights:
    l1: float = 1.0
    perceptual: float = 0.5
    adversarial: float = 0.5
    spectral: float = 0.3
    ssim: float = 0.3


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = 1.0 + 4.0 * mask
    return ((pred - target).abs() * weight).mean()


def visible_l1(pred: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Match reference on cloud-free pixels only (cloudy-only / self-supervised training)."""
    keep = 1.0 - mask
    diff = (pred - reference).abs() * keep
    return diff.sum() / (keep.sum() * pred.shape[1] + 1e-6)


def spectral_angle(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_flat = pred.flatten(2)
    target_flat = target.flatten(2)
    cosine = F.cosine_similarity(pred_flat, target_flat, dim=1).clamp(-0.999, 0.999)
    return torch.acos(cosine).mean()


def band_statistics(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mean = pred.mean(dim=(2, 3))
    target_mean = target.mean(dim=(2, 3))
    pred_std = pred.std(dim=(2, 3))
    target_std = target.std(dim=(2, 3))
    return F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)


def spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return spectral_angle(pred, target) + band_statistics(pred, target)


def _gaussian_window(channels: int, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    channels = pred.shape[1]
    window = _gaussian_window(channels, window_size).to(pred.device, pred.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
    mu2 = F.conv2d(target, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1 = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq
    sigma2 = F.conv2d(target * target, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu1_mu2

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2)
    )
    return ssim_map.mean()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - ssim(pred, target)


def lsgan_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return 0.5 * (F.mse_loss(real_logits, torch.ones_like(real_logits))
                  + F.mse_loss(fake_logits, torch.zeros_like(fake_logits)))


def lsgan_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(fake_logits, torch.ones_like(fake_logits))


class PerceptualLoss(nn.Module):
    """VGG16 feature-space L1. Becomes a no-op if torchvision is unavailable."""

    def __init__(self) -> None:
        super().__init__()
        self.available = False
        try:
            from torchvision import models  # type: ignore

            weights = getattr(models, "VGG16_Weights", None)
            vgg = models.vgg16(weights=weights.IMAGENET1K_V1 if weights else None)
            self.slice = nn.Sequential(*list(vgg.features[:16])).eval()
            for param in self.slice.parameters():
                param.requires_grad = False
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
            self.available = True
        except Exception:
            self.slice = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.available or self.slice is None:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        pred_n = (pred[:, :3] - self.mean) / self.std
        target_n = (target[:, :3] - self.mean) / self.std
        return F.l1_loss(self.slice(pred_n), self.slice(target_n))
