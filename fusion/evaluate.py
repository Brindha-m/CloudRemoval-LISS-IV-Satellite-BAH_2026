"""Evaluation metrics for cloud-free reconstruction.

PSNR, SSIM and SAM. Uses scikit-image when available, otherwise falls back to
NumPy implementations so it runs anywhere. Metrics can be restricted to the
cloud-masked region (where reconstruction actually happened).
"""

from __future__ import annotations

import numpy as np


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def _ssim_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    c1, c2 = (0.01) ** 2, (0.03) ** 2
    mu1, mu2 = pred.mean(), target.mean()
    var1, var2 = pred.var(), target.var()
    cov = ((pred - mu1) * (target - mu2)).mean()
    return float(((2 * mu1 * mu2 + c1) * (2 * cov + c2))
                 / ((mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2)))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity

        return float(
            structural_similarity(
                target, pred, channel_axis=0, data_range=1.0
            )
        )
    except Exception:
        return _ssim_numpy(pred, target)


def sam(pred: np.ndarray, target: np.ndarray) -> float:
    """Spectral Angle Mapper in degrees, averaged over pixels. Inputs [C,H,W]."""
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    dot = (p * t).sum(axis=0)
    norm = np.linalg.norm(p, axis=0) * np.linalg.norm(t, axis=0) + 1e-8
    cosine = np.clip(dot / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)).mean())


def evaluate_pair(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """pred/target: [C,H,W] in [0,1]. mask: [1,H,W] limits metrics to cloud pixels."""
    metrics = {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
        "sam": sam(pred, target),
    }
    if mask is not None and mask.sum() > 0:
        m = mask[0] > 0.5
        pred_m = pred[:, m]
        target_m = target[:, m]
        metrics["psnr_cloud"] = psnr(pred_m, target_m)
        metrics["sam_cloud"] = sam(pred_m[:, None], target_m[:, None])
    return metrics
