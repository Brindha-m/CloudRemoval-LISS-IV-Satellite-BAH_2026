"""Lightweight cloud mask estimation for LISS-IV-style optical tiles.

Uses a probabilistic cloud score (Nimbo-style) rather than a hard binary mask,
so semi-transparent clouds can be tuned with a threshold slider.
"""

from __future__ import annotations

import numpy as np


def estimate_cloud_probability(
    optical: np.ndarray,
    brightness_threshold: float = 0.50,
    whiteness_threshold: float = 0.82,
) -> np.ndarray:
    """Return soft cloud probability [1, H, W] in [0, 1] (G, R, NIR input).

    Nimbo-style: brightness + whiteness + low saturation, plus cloud-shadow hints.
    """
    green, red, nir = optical[0], optical[1], optical[2]
    brightness = optical.mean(axis=0)
    whiteness = 1.0 - optical.std(axis=0)
    # Clouds are bright in all bands with low chroma.
    max_band = np.maximum(np.maximum(green, red), nir)
    min_band = np.minimum(np.minimum(green, red), nir)
    saturation = (max_band - min_band) / (max_band + 1e-6)
    low_chroma = 1.0 - np.clip(saturation / 0.22, 0, 1)

    bright_score = np.clip((brightness - (brightness_threshold - 0.14)) / 0.14, 0, 1)
    white_score = np.clip((whiteness - (whiteness_threshold - 0.12)) / 0.12, 0, 1)
    cloud_prob = bright_score * white_score * (0.55 + 0.45 * low_chroma)

    # Cloud shadows: dark, bluish, often adjacent to bright cloud pixels.
    shadow_dark = np.clip((0.38 - brightness) / 0.18, 0, 1)
    shadow_blue = np.clip((green - red) / 0.12, 0, 1) * np.clip((green - nir) / 0.10, 0, 1)
    shadow_prob = shadow_dark * (0.35 + 0.65 * shadow_blue)
    try:
        from scipy.ndimage import maximum_filter

        cloud_neighbour = maximum_filter(cloud_prob, size=9)
        shadow_prob = shadow_prob * np.clip(cloud_neighbour * 2.5, 0, 1)
    except ImportError:
        pass

    prob = np.clip(np.maximum(cloud_prob, shadow_prob * 0.85), 0, 1)
    return prob.astype(np.float32)[None]


def estimate_cloud_mask(
    optical: np.ndarray,
    brightness_threshold: float = 0.55,
    whiteness_threshold: float = 0.85,
    probability_threshold: float = 0.45,
) -> np.ndarray:
    """Binary mask derived from cloud probability."""
    prob = estimate_cloud_probability(optical, brightness_threshold, whiteness_threshold)
    return (prob >= probability_threshold).astype(np.float32)


def feather_probability(prob: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Soften mask edges to avoid halos (Nimbo-style gradual cloud removal)."""
    if sigma <= 0:
        return prob.astype(np.float32)
    try:
        from scipy.ndimage import gaussian_filter

        if prob.ndim == 3:
            return gaussian_filter(prob, sigma=(0, sigma, sigma)).astype(np.float32)
        return gaussian_filter(prob, sigma=sigma).astype(np.float32)
    except ImportError:
        return prob.astype(np.float32)


def apply_probability_threshold(prob: np.ndarray, threshold: float) -> np.ndarray:
    """Soft cut-off above threshold (Nimbo adjustable opacity)."""
    if prob.ndim == 3:
        prob = prob[0]
    threshold = float(np.clip(threshold, 0.0, 0.95))
    scaled = np.clip((prob - threshold) / (1.0 - threshold + 1e-6), 0, 1)
    return feather_probability(scaled, sigma=0.8)


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Simple binary dilation to capture cloud edges/halos (no SciPy needed)."""
    out = mask.copy()
    for _ in range(iterations):
        padded = np.pad(out, ((0, 0), (1, 1), (1, 1)), mode="edge")
        neighbours = (
            padded[:, :-2, 1:-1] + padded[:, 2:, 1:-1]
            + padded[:, 1:-1, :-2] + padded[:, 1:-1, 2:]
            + out
        )
        out = (neighbours > 0).astype(np.float32)
    return out
