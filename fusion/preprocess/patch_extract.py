"""Extract overlapping patches from large rasters/arrays and stitch them back.

Tiling with overlap + feathered blending avoids seams at patch borders during
inference on full scenes.
"""

from __future__ import annotations

import numpy as np


def iter_patch_origins(height: int, width: int, size: int, overlap: float) -> list[tuple[int, int]]:
    step = max(1, int(size * (1.0 - overlap)))
    ys = list(range(0, max(1, height - size + 1), step))
    xs = list(range(0, max(1, width - size + 1), step))
    if ys[-1] != height - size:
        ys.append(max(0, height - size))
    if xs[-1] != width - size:
        xs.append(max(0, width - size))
    return [(y, x) for y in ys for x in xs]


def _feather_window(size: int) -> np.ndarray:
    ramp = np.minimum(np.arange(size), np.arange(size)[::-1]).astype(np.float32) + 1.0
    window = np.outer(ramp, ramp)
    return window / window.max()


def extract_patches(array: np.ndarray, size: int = 256, overlap: float = 0.5) -> list[dict]:
    """array: [C, H, W]. Returns list of {patch, y, x}."""
    _, height, width = array.shape
    patches = []
    for y, x in iter_patch_origins(height, width, size, overlap):
        patches.append({"patch": array[:, y:y + size, x:x + size], "y": y, "x": x})
    return patches


def stitch_patches(patches: list[dict], channels: int, height: int, width: int, size: int) -> np.ndarray:
    accum = np.zeros((channels, height, width), dtype=np.float32)
    weight = np.zeros((1, height, width), dtype=np.float32)
    window = _feather_window(size)[None]
    for item in patches:
        y, x = item["y"], item["x"]
        patch = item["patch"]
        h = min(size, height - y)
        w = min(size, width - x)
        accum[:, y:y + h, x:x + w] += patch[:, :h, :w] * window[:, :h, :w]
        weight[:, y:y + h, x:x + w] += window[:, :h, :w]
    return accum / np.clip(weight, 1e-6, None)
