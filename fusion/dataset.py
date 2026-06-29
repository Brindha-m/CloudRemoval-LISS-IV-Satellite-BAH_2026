"""Datasets for SAR-Optical fusion cloud removal.

PairedFusionDataset reads aligned tiles from disk:

    root/lissiv_cloudy/<id>.png|.tif   (3-band optical with cloud)
    root/lissiv_clear/<id>.png|.tif    (3-band cloud-free target)
    root/sentinel1/<id>.npy|.tif       (2-band SAR VV/VH)
    root/masks/<id>.png                (cloud mask, white = cloud)

SyntheticFusionDataset / generate_fusion_batch let you train and smoke-test
with zero real data: clean scenes, correlated SAR, fractal cloud masks, and
cloudy composites are generated on the fly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _fractal(batch: int, h: int, w: int, octaves: int, device, generator) -> torch.Tensor:
    field = torch.zeros(batch, 1, h, w, device=device)
    amp, total = 1.0, 0.0
    for octave in range(octaves):
        cells = 2 ** (octave + 1)
        grid = torch.rand(batch, 1, cells, cells, device=device, generator=generator)
        field = field + amp * F.interpolate(grid, size=(h, w), mode="bilinear", align_corners=False)
        total += amp
        amp *= 0.55
    field = field / max(total, 1e-6)
    lo = field.amin(dim=(2, 3), keepdim=True)
    hi = field.amax(dim=(2, 3), keepdim=True)
    return (field - lo) / (hi - lo + 1e-6)


def generate_fusion_batch(
    batch: int,
    size: int = 256,
    coverage: float = 0.4,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Generate a synthetic aligned SAR-optical batch with cloud masks."""
    veg = _fractal(batch, size, size, 5, device, generator)
    moisture = _fractal(batch, size, size, 4, device, generator)
    # Forest-like G/R/NIR: strong NIR, moderate red, deep green (natural mosaic colours).
    nir = 0.35 + 0.50 * veg
    red = 0.12 + 0.22 * (1 - veg) + 0.04 * moisture
    green = 0.22 + 0.42 * veg + 0.06 * (1 - moisture)
    clear = torch.cat([green, red, nir], dim=1).clamp(0, 1)

    edges = _fractal(batch, size, size, 6, device, generator)
    speckle = torch.rand(batch, 2, size, size, device=device, generator=generator)
    structure = 0.5 * edges + 0.5 * veg
    vv = (structure + 0.15 * speckle[:, 0:1]).clamp(0, 1)
    vh = (0.7 * structure + 0.2 * speckle[:, 1:2]).clamp(0, 1)
    sar = torch.cat([vv, vh], dim=1)

    cloud_field = _fractal(batch, size, size, 4, device, generator)
    threshold = 1.0 - coverage
    airlight = 0.82 + 0.12 * torch.rand(batch, 3, 1, 1, device=device, generator=generator)
    airlight[:, 0] *= 0.95  # slightly less green in cloud airlight
    airlight[:, 2] *= 1.05
    haze = 0.35 + 0.65 * _fractal(batch, size, size, 3, device, generator)
    mask = torch.sigmoid((cloud_field - threshold) / 0.06).clamp(0, 1)
    cloudy = clear * (1 - mask * haze) + airlight * (mask * haze)
    cloudy = cloudy.clamp(0, 1)

    return {"cloudy": cloudy, "clear": clear, "sar": sar, "mask": mask}


class SyntheticFusionDataset(Dataset):
    def __init__(self, length: int = 512, size: int = 256, coverage: float = 0.4, seed: int = 0) -> None:
        self.length = length
        self.size = size
        self.coverage = coverage
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(self.seed + index)
        batch = generate_fusion_batch(1, self.size, self.coverage, generator=gen)
        return {key: value[0] for key, value in batch.items()}


def _read_image(path: Path, channels: int, size: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path).astype(np.float32)
        if array.ndim == 2:
            array = array[None]
    else:
        from PIL import Image

        mode = "RGB" if channels == 3 else "L"
        image = Image.open(path).convert(mode).resize((size, size), Image.BILINEAR)
        array = np.asarray(image).astype(np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1)) if channels == 3 else array[None]
    return array


class PairedFusionDataset(Dataset):
    def __init__(self, root: str | Path, size: int = 256) -> None:
        self.root = Path(root)
        self.size = size
        self.cloudy_dir = self.root / "lissiv_cloudy"
        self.clear_dir = self.root / "lissiv_clear"
        self.sar_dir = self.root / "sentinel1"
        self.mask_dir = self.root / "masks"
        self.ids = sorted(p.stem for p in self.cloudy_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not self.ids:
            raise FileNotFoundError(f"No tiles found in {self.cloudy_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def _find(self, directory: Path, stem: str, extra: set[str] | None = None) -> Path:
        exts = IMAGE_EXTS | (extra or set())
        for ext in exts:
            candidate = directory / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Missing {stem} in {directory}")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        stem = self.ids[index]
        cloudy = _read_image(self._find(self.cloudy_dir, stem), 3, self.size)
        clear = _read_image(self._find(self.clear_dir, stem), 3, self.size)
        sar = _read_image(self._find(self.sar_dir, stem, {".npy"}), 2, self.size)
        mask = _read_image(self._find(self.mask_dir, stem), 1, self.size)
        mask = (mask > 0.5).astype(np.float32)

        sample = {
            "cloudy": torch.from_numpy(cloudy).float(),
            "clear": torch.from_numpy(clear).float(),
            "sar": torch.from_numpy(sar).float(),
            "mask": torch.from_numpy(mask).float(),
        }
        return _augment(sample)


def _sar_from_optical(optical: np.ndarray) -> np.ndarray:
    luma = optical.mean(axis=0)
    gx = np.abs(np.gradient(luma, axis=1))
    gy = np.abs(np.gradient(luma, axis=0))
    edge = np.clip(gx + gy, 0, 1)
    return np.stack([luma, edge], axis=0).astype(np.float32)


def has_paired_data(root: str | Path) -> bool:
    """True only when cloudy, clear, SAR, and mask tiles share the same scene id."""
    root = Path(root)
    cloudy_dir = root / "lissiv_cloudy"
    clear_dir = root / "lissiv_clear"
    sar_dir = root / "sentinel1"
    mask_dir = root / "masks"
    if not all(d.exists() for d in (cloudy_dir, clear_dir, sar_dir, mask_dir)):
        return False

    def _stems(directory: Path, extra: set[str] | None = None) -> set[str]:
        exts = IMAGE_EXTS | (extra or set())
        return {p.stem for p in directory.iterdir() if p.suffix.lower() in exts}

    cloudy_ids = _stems(cloudy_dir)
    if not cloudy_ids:
        return False
    clear_ids = _stems(clear_dir)
    sar_ids = _stems(sar_dir, {".npy"})
    mask_ids = _stems(mask_dir)
    return bool(cloudy_ids & clear_ids & sar_ids & mask_ids)


class CloudySceneDataset(Dataset):
    """Random patches from cloudy LISS-IV stacks — no clear reference required."""

    def __init__(
        self,
        root: str | Path,
        size: int = 256,
        patches_per_scene: int = 40,
        seed: int = 0,
    ) -> None:
        from .preprocess.cloud_mask import estimate_cloud_probability, feather_probability

        self.root = Path(root)
        self.size = size
        self.patches_per_scene = patches_per_scene
        self.seed = seed
        self._estimate_mask = lambda chw: feather_probability(
            estimate_cloud_probability(chw), sigma=1.0
        )
        self.cloudy_dir = self.root / "lissiv_cloudy"
        self.paths = sorted(p for p in self.cloudy_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not self.paths:
            raise FileNotFoundError(f"No cloudy scenes in {self.cloudy_dir}")

    def __len__(self) -> int:
        return len(self.paths) * self.patches_per_scene

    def _read_patch(self, path: Path, index: int) -> np.ndarray:
        import rasterio
        from rasterio.windows import Window

        rng = np.random.default_rng(self.seed + index)
        with rasterio.open(path) as src:
            height, width = src.height, src.width
            crop = min(self.size, height, width)
            if height > crop:
                y = int(rng.integers(0, height - crop + 1))
            else:
                y = 0
            if width > crop:
                x = int(rng.integers(0, width - crop + 1))
            else:
                x = 0
            data = src.read(
                (1, 2, 3),
                window=Window(x, y, crop, crop),
                out_dtype="float32",
            )
        data = data / 255.0 if data.max() > 1.5 else data
        if data.shape[1] != self.size or data.shape[2] != self.size:
            tensor = torch.from_numpy(data[None]).float()
            data = F.interpolate(tensor, size=(self.size, self.size), mode="bilinear", align_corners=False)[0].numpy()
        return data.astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = self.paths[index // self.patches_per_scene]
        cloudy = self._read_patch(path, index)
        mask = self._estimate_mask(cloudy)
        sar = _sar_from_optical(cloudy)
        clear = cloudy.copy()

        sample = {
            "cloudy": torch.from_numpy(cloudy).float(),
            "clear": torch.from_numpy(clear).float(),
            "sar": torch.from_numpy(sar).float(),
            "mask": torch.from_numpy(mask).float(),
        }
        return _augment(sample)


def _augment(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if torch.rand(1).item() < 0.5:
        sample = {k: torch.flip(v, dims=[-1]) for k, v in sample.items()}
    if torch.rand(1).item() < 0.5:
        sample = {k: torch.flip(v, dims=[-2]) for k, v in sample.items()}
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        sample = {key: torch.rot90(value, k, dims=[-2, -1]) for key, value in sample.items()}
    return sample
