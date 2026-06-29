"""Inference for SAR-Optical fusion cloud removal.

Two entry points:
  - reconstruct_array: pure NumPy/torch, tiles a full scene and stitches.
  - reconstruct_geotiff: reads/writes GeoTIFF, preserving CRS and transform.

CLI:
    python -m fusion.infer --checkpoint outputs/checkpoints/fusion_generator.pt \
        --optical scene_optical.tif --sar scene_sar.tif --out scene_clear.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .generator import DCMFUNet
from .preprocess.cloud_mask import estimate_cloud_mask, estimate_cloud_probability, feather_probability
from .preprocess.patch_extract import extract_patches, iter_patch_origins, stitch_patches


def load_generator(checkpoint: str | Path, device: torch.device | str = "cpu") -> DCMFUNet:
    model = DCMFUNet()
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.to(device).eval()
    return model


@torch.no_grad()
def reconstruct_array(
    model: DCMFUNet,
    optical: np.ndarray,
    sar: np.ndarray,
    mask: np.ndarray | None = None,
    cloud_probability: np.ndarray | None = None,
    size: int = 256,
    overlap: float = 0.5,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """optical: [3,H,W], sar: [2,H,W]. Returns [3,H,W] cloud-free cube."""
    if mask is None and cloud_probability is None:
        cloud_probability = estimate_cloud_probability(optical)
    if cloud_probability is None:
        cloud_probability = mask.astype(np.float32)
    if cloud_probability.ndim == 2:
        cloud_probability = cloud_probability[None]

    binary = (cloud_probability >= 0.45).astype(np.float32)
    channels, height, width = optical.shape
    stacked = np.concatenate([optical, sar, binary], axis=0)
    tiles = extract_patches(stacked, size=size, overlap=overlap)

    outputs = []
    for tile in tiles:
        patch = tile["patch"]
        opt = torch.from_numpy(patch[:3]).unsqueeze(0).float().to(device)
        sar_t = torch.from_numpy(patch[3:5]).unsqueeze(0).float().to(device)
        msk = torch.from_numpy(patch[5:6]).unsqueeze(0).float().to(device)
        pred = model.generate(opt, sar_t, msk).squeeze(0).cpu().numpy()
        outputs.append({"patch": pred, "y": tile["y"], "x": tile["x"]})

    actual = outputs[0]["patch"].shape[-1]
    generated = stitch_patches(outputs, channels, height, width, actual)
    soft = feather_probability(cloud_probability, sigma=1.2)
    return optical * (1.0 - soft) + generated * soft


def reconstruct_geotiff(
    checkpoint: str | Path,
    optical_path: str | Path,
    sar_path: str | Path,
    output_path: str | Path,
    mask_path: str | Path | None = None,
    size: int = 256,
    overlap: float = 0.5,
) -> Path:
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover
        raise ImportError("reconstruct_geotiff requires rasterio. `pip install rasterio`.") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_generator(checkpoint, device)

    with rasterio.open(optical_path) as src:
        optical = src.read(out_dtype="float32")[:3]
        profile = src.profile
    optical = _normalize(optical)

    with rasterio.open(sar_path) as src:
        sar = _normalize(src.read(out_dtype="float32")[:2])

    mask = None
    if mask_path is not None:
        with rasterio.open(mask_path) as src:
            mask = (src.read(out_dtype="float32")[:1] > 0.5).astype("float32")

    result = reconstruct_array(model, optical, sar, mask, size=size, overlap=overlap, device=device)

    profile.update(count=3, dtype="float32")
    output_path = Path(output_path)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(result.astype("float32"))
    return output_path


def _normalize(array: np.ndarray) -> np.ndarray:
    array = array.astype("float32")
    lo = array.min(axis=(1, 2), keepdims=True)
    hi = array.max(axis=(1, 2), keepdims=True)
    return (array - lo) / (hi - lo + 1e-6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cloud-free reconstruction (GeoTIFF in/out)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--optical", required=True)
    parser.add_argument("--sar", required=True)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--overlap", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = reconstruct_geotiff(
        args.checkpoint, args.optical, args.sar, args.out, args.mask, args.size, args.overlap
    )
    print(f"Wrote cloud-free GeoTIFF: {out}")


if __name__ == "__main__":
    main()
