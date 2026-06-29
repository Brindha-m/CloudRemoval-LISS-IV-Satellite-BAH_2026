"""Co-register an auxiliary raster (e.g. Sentinel-1 SAR) onto the LISS-IV grid.

Uses rasterio's reproject to resample the source raster to the reference
raster's CRS, transform, and shape. Requires rasterio (and GDAL under the
hood); import is guarded so the rest of the package works without it.
"""

from __future__ import annotations

from pathlib import Path


def coregister(reference_path: str | Path, source_path: str | Path, output_path: str | Path) -> Path:
    try:
        import numpy as np
        import rasterio
        from rasterio.warp import Resampling, reproject
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("coregister requires rasterio. Install with `pip install rasterio`.") from exc

    output_path = Path(output_path)
    with rasterio.open(reference_path) as ref:
        ref_profile = ref.profile
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_height, ref_width = ref.height, ref.width

    with rasterio.open(source_path) as src:
        bands = src.count
        destination = np.zeros((bands, ref_height, ref_width), dtype="float32")
        for band in range(1, bands + 1):
            reproject(
                source=rasterio.band(src, band),
                destination=destination[band - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )

    profile = ref_profile.copy()
    profile.update(count=bands, dtype="float32")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(destination)
    return output_path
