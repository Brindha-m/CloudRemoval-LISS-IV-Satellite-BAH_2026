"""Process extracted Bhoonidhi LISS-IV product folders.

Typical Bhoonidhi LISS-IV MX70 extract layout:

    R2F24JAN2026076626010500057SSANSTUCOOGTDB/
        BAND2.tif          # Green
        BAND3.tif          # Red
        BAND4.tif          # NIR
        BAND_META.txt
        ACC_REP.txt
        *.jpg              # quicklook
        *.meta / *.META

Full-resolution stacks are ~3.5 GB per scene. By default we downsample to
max_edge=2048 and store compressed uint8 (~5–15 MB/scene) to avoid disk errors.
Inference can read BAND2/3/4 directly from data/raw/liss4 without stacking.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

from rasterio.transform import Affine


BAND_MAP = {
    "BAND2.tif": "green",
    "BAND3.tif": "red",
    "BAND4.tif": "nir",
}


def find_product_root(path: Path) -> Path | None:
    """Return folder containing BAND2/3/4, searching one level down if needed."""
    path = Path(path)
    if not path.exists():
        return None
    if (path / "BAND2.tif").exists():
        return path
    for child in path.iterdir():
        if child.is_dir() and (child / "BAND2.tif").exists():
            return child
    return None


def stack_liss4_bands(
    product_dir: Path,
    output_path: Path,
    max_edge: int | None = 2048,
    compress: str = "deflate",
) -> Path:
    """Stack Green/Red/NIR into one GeoTIFF [3, H, W]. Requires rasterio."""
    try:
        import numpy as np
        import rasterio
    except ImportError as exc:
        raise ImportError("stack_liss4_bands requires rasterio and numpy.") from exc

    product_dir = find_product_root(product_dir)
    if product_dir is None:
        raise FileNotFoundError(f"No BAND2.tif found under {product_dir}")

    band_paths = [product_dir / name for name in ("BAND2.tif", "BAND3.tif", "BAND4.tif")]
    for bp in band_paths:
        if not bp.exists():
            raise FileNotFoundError(f"Missing {bp}")

    with rasterio.open(band_paths[0]) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width

    out_h, out_w = height, width
    if max_edge and max(height, width) > max_edge:
        scale = max_edge / max(height, width)
        out_h = max(1, int(height * scale))
        out_w = max(1, int(width * scale))

    arrays = []
    for bp in band_paths:
        with rasterio.open(bp) as src:
            if (out_h, out_w) == (src.height, src.width):
                band = src.read(1).astype("float32")
            else:
                band = src.read(
                    1,
                    out_shape=(out_h, out_w),
                    resampling=rasterio.enums.Resampling.bilinear,
                ).astype("float32")
            arrays.append(band)

    stacked = np.stack(arrays, axis=0)
    for i in range(stacked.shape[0]):
        lo, hi = np.percentile(stacked[i], 2), np.percentile(stacked[i], 98)
        stacked[i] = np.clip((stacked[i] - lo) / (hi - lo + 1e-6), 0, 1)
    stacked_u8 = (stacked * 255).astype("uint8")

    scale_x = width / out_w
    scale_y = height / out_h
    profile.update(
        count=3,
        dtype="uint8",
        height=out_h,
        width=out_w,
        transform=profile["transform"] * Affine.scale(scale_x, scale_y),
        compress=compress,
        predictor=2,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    try:
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(stacked_u8)
            dst.set_band_description(1, "Green")
            dst.set_band_description(2, "Red")
            dst.set_band_description(3, "NIR")
            if max_edge and (out_h, out_w) != (height, width):
                dst.update_tags(
                    source_resolution=f"{height}x{width}",
                    downsample_max_edge=str(max_edge),
                )
    except Exception:
        output_path.unlink(missing_ok=True)
        raise

    return output_path


def extract_zip(zip_path: Path, output_dir: Path) -> Path:
    """Unzip a Bhoonidhi download and return the product folder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    root = find_product_root(output_dir)
    if root is None:
        raise FileNotFoundError(f"Could not find LISS-IV bands in {output_dir}")
    return root


def product_id_from_folder(folder: Path) -> str:
    """Use folder name as scene id, sanitized for filenames."""
    name = folder.name
    return re.sub(r"[^\w\-]", "_", name)[:80]


def batch_process_raw(
    raw_dir: Path,
    output_dir: Path,
    role: str = "cloudy",
    max_edge: int | None = 2048,
    compress: str = "deflate",
    overwrite: bool = False,
) -> list[Path]:
    """Process all extracted products or zips under raw_dir."""
    raw_dir = Path(raw_dir)
    out_role = Path(output_dir) / f"lissiv_{role}"
    out_role.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def process_root(root: Path) -> None:
        scene_id = product_id_from_folder(root)
        out_path = out_role / f"{scene_id}.tif"
        if out_path.exists() and not overwrite:
            written.append(out_path)
            return
        stack_liss4_bands(root, out_path, max_edge=max_edge, compress=compress)
        written.append(out_path)

    for zip_path in raw_dir.glob("*.zip"):
        extract_to = raw_dir / zip_path.stem
        if not find_product_root(extract_to):
            extract_zip(zip_path, extract_to)
        root = find_product_root(extract_to)
        if root is not None:
            process_root(root)

    for folder in raw_dir.iterdir():
        if not folder.is_dir():
            continue
        root = find_product_root(folder)
        if root is not None:
            process_root(root)

    return written
