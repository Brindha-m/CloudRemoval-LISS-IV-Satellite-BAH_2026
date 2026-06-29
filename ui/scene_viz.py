"""Load LISS-IV scenes and build cloudy vs cloud-free comparison panels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fusion.preprocess.bhoonidhi_extract import find_product_root
from fusion.preprocess.cloud_mask import (
    apply_probability_threshold,
    dilate,
    estimate_cloud_mask,
    estimate_cloud_probability,
    feather_probability,
)

MODEL_SIZE = 256


def discover_scenes(raw_dir: Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    scenes: list[Path] = []
    if not raw_dir.exists():
        return scenes
    for folder in sorted(raw_dir.iterdir()):
        if folder.is_dir() and find_product_root(folder) is not None:
            scenes.append(find_product_root(folder) or folder)
    return scenes


def _normalize_chw(array: np.ndarray) -> np.ndarray:
    lo = array.min(axis=(1, 2), keepdims=True)
    hi = array.max(axis=(1, 2), keepdims=True)
    return (array.astype("float32") - lo) / (hi - lo + 1e-6)


def _resize_chw(chw: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(chw[None]).float()
    out = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return out[0].numpy()


def resize_chw(chw: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(chw[None]).float()
    out = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return out[0].numpy()


def resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(mask[None]).float()
    out = F.interpolate(tensor, size=(height, width), mode="nearest")
    return (out[0].numpy() > 0.5).astype(np.float32)


def rgb_to_liss4_chw(rgb: np.ndarray) -> np.ndarray:
    """Map RGB quicklook to pseudo Green / Red / NIR band order."""
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    green = g
    red = r
    nir = np.clip(0.5 * r + 0.45 * g + 0.05 * b, 0, 1)
    return np.stack([green, red, nir], axis=0).astype(np.float32)


def load_upload_rgb(
    image,
    max_edge: int = 1024,
) -> np.ndarray:
    """Load uploaded PNG/JPEG/TIF as RGB float32 [H, W, 3] in 0–1 (natural colors, no per-band stretch)."""
    from PIL import Image

    image = image.convert("RGB")
    rgb = np.asarray(image).astype(np.float32) / 255.0
    height, width = rgb.shape[:2]
    long_edge = max(height, width)
    if long_edge > max_edge:
        scale = max_edge / long_edge
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        rgb = np.asarray(image).astype(np.float32) / 255.0
    return crop_to_content(rgb)


def load_upload_chw(
    image,
    max_edge: int = 1024,
) -> np.ndarray:
    """G/R/NIR cube for model paths (no per-band normalize — keeps cloud detection stable)."""
    rgb = load_upload_rgb(image, max_edge=max_edge)
    return rgb_to_liss4_chw(rgb)


def crop_to_content(rgb_hwc: np.ndarray, threshold: float = 0.04) -> np.ndarray:
    """Drop black letterbox borders from screenshots before processing."""
    valid = rgb_hwc.mean(axis=-1) > threshold
    if not np.any(valid):
        return rgb_hwc
    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]
    return rgb_hwc[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def erode_mask(mask_hw: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask_hw.astype(np.float32)
    for _ in range(iterations):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        out = (
            padded[:-2, 1:-1]
            * padded[2:, 1:-1]
            * padded[1:-1, :-2]
            * padded[1:-1, 2:]
            * out
        )
    return (out > 0.5).astype(np.float32)


def build_cloud_mask(
    optical_chw: np.ndarray,
    brightness_threshold: float = 0.50,
    whiteness_threshold: float = 0.82,
    probability_threshold: float = 0.45,
    dilate_iters: int = 0,
    erode_iters: int = 0,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    prob = estimate_cloud_probability(optical_chw, brightness_threshold, whiteness_threshold)
    if valid is not None:
        prob = prob * valid.astype(np.float32)[None]
    mask = (prob >= probability_threshold).astype(np.float32)
    if erode_iters > 0:
        mask = erode_mask(mask[0], erode_iters)[None]
    if dilate_iters > 0:
        mask = dilate(mask, iterations=dilate_iters)
    return (mask > 0.5).astype(np.float32)[0]


def build_cloud_probability_rgb(
    rgb_hwc: np.ndarray,
    brightness_threshold: float = 0.50,
    whiteness_threshold: float = 0.82,
    dilate_iters: int = 0,
) -> np.ndarray:
    """Soft cloud probability map [H, W] in 0–1 (Nimbo-style)."""
    valid = rgb_hwc.mean(axis=-1) > 0.04
    chw = rgb_to_liss4_chw(rgb_hwc)
    prob = estimate_cloud_probability(chw, brightness_threshold, whiteness_threshold)[0]
    prob = prob * valid.astype(np.float32)
    if dilate_iters > 0:
        binary = dilate((prob >= 0.45).astype(np.float32)[None], dilate_iters)[0]
        prob = np.maximum(prob, binary * 0.85)
    return feather_probability(prob, sigma=1.2)


def build_cloud_mask_rgb(
    rgb_hwc: np.ndarray,
    brightness_threshold: float = 0.50,
    whiteness_threshold: float = 0.82,
    probability_threshold: float = 0.45,
    dilate_iters: int = 0,
    erode_iters: int = 0,
) -> np.ndarray:
    valid = rgb_hwc.mean(axis=-1) > 0.04
    chw = rgb_to_liss4_chw(rgb_hwc)
    return build_cloud_mask(
        chw,
        brightness_threshold,
        whiteness_threshold,
        probability_threshold,
        dilate_iters,
        erode_iters,
        valid=valid,
    )


def _denormalize_chw(norm_chw: np.ndarray, reference_chw: np.ndarray) -> np.ndarray:
    out = np.zeros_like(norm_chw, dtype=np.float32)
    for band in range(3):
        lo = float(reference_chw[band].min())
        hi = float(reference_chw[band].max())
        out[band] = norm_chw[band] * (hi - lo + 1e-6) + lo
    return np.clip(out, 0, 1)


def chw_to_rgb_hwc(chw: np.ndarray) -> np.ndarray:
    """Natural-colour RGB from G/R/NIR (red, green, NIR-as-red-edge)."""
    red = chw[1]
    green = chw[0]
    blue = np.clip(0.65 * chw[2] + 0.35 * chw[0], 0, 1)
    return np.stack([red, green, blue], axis=-1)


def _edge_preserve_smooth(rgb_hwc: np.ndarray, cloud_prob: np.ndarray, radius: int = 2) -> np.ndarray:
    """Light bilateral-style smooth inside cloud fill to suppress halos without blurring forests."""
    if cloud_prob.max() < 0.05:
        return rgb_hwc
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return rgb_hwc

    weight = np.clip(cloud_prob, 0, 1)[..., None]
    blurred = gaussian_filter(rgb_hwc, sigma=(radius, radius, 0))
    detail = rgb_hwc - blurred
    # Keep texture in clear areas; dampen only high-probability cloud fill.
    softened = blurred + detail * (1.0 - 0.65 * weight)
    return np.clip(softened, 0, 1)


def harmonize_colors(rgb_hwc: np.ndarray, clear_weight: np.ndarray) -> np.ndarray:
    """Match cloud-filled colours to clear-area statistics (natural mosaic look)."""
    out = rgb_hwc.copy()
    keep = clear_weight > 0.55
    if keep.sum() < 64:
        return out
    for channel in range(3):
        ref = rgb_hwc[..., channel][keep]
        ref_mean = float(ref.mean())
        ref_std = float(ref.std()) + 1e-6
        fill = ~keep
        if not np.any(fill):
            continue
        cur = out[..., channel][fill]
        cur_std = float(cur.std()) + 1e-6
        out[..., channel][fill] = np.clip((cur - cur.mean()) * (ref_std / cur_std) + ref_mean, 0, 1)
    return out


def rgb_pair_stretch(cloudy_hwc: np.ndarray, clear_hwc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match contrast on both panels without changing hue."""
    lo = np.percentile(cloudy_hwc, 2)
    hi = np.percentile(cloudy_hwc, 98)

    def stretch(img: np.ndarray) -> np.ndarray:
        return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)

    return stretch(cloudy_hwc), stretch(clear_hwc.copy())


def _read_bands(
    product_dir: Path,
    out_shape: tuple[int, int] | None = None,
    window: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Read BAND2/3/4. Optional decimated out_shape or pixel window (row_off, col_off, h, w)."""
    import rasterio
    from rasterio.windows import Window

    root = find_product_root(product_dir)
    if root is None:
        raise FileNotFoundError(f"No BAND2.tif under {product_dir}")

    bands = []
    for name in ("BAND2.tif", "BAND3.tif", "BAND4.tif"):
        with rasterio.open(root / name) as src:
            if window is not None:
                row_off, col_off, height, width = window
                data = src.read(
                    1,
                    window=Window(col_off, row_off, width, height),
                    out_dtype="float32",
                )
            elif out_shape is not None:
                data = src.read(1, out_shape=out_shape, resampling=rasterio.enums.Resampling.bilinear)
                data = data.astype("float32")
            else:
                data = src.read(1).astype("float32")
            bands.append(data)
    return np.stack(bands, axis=0)


def load_optical_chw(
    product_dir: Path,
    max_edge: int = 1024,
    window: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Load optical cube. Full scenes are ~16k px — use max_edge preview or a window."""
    if window is not None:
        raw = _read_bands(product_dir, window=window)
        if raw.max() > 1.5:
            raw = raw / 255.0
        return _normalize_chw(raw)

    import rasterio

    root = find_product_root(product_dir)
    with rasterio.open(root / "BAND2.tif") as src:
        height, width = src.height, src.width
    long_edge = max(height, width)
    if long_edge <= max_edge:
        return _normalize_chw(_read_bands(product_dir))
    scale = max_edge / long_edge
    out_shape = (max(1, int(height * scale)), max(1, int(width * scale)))
    return _normalize_chw(_read_bands(product_dir, out_shape=out_shape))


def sar_from_optical(optical: np.ndarray) -> np.ndarray:
    luma = optical.mean(axis=0)
    gx = np.abs(np.gradient(luma, axis=1))
    gy = np.abs(np.gradient(luma, axis=0))
    edge = np.clip(gx + gy, 0, 1)
    return np.stack([luma, edge], axis=0).astype("float32")


def chw_to_rgb(chw: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    green, red, nir = chw[0], chw[1], chw[2]
    rgb = np.stack([red, green, nir], axis=-1)
    lo = np.percentile(rgb, lo_pct)
    hi = np.percentile(rgb, hi_pct)
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)


def chw_pair_to_rgb(cloudy_chw: np.ndarray, clear_chw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same colour stretch on both panels so comparison is fair."""
    ref = np.stack([cloudy_chw[1], cloudy_chw[0], cloudy_chw[2]], axis=-1)
    lo = np.percentile(ref, 2)
    hi = np.percentile(ref, 98)

    def stretch(chw: np.ndarray) -> np.ndarray:
        rgb = np.stack([chw[1], chw[0], chw[2]], axis=-1)
        return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)

    return stretch(cloudy_chw), stretch(clear_chw)


def best_cloud_crop(mask: np.ndarray, crop_size: int = 512, stride: int = 64) -> tuple[int, int]:
    _, height, width = mask.shape
    crop_size = min(crop_size, height, width)
    if crop_size <= 0:
        return 0, 0

    best_score = -1.0
    best_y, best_x = 0, 0
    for y in range(0, max(height - crop_size + 1, 1), stride):
        for x in range(0, max(width - crop_size + 1, 1), stride):
            patch = mask[0, y : y + crop_size, x : x + crop_size]
            score = float(patch.mean())
            if 0.08 < score < 0.75 and score > best_score:
                best_score = score
                best_y, best_x = y, x
    if best_score < 0:
        for y in range(0, max(height - crop_size + 1, 1), stride):
            for x in range(0, max(width - crop_size + 1, 1), stride):
                score = float(mask[0, y : y + crop_size, x : x + crop_size].mean())
                if score > best_score:
                    best_score = score
                    best_y, best_x = y, x
    return best_y, best_x


def cloud_fraction(product_dir: Path) -> float:
    preview = load_optical_chw(product_dir, max_edge=512)
    return float(estimate_cloud_mask(preview)[0].mean())


def _full_scene_shape(product_dir: Path) -> tuple[int, int]:
    import rasterio

    root = find_product_root(product_dir)
    with rasterio.open(root / "BAND2.tif") as src:
        return src.height, src.width


def _run_generator(
    model,
    cloudy: np.ndarray,
    sar: np.ndarray,
    cloud_prob: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    from fusion.infer import reconstruct_array

    prob = cloud_prob[None] if cloud_prob.ndim == 2 else cloud_prob
    return reconstruct_array(
        model,
        cloudy,
        sar,
        cloud_probability=prob,
        size=MODEL_SIZE,
        overlap=0.5,
        device=device,
    )


def reconstruct_upload(
    rgb_hwc: np.ndarray,
    cloud_prob: np.ndarray,
    checkpoint: Path,
    model=None,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nimbo-style cloud removal: probabilistic mask + DCMF-UNet + colour harmonization."""
    from fusion.infer import load_generator

    prob = cloud_prob[0] if cloud_prob.ndim == 3 else cloud_prob.astype(np.float32)
    prob = np.clip(prob, 0, 1)
    chw = rgb_to_liss4_chw(rgb_hwc)
    chw_norm = _normalize_chw(chw)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint.exists():
        if model is None:
            model = load_generator(checkpoint, device)
        sar = sar_from_optical(chw_norm)
        clear_norm = _run_generator(model, chw_norm, sar, prob, device)
        clear_chw = _denormalize_chw(clear_norm, chw)
        clear_rgb = chw_to_rgb_hwc(clear_chw)
    else:
        # No trained weights: soft blend keeps clear pixels, dampens cloud brightness.
        damp = rgb_hwc * (1.0 - prob[..., None]) + rgb_hwc * (1.0 - 0.55 * prob[..., None])
        clear_rgb = np.clip(damp, 0, 1)

    clear_rgb = harmonize_colors(clear_rgb, 1.0 - prob)
    clear_rgb = _edge_preserve_smooth(clear_rgb, prob)
    cloudy_rgb, clear_rgb = rgb_pair_stretch(rgb_hwc, clear_rgb)
    mask_rgb = np.stack([prob] * 3, axis=-1)
    return cloudy_rgb, clear_rgb, mask_rgb


def reconstruct_crop(
    product_dir: Path,
    checkpoint: Path,
    crop_size: int = 512,
    model=None,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Return cloudy RGB, clear RGB, mask RGB, and full-res crop origin."""
    from fusion.infer import load_generator

    preview = load_optical_chw(product_dir, max_edge=1024)
    preview_mask = estimate_cloud_mask(preview)
    py, px = best_cloud_crop(preview_mask, crop_size=min(crop_size, preview.shape[1], preview.shape[2]))

    full_h, full_w = _full_scene_shape(product_dir)
    scale_y = full_h / preview.shape[1]
    scale_x = full_w / preview.shape[2]
    y = int(py * scale_y)
    x = int(px * scale_x)
    size = min(crop_size, full_h - y, full_w - x)

    cloudy = load_optical_chw(product_dir, window=(y, x, size, size))
    sar = sar_from_optical(cloudy)
    msk = estimate_cloud_mask(cloudy)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = load_generator(checkpoint, device)

    clear = _run_generator(model, cloudy, sar, msk[0], device)
    cloudy_rgb, clear_rgb = chw_pair_to_rgb(cloudy, clear)
    mask_rgb = np.stack([msk[0]] * 3, axis=-1)
    return cloudy_rgb, clear_rgb, mask_rgb, (y, x)


def save_side_by_side(cloudy_rgb: np.ndarray, clear_rgb: np.ndarray, path: Path) -> Path:
    from PIL import Image

    cloudy_u8 = (cloudy_rgb * 255).astype(np.uint8)
    clear_u8 = (clear_rgb * 255).astype(np.uint8)
    h = max(cloudy_u8.shape[0], clear_u8.shape[0])
    w = cloudy_u8.shape[1] + clear_u8.shape[1]
    canvas = Image.new("RGB", (w, h))
    canvas.paste(Image.fromarray(cloudy_u8), (0, 0))
    canvas.paste(Image.fromarray(clear_u8), (cloudy_u8.shape[1], 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def save_comparison_grid(pairs: list[tuple[np.ndarray, np.ndarray]], path: Path, title_gap: int = 8) -> Path:
    """Stack multiple cloudy|clear pairs vertically (ISRO poster layout)."""
    from PIL import Image, ImageDraw

    row_images: list[Image.Image] = []
    for cloudy, clear in pairs:
        cloudy_u8 = (cloudy * 255).astype(np.uint8)
        clear_u8 = (clear * 255).astype(np.uint8)
        w = cloudy_u8.shape[1] + clear_u8.shape[1]
        h = max(cloudy_u8.shape[0], clear_u8.shape[0])
        row = Image.new("RGB", (w, h))
        row.paste(Image.fromarray(cloudy_u8), (0, 0))
        row.paste(Image.fromarray(clear_u8), (cloudy_u8.shape[1], 0))
        row_images.append(row)

    row_w = max(r.width for r in row_images)
    row_h = sum(r.height for r in row_images) + title_gap * (len(row_images) - 1)
    canvas = Image.new("RGB", (row_w, row_h + 40), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((max(10, row_w // 2 - 120), 8), "Visualization of Cloud removal", fill=(0, 0, 0))
    y_off = 40
    for row in row_images:
        canvas.paste(row, (0, y_off))
        y_off += row.height + title_gap
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path
