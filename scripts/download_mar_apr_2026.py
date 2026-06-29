"""Download LISS-IV scenes (Mar–Apr 2026) via Bhoonidhi portal and organize by cloud cover.

Uses bhoonidhi-downloader session/search helpers against bhoonidhi.nrsc.gov.in
(the STAC API at bhoonidhi-api.nrsc.gov.in is often unreachable outside India).

Usage:
    python scripts/download_mar_apr_2026.py
    python scripts/download_mar_apr_2026.py --skip-download   # process existing raw only
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import requests
from geopandas import GeoDataFrame
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bhoonidhi_downloader.authenticate import save_session_info  # noqa: E402
from bhoonidhi_downloader.scene_search import search_for_scenes  # noqa: E402
from bhoonidhi_downloader.utils import get_download_url  # noqa: E402

BHOONIDHI_BASE = "https://bhoonidhi.nrsc.gov.in"


def portal_login(user_id: str, password: str) -> dict:
    """Login via Bhoonidhi portal (avoid os.path.join URL bug on Windows)."""
    response = requests.post(
        f"{BHOONIDHI_BASE}/bhoonidhi/LoginServlet",
        data=json.dumps(
            {"userId": user_id, "password": password, "action": "VALIDATE_LOGIN", "oldDB": "false"}
        ),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Bhoonidhi login HTTP {response.status_code}: {response.text[:200]}")
    result = response.json()["Results"][0]
    if not result.get("JWT"):
        raise RuntimeError(f"Bhoonidhi login rejected: {result.get('MSG', result)}")
    return result
from fusion.preprocess.bhoonidhi_extract import (  # noqa: E402
    extract_zip,
    find_product_root,
    product_id_from_folder,
    stack_liss4_bands,
)
from src.config import CATALOG_DIR, RAW_DIR, ensure_data_dirs, get_credentials  # noqa: E402
from src.regions import NER_FULL  # noqa: E402

CLOUDY_THRESHOLD = 0.15  # scenes above this -> lissiv_cloudy, else lissiv_clear


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and organize Mar–Apr 2026 LISS-IV NER data")
    parser.add_argument("--start", default="2026-03-01")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--skip-download", action="store_true", help="Only extract/process existing zips")
    parser.add_argument("--max-scenes", type=int, default=0, help="Limit downloads (0 = all catalogue hits)")
    parser.add_argument("--cloud-threshold", type=float, default=CLOUDY_THRESHOLD)
    return parser.parse_args()


def session_from_credentials(user_id: str, password: str) -> dict:
    session_data = portal_login(user_id, password)
    if not session_data or not session_data.get("JWT"):
        raise RuntimeError("Bhoonidhi login failed — check BHOONIDHI_USER_ID / BHOONIDHI_PASSWORD in .env")
    session = {
        "jwt": session_data["JWT"],
        "userId": session_data["USERID"],
        "user_email": session_data.get("USEREMAIL", ""),
        "user_name": session_data.get("USERNAME", ""),
        "username": user_id,
        "password": password,
        "sid": None,
        "scenes": [],
    }
    save_session_info(session)
    return session


def search_liss4_scenes(session: dict, start: datetime, end: datetime) -> list[dict]:
    min_lon, min_lat, max_lon, max_lat = NER_FULL.bbox
    gdf = GeoDataFrame(geometry=[box(min_lon, min_lat, max_lon, max_lat)], crs="EPSG:4326")
    scenes: list[dict] = []
    for satellite in ("ResourceSat-2", "ResourceSat-2A"):
        batch = search_for_scenes(gdf, satellite, "LISS4", start, end, session) or []
        scenes.extend(batch)
    open_scenes = [s for s in scenes if s.get("PRICED") == "OpenData_DirectDownload"]
    session["scenes"] = open_scenes
    if open_scenes:
        session["sid"] = open_scenes[0].get("srt")
    save_session_info(session)
    return open_scenes


def download_scene_requests(url: str, out_path: Path, jwt: str, user_id: str, password: str) -> Path:
    """Download a scene zip using requests (Windows-friendly; no wget required)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(out_path.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(4):
        try:
            token = portal_login(user_id, password)
            if token and token.get("JWT"):
                url = re.sub(r"token=[^&]+", f"token={token['JWT']}", url)
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0) or 0)
            downloaded_bytes = 0
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        downloaded_bytes += len(chunk)
                        if total and downloaded_bytes % (50 * 1024 * 1024) < len(chunk):
                            pct = downloaded_bytes * 100 // total
                            print(f"    {downloaded_bytes // (1024*1024)} / {total // (1024*1024)} MB ({pct}%)", flush=True)
            partial.replace(out_path)
            return out_path
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(2 ** attempt, 15))
            if partial.exists():
                partial.unlink(missing_ok=True)

    raise RuntimeError(f"Download failed for {out_path.name}: {last_error}")


def dedupe_scenes(scenes: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for scene in scenes:
        scene_id = scene.get("ID", "")
        if scene_id and scene_id not in seen:
            seen.add(scene_id)
            unique.append(scene)
    return unique


def download_one(scene: dict, session: dict, raw_dir: Path) -> Path | None:
    scene_id = scene["ID"]
    zip_path = raw_dir / f"{scene_id}.zip"
    if zip_path.exists() and zip_path.stat().st_size > 1_000_000:
        return zip_path

    user_id = session["username"]
    password = session["password"]
    jwt = portal_login(user_id, password)["JWT"]
    session["jwt"] = jwt
    url = get_download_url(scene_id, session)
    return download_scene_requests(url, zip_path, jwt, user_id, password)


def pipeline_scenes(
    scenes: list[dict],
    session: dict,
    raw_dir: Path,
    fusion_root: Path,
    threshold: float,
) -> tuple[int, int]:
    """Download, extract, stack, and classify one scene at a time (disk-friendly)."""
    n_cloudy = n_clear = 0
    cloudy_dir = fusion_root / "lissiv_cloudy"
    clear_dir = fusion_root / "lissiv_clear"
    cloudy_dir.mkdir(parents=True, exist_ok=True)
    clear_dir.mkdir(parents=True, exist_ok=True)

    for index, scene in enumerate(scenes, start=1):
        scene_id = scene["ID"]
        print(f"[{index}/{len(scenes)}] {scene_id} ({scene.get('DOP', '')})", flush=True)
        try:
            zip_path = download_one(scene, session, raw_dir)
            if zip_path is None:
                continue
            extract_to = raw_dir / zip_path.stem
            if find_product_root(extract_to) is None:
                print("  extracting ...", flush=True)
                extract_zip(zip_path, extract_to)
            zip_path.unlink(missing_ok=True)
            root = find_product_root(extract_to)
            if root is None:
                print("  WARN: no BAND2/3/4 after extract", flush=True)
                continue

            pid = product_id_from_folder(root)
            try:
                fraction = cloud_fraction_from_bands(root)
            except Exception as exc:  # noqa: BLE001
                print(f"  WARN: cloud estimate failed: {exc}", flush=True)
                fraction = 0.5

            role_dir = cloudy_dir if fraction >= threshold else clear_dir
            out_path = role_dir / f"{pid}.tif"
            if not out_path.exists():
                stack_liss4_bands(root, out_path, max_edge=2048)
            print(f"  -> {role_dir.name} (cloud ~{fraction * 100:.0f}%)", flush=True)
            # Remove full-res extract to save disk (~3 GB/scene).
            shutil.rmtree(extract_to, ignore_errors=True)
            if fraction >= threshold:
                n_cloudy += 1
            else:
                n_clear += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}", flush=True)

    return n_cloudy, n_clear


def extract_all(raw_dir: Path) -> list[Path]:
    roots: list[Path] = []
    for zip_path in raw_dir.glob("*.zip"):
        extract_to = raw_dir / zip_path.stem
        if find_product_root(extract_to) is None:
            print(f"  extracting {zip_path.name}", flush=True)
            extract_zip(zip_path, extract_to)
        root = find_product_root(extract_to)
        if root is not None:
            roots.append(root)
            # Free disk: raw zip no longer needed once bands are extracted.
            try:
                zip_path.unlink(missing_ok=True)
            except OSError:
                pass
    for folder in raw_dir.iterdir():
        if folder.is_dir():
            root = find_product_root(folder)
            if root is not None and root not in roots:
                roots.append(root)
    return roots


def cloud_fraction_from_bands(root: Path) -> float:
    """Estimate cloud fraction from BAND2/3/4 using simple brightness threshold."""
    try:
        import numpy as np
        import rasterio
    except ImportError as exc:
        raise ImportError("cloud_fraction_from_bands requires rasterio and numpy") from exc

    arrays = []
    for name in ("BAND2.tif", "BAND3.tif", "BAND4.tif"):
        with rasterio.open(root / name) as src:
            band = src.read(1, out_shape=(512, 512), resampling=rasterio.enums.Resampling.bilinear)
            arrays.append(band.astype("float32"))
    rgb = np.stack(arrays, axis=0)
    brightness = rgb.mean(axis=0)
    lo, hi = np.percentile(brightness, 2), np.percentile(brightness, 98)
    norm = np.clip((brightness - lo) / (hi - lo + 1e-6), 0, 1)
    return float((norm > 0.72).mean())


def process_into_fusion(raw_dir: Path, fusion_root: Path, threshold: float) -> tuple[int, int]:
    roots = extract_all(raw_dir)
    cloudy_dir = fusion_root / "lissiv_cloudy"
    clear_dir = fusion_root / "lissiv_clear"
    cloudy_dir.mkdir(parents=True, exist_ok=True)
    clear_dir.mkdir(parents=True, exist_ok=True)

    n_cloudy = n_clear = 0
    for root in roots:
        scene_id = product_id_from_folder(root)
        try:
            fraction = cloud_fraction_from_bands(root)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN: could not estimate cloud for {scene_id}: {exc}")
            fraction = 0.5

        role_dir = cloudy_dir if fraction >= threshold else clear_dir
        out_path = role_dir / f"{scene_id}.tif"
        if out_path.exists():
            print(f"  skip stacked {out_path.name} (cloud ~{fraction * 100:.0f}%)")
        else:
            stack_liss4_bands(root, out_path, max_edge=2048)
            print(f"  stacked {out_path.name} -> {role_dir.name} (cloud ~{fraction * 100:.0f}%)")

        if fraction >= threshold:
            n_cloudy += 1
        else:
            n_clear += 1

    return n_cloudy, n_clear


def main() -> int:
    args = parse_args()
    ensure_data_dirs()
    raw_dir = RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    fusion_root = PROJECT_ROOT / "data" / "fusion"
    fusion_root.mkdir(parents=True, exist_ok=True)

    credentials = get_credentials()
    if not credentials.is_complete:
        print("ERROR: Set BHOONIDHI_USER_ID and BHOONIDHI_PASSWORD in .env")
        return 1

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    if not args.skip_download:
        print(f"Authenticating as {credentials.user_id} ...")
        session = session_from_credentials(credentials.user_id, credentials.password)
        print(f"Searching LISS-IV over {NER_FULL.name} ({args.start} to {args.end}) ...")
        scenes = search_liss4_scenes(session, start, end)
        scenes = dedupe_scenes(scenes)
        print(f"Found {len(scenes)} open direct-download scene(s).")
        if args.max_scenes > 0:
            scenes = scenes[: args.max_scenes]
            print(f"Limited to first {len(scenes)} scene(s) (--max-scenes).")

        catalog_path = CATALOG_DIR / f"liss4_ner_{args.start[:7]}_{args.end[:7]}.json"
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text(json.dumps(scenes, indent=2), encoding="utf-8")
        print(f"Catalogue saved: {catalog_path}")

        if scenes:
            print(f"Downloading + processing to {raw_dir} ...", flush=True)
            n_cloudy, n_clear = pipeline_scenes(scenes, session, raw_dir, fusion_root, args.cloud_threshold)
            print(f"Done. Cloudy: {n_cloudy}, Clear: {n_clear}")
            print(f"  Raw:    {raw_dir}")
            print(f"  Cloudy: {fusion_root / 'lissiv_cloudy'}")
            print(f"  Clear:  {fusion_root / 'lissiv_clear'}")
            return 0

    print("Extracting zips and stacking into lissiv_cloudy / lissiv_clear ...", flush=True)
    n_cloudy, n_clear = process_into_fusion(raw_dir, fusion_root, args.cloud_threshold)
    print(f"Done. Cloudy: {n_cloudy}, Clear: {n_clear}")
    print(f"  Raw:    {raw_dir}")
    print(f"  Cloudy: {fusion_root / 'lissiv_cloudy'}")
    print(f"  Clear:  {fusion_root / 'lissiv_clear'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
