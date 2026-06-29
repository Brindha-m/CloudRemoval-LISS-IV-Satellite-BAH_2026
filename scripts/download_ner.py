"""Batch-download LISS-IV products over NER India from the Bhoonidhi API.

Defaults to the full North Eastern Region for March 2026.

Usage:
    python scripts/download_ner.py
    python scripts/download_ner.py --start 2026-03-01 --end 2026-03-31 --region "Assam"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bhoonidhi import (  # noqa: E402
    LISS4_COLLECTIONS,
    BhoonidhiClient,
    BhoonidhiError,
    build_datetime_range,
    summarize_feature,
)
from src.config import CATALOG_DIR, RAW_DIR, ensure_data_dirs, get_credentials  # noqa: E402
from src.regions import NER_FULL, get_region  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LISS-IV NER India scenes from Bhoonidhi")
    parser.add_argument("--start", default="2026-03-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-03-31", help="End date YYYY-MM-DD")
    parser.add_argument("--region", default=NER_FULL.name, help="NER region or state name")
    parser.add_argument("--use-polygon", action="store_true", help="Use polygon intersect instead of bbox")
    parser.add_argument("--max-items", type=int, default=2000, help="Maximum catalogue items to fetch")
    parser.add_argument("--search-only", action="store_true", help="Only search and save catalogue, skip download")
    parser.add_argument("--include-offline", action="store_true", help="Include products not available online")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_data_dirs()

    credentials = get_credentials()
    if not credentials.is_complete:
        print("ERROR: Missing credentials. Set BHOONIDHI_USER_ID and BHOONIDHI_PASSWORD in .env")
        return 1

    region = get_region(args.region)
    period = f"{args.start[:7]}"  # YYYY-MM
    safe_region = region.name.replace(" ", "_")
    raw_dir = RAW_DIR / period / safe_region
    raw_dir.mkdir(parents=True, exist_ok=True)

    client = BhoonidhiClient()
    print(f"Authenticating as {credentials.user_id} ...")
    try:
        token = client.authenticate(credentials.user_id, credentials.password)
    except BhoonidhiError as exc:
        print(f"Authentication failed: {exc}")
        return 1

    datetime_range = build_datetime_range(args.start, args.end)
    print(f"Searching LISS-IV over {region.name} for {args.start} to {args.end} ...")
    features = client.search_all(
        token=token,
        collections=LISS4_COLLECTIONS,
        datetime_range=datetime_range,
        bbox=None if args.use_polygon else region.bbox,
        intersects=region.geojson if args.use_polygon else None,
        max_items=args.max_items,
        online_only=not args.include_offline,
    )
    print(f"Found {len(features)} item(s).")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"liss4_{safe_region}_{period}_{stamp}"
    _save_catalog(features, CATALOG_DIR, base_name)

    if args.search_only or not features:
        print(f"Catalogue saved to {CATALOG_DIR}. Skipping downloads.")
        return 0

    print(f"Downloading {len(features)} product(s) to {raw_dir} ...")

    def on_progress(index: int, total: int, item_id: str) -> None:
        print(f"  [{index}/{total}] {item_id}")

    results = client.batch_download(token, features, raw_dir, progress_callback=on_progress)
    _save_log(results, CATALOG_DIR / f"{base_name}_download_log.csv")

    downloaded = sum(1 for r in results if r["status"] == "downloaded")
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"Done. Downloaded: {downloaded}, Failed: {failed}. Files in {raw_dir}")
    return 0


def _save_catalog(features: list[dict], catalog_dir: Path, base_name: str) -> None:
    catalog_dir.mkdir(parents=True, exist_ok=True)
    json_path = catalog_dir / f"{base_name}.json"
    json_path.write_text(json.dumps(features, indent=2), encoding="utf-8")

    summaries = [summarize_feature(feature) for feature in features]
    csv_path = catalog_dir / f"{base_name}.csv"
    _write_csv(csv_path, summaries)
    print(f"Saved catalogue: {json_path.name}, {csv_path.name}")


def _save_log(results: list[dict], path: Path) -> None:
    _write_csv(path, results)
    print(f"Saved download log: {path.name}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
