"""Process Bhoonidhi LISS-IV downloads into 3-band GeoTIFF stacks.

Full-resolution output is ~3.5 GB per scene. Default downsamples to 2048 px
(~5–15 MB/scene). The Results tab reads raw BAND2/3/4 directly — stacking is
only needed for fusion training in data/fusion/.

Usage:
    python scripts/process_bhoonidhi.py --input data/raw/liss4 --role cloudy
    python scripts/process_bhoonidhi.py --input data/raw/liss4 --full-res
    python scripts/process_bhoonidhi.py --input path/to/scene --output out.tif --max-edge 2048
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.preprocess.bhoonidhi_extract import batch_process_raw, find_product_root, stack_liss4_bands  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stack Bhoonidhi BAND2/3/4 into G/R/NIR GeoTIFF")
    parser.add_argument("--input", required=True, help="Raw download folder or single product folder")
    parser.add_argument("--output", default="", help="Single output .tif (optional)")
    parser.add_argument("--role", default="cloudy", choices=["cloudy", "clear"], help="Batch mode subfolder")
    parser.add_argument("--fusion-root", default="data/fusion", help="Batch output root")
    parser.add_argument(
        "--max-edge",
        type=int,
        default=2048,
        help="Max long edge in pixels (default 2048, ~5-15 MB/scene). Use 0 for full resolution.",
    )
    parser.add_argument("--full-res", action="store_true", help="Write full resolution (~3.5 GB/scene)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing stacked files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    max_edge = None if args.full_res or args.max_edge == 0 else args.max_edge

    if args.output:
        root = find_product_root(input_path)
        if root is None:
            print(f"ERROR: No BAND2/3/4 found in {input_path}")
            return 1
        out = stack_liss4_bands(root, Path(args.output), max_edge=max_edge)
        print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return 0

    if max_edge is None:
        print("WARNING: Full-resolution mode needs ~3.5 GB free disk space per scene.")

    written = batch_process_raw(
        input_path,
        Path(args.fusion_root),
        role=args.role,
        max_edge=max_edge,
        overwrite=args.overwrite,
    )
    print(f"Processed {len(written)} scene(s) -> {args.fusion_root}/lissiv_{args.role}/")
    for path in written[:10]:
        size_mb = path.stat().st_size / 1e6 if path.exists() else 0
        print(f"  {path} ({size_mb:.1f} MB)")
    if len(written) > 10:
        print(f"  ... and {len(written) - 10} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
