"""Generate ISRO-style cloudy vs cloud-free comparison images from raw LISS-IV.

Usage:
    python scripts/generate_viz.py
    python scripts/generate_viz.py --raw data/raw/liss4 --pairs 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from fusion.infer import load_generator
from src.config import RAW_DIR
from ui.scene_viz import (
    cloud_fraction,
    discover_scenes,
    reconstruct_crop,
    save_comparison_grid,
    save_side_by_side,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cloud removal visualizations")
    parser.add_argument("--raw", default=str(RAW_DIR))
    parser.add_argument("--checkpoint", default="outputs/checkpoints/fusion_generator.pt")
    parser.add_argument("--out", default="outputs/cloud_free")
    parser.add_argument("--pairs", type=int, default=2, help="Number of scene pairs in poster")
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--bootstrap-steps", type=int, default=30)
    return parser.parse_args()


def ensure_checkpoint(path: Path, steps: int) -> None:
    if path.exists():
        return
    print("No checkpoint found — running bootstrap training...")
    import subprocess

    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "bootstrap_model.py"), "--steps", str(steps)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw)
    ckpt = Path(args.checkpoint)
    out_dir = Path(args.out)

    scenes = discover_scenes(raw_dir)
    if not scenes:
        print(f"ERROR: No LISS-IV scenes in {raw_dir}")
        return 1

    ensure_checkpoint(ckpt, args.bootstrap_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_generator(ckpt, device)

    ranked = sorted(scenes, key=cloud_fraction, reverse=True)
    picks = ranked[: max(1, min(args.pairs, len(ranked)))]
    print(f"Processing {len(picks)} scene(s)...")

    grid_pairs: list[tuple] = []
    for scene in picks:
        print(f"  {scene.name} (cloud ~{cloud_fraction(scene) * 100:.0f}%)")
        cloudy, clear, _, origin = reconstruct_crop(
            scene, ckpt, crop_size=args.crop, model=model, device=device
        )
        stem = scene.name.replace(" ", "_")[:60]
        save_side_by_side(cloudy, clear, out_dir / f"{stem}_pair.png")
        from PIL import Image

        Image.fromarray((cloudy * 255).astype("uint8")).save(out_dir / f"{stem}_cloudy.png")
        Image.fromarray((clear * 255).astype("uint8")).save(out_dir / f"{stem}_clear.png")
        print(f"    crop y={origin[0]} x={origin[1]}")
        grid_pairs.append((cloudy, clear))

    poster = save_comparison_grid(grid_pairs, out_dir / "visualization_of_cloud_removal.png")
    print(f"Done.\n  Poster: {poster}\n  Folder: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
