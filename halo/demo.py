"""Zero-data smoke test for HALO.

Runs the full forge -> pipeline -> loss path on synthetic tensors and prints
shapes and metrics. Optionally trains a few steps to confirm the loss drops.

    python -m halo.demo
    python -m halo.demo --train-steps 40
"""

from __future__ import annotations

import argparse

import torch

from .cloudforge import ForgeConfig, curriculum_config, forge_batch
from .losses import final_loss
from .pipeline import HaloPipeline
from .train import train_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HALO zero-data smoke test")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HaloPipeline().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"HALO parameters: {params/1e6:.2f}M | device: {device}")

    batch = forge_batch(args.batch_size, args.size, args.size, ForgeConfig(), device=device)
    print("Forged tensors:")
    for key, value in batch.items():
        print(f"  {key:12s} {tuple(value.shape)}")

    model.eval()
    with torch.no_grad():
        result = model(batch["cloudy"], batch["sar"])
    losses = final_loss(result["output"], batch["clean"], batch["cloud_prob"], result["hole_soft"])

    print("Forward pass OK. Output shape:", tuple(result["output"].shape))
    print("Initial losses:")
    for key, value in losses.items():
        print(f"  {key:6s} {float(value):.4f}")

    if args.train_steps > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        model.train()
        first = last = None
        for step in range(1, args.train_steps + 1):
            config = curriculum_config(step / args.train_steps)
            train_batch = forge_batch(args.batch_size, args.size, args.size, config, device=device)
            metrics = train_step(model, train_batch, optimizer)
            first = metrics["total"] if first is None else first
            last = metrics["total"]
        print(f"Trained {args.train_steps} steps. total loss {first:.4f} -> {last:.4f}")

    print("Smoke test complete.")


if __name__ == "__main__":
    main()
