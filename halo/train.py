"""Self-supervised curriculum trainer for HALO.

No dataset on disk is required: each step forges a fresh batch of
(cloudy, clean, sar, tau, hole) via Cloud Forge and supervises every module
with exact ground truth. Difficulty ramps from thin/small to thick/large
clouds.

To train on real LISS-IV later, replace `generate_clean_scene` inputs in
Cloud Forge with real clear tiles; everything else is unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .cloudforge import curriculum_config, forge_batch
from .losses import cartographer_loss, final_loss, reclaimer_loss
from .pipeline import HaloPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HALO self-supervised on forged clouds")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--checkpoint", default="outputs/checkpoints/halo.pt")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def train_step(model: HaloPipeline, batch: dict, optimizer: torch.optim.Optimizer) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    result = model(batch["cloudy"], batch["sar"])

    carto = cartographer_loss(result["tau"], batch["tau"], result["cloud_prob"], batch["cloud_prob"])
    reclaim = reclaimer_loss(result["recovered"], batch["clean"], batch["tau"])
    losses = final_loss(result["output"], batch["clean"], batch["cloud_prob"], result["hole_soft"])
    total = losses["total"] + 0.5 * carto + 0.5 * reclaim

    total.backward()
    optimizer.step()
    return {
        "total": float(total.detach()),
        "carto": float(carto.detach()),
        "reclaim": float(reclaim.detach()),
        "recon": float(losses["recon"].detach()),
        "sam": float(losses["sam"].detach()),
    }


def train(args: argparse.Namespace) -> HaloPipeline:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HaloPipeline().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for step in range(1, args.steps + 1):
        progress = step / args.steps
        config = curriculum_config(progress)
        batch = forge_batch(args.batch_size, args.size, args.size, config, device=device)
        metrics = train_step(model, batch, optimizer)

        if step % args.log_every == 0 or step == 1:
            print(
                f"step={step:04d} cov={config.coverage:.2f} thick={config.max_thickness:.2f} "
                f"total={metrics['total']:.4f} carto={metrics['carto']:.4f} "
                f"reclaim={metrics['reclaim']:.4f} sam={metrics['sam']:.4f}"
            )

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    return model


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
