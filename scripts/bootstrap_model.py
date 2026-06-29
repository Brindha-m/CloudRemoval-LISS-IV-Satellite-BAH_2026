"""Fast bootstrap checkpoint (no full GAN training loop).

Usage:
    python scripts/bootstrap_model.py
    python scripts/bootstrap_model.py --steps 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from fusion.dataset import generate_fusion_batch
from fusion.discriminator import PatchDiscriminator
from fusion.generator import DCMFUNet
from fusion.losses import lsgan_d_loss, lsgan_g_loss, masked_l1, spectral_loss, ssim_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap fusion generator checkpoint")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--checkpoint", default="outputs/checkpoints/fusion_generator.pt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = DCMFUNet().to(device)
    discriminator = PatchDiscriminator().to(device)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    print(f"Bootstrap training on {device} for {args.steps} steps...")
    for step in range(1, args.steps + 1):
        batch = generate_fusion_batch(args.batch_size, args.size, device=device)
        with torch.no_grad():
            fake = generator(batch["cloudy"], batch["sar"], batch["mask"])
        d_loss = lsgan_d_loss(
            discriminator(batch["clear"], batch["sar"]),
            discriminator(fake, batch["sar"]),
        )
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_d.step()

        fake = generator(batch["cloudy"], batch["sar"], batch["mask"])
        g_loss = (
            masked_l1(fake, batch["clear"], batch["mask"])
            + 0.3 * lsgan_g_loss(discriminator(fake, batch["sar"]))
            + 0.2 * spectral_loss(fake, batch["clear"])
            + 0.2 * ssim_loss(fake, batch["clear"])
        )
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()
        if step % 10 == 0 or step == args.steps:
            print(f"  step {step}/{args.steps}  g_loss={float(g_loss):.4f}")

    ckpt = Path(args.checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": generator.state_dict(), "step": args.steps}, ckpt)
    print(f"Saved {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
