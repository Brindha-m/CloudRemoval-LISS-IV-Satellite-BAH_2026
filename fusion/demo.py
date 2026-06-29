"""Zero-data smoke test for the SAR-Optical fusion model.

Runs synthetic batch -> generator -> discriminator -> losses -> optional
training steps -> tiled inference -> evaluation, printing shapes and metrics.

    python -m fusion.demo
    python -m fusion.demo --train-steps 40
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .dataset import generate_fusion_batch
from .discriminator import PatchDiscriminator
from .evaluate import evaluate_pair
from .generator import DCMFUNet
from .infer import reconstruct_array
from .losses import (
    LossWeights,
    PerceptualLoss,
    lsgan_d_loss,
    lsgan_g_loss,
    masked_l1,
    spectral_loss,
    ssim_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fusion model zero-data smoke test")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = DCMFUNet().to(device)
    discriminator = PatchDiscriminator().to(device)
    perceptual = PerceptualLoss().to(device)
    weights = LossWeights()

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator: {g_params/1e6:.2f}M | Discriminator: {d_params/1e6:.2f}M | device: {device}")
    print(f"Perceptual (VGG) available: {perceptual.available}")

    batch = generate_fusion_batch(args.batch_size, args.size, generator=torch.Generator().manual_seed(0))
    batch = {k: v.to(device) for k, v in batch.items()}
    for key, value in batch.items():
        print(f"  {key:7s} {tuple(value.shape)}")

    generator.eval()
    with torch.no_grad():
        fake = generator(batch["cloudy"], batch["sar"], batch["mask"])
        d_real = discriminator(batch["clear"], batch["sar"])
    print("Generator output:", tuple(fake.shape), "| Discriminator output:", tuple(d_real.shape))

    g_l1 = masked_l1(fake, batch["clear"], batch["mask"])
    g_spec = spectral_loss(fake, batch["clear"])
    g_ssim = ssim_loss(fake, batch["clear"])
    print(f"Initial losses: l1={float(g_l1):.4f} spec={float(g_spec):.4f} ssim={float(g_ssim):.4f}")

    if args.train_steps > 0:
        opt_g = torch.optim.AdamW(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        opt_d = torch.optim.AdamW(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        generator.train()
        first = last = None
        for step in range(1, args.train_steps + 1):
            b = generate_fusion_batch(args.batch_size, args.size, device=device)
            with torch.no_grad():
                fake = generator(b["cloudy"], b["sar"], b["mask"])
            d_loss = lsgan_d_loss(discriminator(b["clear"], b["sar"]), discriminator(fake, b["sar"]))
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            fake = generator(b["cloudy"], b["sar"], b["mask"])
            g_loss = (
                weights.l1 * masked_l1(fake, b["clear"], b["mask"])
                + weights.perceptual * perceptual(fake, b["clear"])
                + weights.adversarial * lsgan_g_loss(discriminator(fake, b["sar"]))
                + weights.spectral * spectral_loss(fake, b["clear"])
                + weights.ssim * ssim_loss(fake, b["clear"])
            )
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()
            first = float(g_loss) if first is None else first
            last = float(g_loss)
        print(f"Trained {args.train_steps} steps. generator loss {first:.4f} -> {last:.4f}")

    # Tiled inference + evaluation on a fresh scene.
    scene = generate_fusion_batch(1, args.size, generator=torch.Generator().manual_seed(7))
    optical = scene["cloudy"][0].cpu().numpy()
    sar = scene["sar"][0].cpu().numpy()
    mask = scene["mask"][0].cpu().numpy()
    clear = scene["clear"][0].cpu().numpy()
    generator.eval()
    reconstructed = reconstruct_array(generator, optical, sar, mask, size=args.size, device=device)
    metrics = evaluate_pair(reconstructed.astype(np.float32), clear.astype(np.float32), mask)
    print("Inference output:", reconstructed.shape)
    print("Metrics:", {k: round(v, 4) for k, v in metrics.items()})
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
