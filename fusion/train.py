"""Train the SAR-Optical fusion cloud-removal GAN.

Real data:
    python -m fusion.train --data-root data/fusion --epochs 10

Zero-data smoke / bootstrap (synthetic):
    python -m fusion.train --synthetic --steps 200
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from .dataset import CloudySceneDataset, PairedFusionDataset, SyntheticFusionDataset, has_paired_data
from .discriminator import PatchDiscriminator
from .generator import DCMFUNet
from .losses import (
    LossWeights,
    PerceptualLoss,
    lsgan_d_loss,
    lsgan_g_loss,
    masked_l1,
    spectral_loss,
    ssim_loss,
    visible_l1,
)

ProgressFn = Callable[[dict], None]


def default_train_args() -> argparse.Namespace:
    return argparse.Namespace(
        data_root="data/fusion",
        synthetic=False,
        cloudy_only=False,
        epochs=10,
        steps=0,
        batch_size=4,
        size=256,
        lr=2e-4,
        checkpoint="outputs/checkpoints/fusion_generator.pt",
        log_every=5,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAR-Optical fusion cloud removal")
    parser.add_argument("--data-root", default="data/fusion")
    parser.add_argument("--synthetic", action="store_true", help="Train on synthetic data (no disk data)")
    parser.add_argument("--cloudy-only", action="store_true", help="Train from lissiv_cloudy only (no clear pairs)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=0, help="If >0, cap total optimizer steps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--checkpoint", default="outputs/checkpoints/fusion_generator.pt")
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args(argv)


def build_loader(args: argparse.Namespace) -> tuple[DataLoader, str]:
    if args.synthetic:
        dataset = SyntheticFusionDataset(length=max(args.batch_size * 50, 200), size=args.size)
        mode = "synthetic"
    elif args.cloudy_only or not has_paired_data(args.data_root):
        dataset = CloudySceneDataset(args.data_root, size=args.size)
        mode = "cloudy-only"
    else:
        dataset = PairedFusionDataset(args.data_root, size=args.size)
        mode = "paired"

    batch_size = min(args.batch_size, len(dataset))
    drop_last = len(dataset) >= batch_size
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=drop_last)
    return loader, mode


def _emit(progress: ProgressFn | None, payload: dict) -> None:
    line = payload.get("message", "")
    print(line, flush=True)
    if progress is not None:
        progress(payload)


def _save_checkpoint(path: Path, generator: DCMFUNet, step: int, epoch: int, mode: str, best_l1: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": generator.state_dict(),
            "step": step,
            "epoch": epoch,
            "mode": mode,
            "best_l1": best_l1,
        },
        path,
    )


def train(args: argparse.Namespace, progress: ProgressFn | None = None) -> DCMFUNet:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = LossWeights()

    generator = DCMFUNet().to(device)
    discriminator = PatchDiscriminator().to(device)
    perceptual = PerceptualLoss().to(device)

    opt_g = torch.optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    loader, mode = build_loader(args)
    checkpoint_path = Path(args.checkpoint)
    steps_per_epoch = max(len(loader), 1)
    total_steps = steps_per_epoch * args.epochs if not args.steps else args.steps

    _emit(
        progress,
        {
            "phase": "start",
            "epoch": 0,
            "epochs": args.epochs,
            "step": 0,
            "total_steps": total_steps,
            "mode": mode,
            "device": str(device),
            "samples": len(loader.dataset),
            "batch_size": loader.batch_size,
            "progress": 0.0,
            "message": (
                f"Training mode: {mode} | device: {device} | "
                f"samples={len(loader.dataset)} | batch={loader.batch_size} | epochs={args.epochs}"
            ),
        },
    )

    step = 0
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        epoch_g_losses: list[float] = []
        epoch_l1_losses: list[float] = []

        for batch_idx, batch in enumerate(loader, start=1):
            cloudy = batch["cloudy"].to(device)
            clear = batch["clear"].to(device)
            sar = batch["sar"].to(device)
            mask = batch["mask"].to(device)

            with torch.no_grad():
                fake = generator(cloudy, sar, mask)
            real_logits = discriminator(clear, sar)
            fake_logits = discriminator(fake.detach(), sar)
            d_loss = lsgan_d_loss(real_logits, fake_logits)
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            fake = generator(cloudy, sar, mask)
            g_adv = lsgan_g_loss(discriminator(fake, sar))
            if mode == "cloudy-only":
                g_l1 = visible_l1(fake, cloudy, mask)
                g_perc = perceptual(fake, cloudy)
                g_spec = spectral_loss(fake, cloudy)
                g_ssim = ssim_loss(fake, cloudy)
            else:
                g_l1 = masked_l1(fake, clear, mask)
                g_perc = perceptual(fake, clear)
                g_spec = spectral_loss(fake, clear)
                g_ssim = ssim_loss(fake, clear)

            g_loss = (
                weights.l1 * g_l1
                + weights.perceptual * g_perc
                + weights.adversarial * g_adv
                + weights.spectral * g_spec
                + weights.ssim * g_ssim
            )
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            step += 1
            epoch_g_losses.append(float(g_loss))
            epoch_l1_losses.append(float(g_l1))

            if float(g_l1) < best:
                best = float(g_l1)

            if step % max(args.log_every, 1) == 0 or batch_idx == steps_per_epoch:
                frac = step / max(total_steps, 1)
                _emit(
                    progress,
                    {
                        "phase": "step",
                        "epoch": epoch,
                        "epochs": args.epochs,
                        "step": step,
                        "total_steps": total_steps,
                        "step_in_epoch": batch_idx,
                        "steps_per_epoch": steps_per_epoch,
                        "progress": min(frac, 1.0),
                        "g_loss": float(g_loss),
                        "d_loss": float(d_loss),
                        "l1": float(g_l1),
                        "spec": float(g_spec),
                        "ssim": float(g_ssim),
                        "message": (
                            f"epoch {epoch}/{args.epochs} | step {step}/{total_steps} | "
                            f"g={float(g_loss):.4f} d={float(d_loss):.4f} "
                            f"l1={float(g_l1):.4f} spec={float(g_spec):.4f} ssim={float(g_ssim):.4f}"
                        ),
                    },
                )

            if args.steps and step >= args.steps:
                _save_checkpoint(checkpoint_path, generator, step, epoch, mode, best)
                _emit(
                    progress,
                    {
                        "phase": "done",
                        "epoch": epoch,
                        "epochs": args.epochs,
                        "step": step,
                        "total_steps": total_steps,
                        "progress": 1.0,
                        "checkpoint": str(checkpoint_path),
                        "message": f"Step cap reached. Model saved to {checkpoint_path}",
                    },
                )
                return generator

        avg_g = sum(epoch_g_losses) / max(len(epoch_g_losses), 1)
        avg_l1 = sum(epoch_l1_losses) / max(len(epoch_l1_losses), 1)
        _save_checkpoint(checkpoint_path, generator, step, epoch, mode, best)
        _emit(
            progress,
            {
                "phase": "epoch_done",
                "epoch": epoch,
                "epochs": args.epochs,
                "step": step,
                "total_steps": total_steps,
                "progress": epoch / args.epochs,
                "avg_g_loss": avg_g,
                "avg_l1": avg_l1,
                "checkpoint": str(checkpoint_path),
                "message": (
                    f"Epoch {epoch}/{args.epochs} finished | avg_g={avg_g:.4f} avg_l1={avg_l1:.4f} | "
                    f"saved {checkpoint_path}"
                ),
            },
        )

    _emit(
        progress,
        {
            "phase": "done",
            "epoch": args.epochs,
            "epochs": args.epochs,
            "step": step,
            "total_steps": total_steps,
            "progress": 1.0,
            "checkpoint": str(checkpoint_path),
            "best_l1": best,
            "message": f"Training complete. Model saved to {checkpoint_path} (best_l1={best:.4f})",
        },
    )
    return generator


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
