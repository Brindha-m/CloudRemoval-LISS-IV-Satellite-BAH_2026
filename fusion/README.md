# SAR-Optical Multi-Modal Fusion for LISS-IV Cloud Removal

A multi-modal cloud-removal model that fuses optical (LISS-IV: Green, Red, NIR)
with Sentinel-1 SAR (VV, VH). SAR penetrates cloud, so it provides structure
exactly where the optical signal is lost.

## Architecture: DCMF-UNet (Dual-Encoder Cross-Modal Fusion U-Net)

```
optical (3) + cloud mask (1) ─► Optical Encoder (mask-aware gated convs) ─┐
                                                                          ├─► CrossModalFusionGate (per scale)
SAR (VV, VH)                 ─► SAR Encoder (conv blocks) ────────────────┘            │
                                                                          SE bottleneck │
                                                                                        ▼
                                                          U-Net Decoder (gated convs) ─► generated optical
                                                                                        │
                                          output = optical·(1-mask) + generated·mask  ◄─┘
```

Key design choices (different from a flat detect/fuse/reconstruct stack):

- **Two parallel encoders** so SAR and optical are learned in their own
  statistics before fusion.
- **CrossModalFusionGate at every scale**: a per-pixel gate (conditioned on the
  cloud mask) decides how much to pull from SAR vs optical. It opens toward SAR
  inside clouds.
- **Mask-aware gated convolutions** so the network knows which optical pixels
  are valid.
- **Clear-pixel preservation**: the output keeps original pixels outside the
  mask and only synthesizes inside it — ideal for the evaluation metrics.

Discriminator: a conditional 70x70 PatchGAN that sees `[optical | SAR]`, forcing
generated content to be consistent with the radar structure.

## Losses

`L = w1·maskedL1 + w2·perceptual(VGG) + w3·LSGAN + w4·spectral(SAM + band stats) + w5·(1-SSIM)`

Perceptual loss auto-disables if torchvision is missing.

## Files

| File | Role |
|------|------|
| `generator.py` | DCMF-UNet generator |
| `discriminator.py` | Conditional PatchGAN |
| `blocks.py` | Gated conv, cross-modal fusion gate, SE |
| `losses.py` | L1 / perceptual / LSGAN / spectral / SSIM |
| `dataset.py` | Paired loader + synthetic generator |
| `preprocess/` | co-registration, patch extraction, cloud mask |
| `train.py` | GAN training loop |
| `infer.py` | Tiled inference + GeoTIFF in/out (CRS preserved) |
| `evaluate.py` | PSNR / SSIM / SAM |
| `demo.py` | Zero-data smoke test |

## Run (no data needed)

```bash
python -m fusion.demo --train-steps 40
```

## Train

```bash
# synthetic bootstrap
python -m fusion.train --synthetic --steps 300

# real paired data
python -m fusion.train --data-root data/fusion --epochs 10 --size 256
```

Expected layout for real data:

```
data/fusion/
  lissiv_cloudy/<id>.tif
  lissiv_clear/<id>.tif
  sentinel1/<id>.npy        # 2-band VV/VH
  masks/<id>.png            # white = cloud
```

## Inference (GeoTIFF in, cloud-free GeoTIFF out)

```bash
python -m fusion.infer \
  --checkpoint outputs/checkpoints/fusion_generator.pt \
  --optical scene_optical.tif --sar scene_sar.tif --out scene_clear.tif
```

CRS and transform are copied from the input optical scene.

## Recommended Workflow

1. Pretrain on abundant Sentinel-2 + Sentinel-1 (similar spectral) if available.
2. Fine-tune on a small set of LISS-IV + SAR pairs (even 50-100 helps).
3. Co-register SAR to the LISS-IV grid via `preprocess/coregister.py`.
4. Generate masks via `preprocess/cloud_mask.py` (or s2cloudless/Fmask).
5. Train, then run `fusion.infer` on held-out cloudy scenes.
6. Report PSNR/SSIM/SAM with `fusion.evaluate`.
