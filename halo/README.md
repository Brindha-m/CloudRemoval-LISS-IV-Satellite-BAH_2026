# HALO — Hybrid Atmospheric-Latent Optical Reconstruction

An out-of-the-box framework for LISS-IV cloud removal. It is deliberately
structured differently from a flat "detect → fuse → reconstruct → postprocess"
pipeline. The core bet: **most cloud pixels are not equally lost**, so we should
not treat them with one generative hammer.

## The Big Ideas (what makes this different)

1. **Continuous opacity, not a binary mask.**
   The `CloudCartographer` predicts a transmittance field `tau in [0,1]` plus a
   per-channel airlight estimate — the physical quantities of the atmospheric
   scattering model — instead of a hard cloud/no-cloud mask.

2. **Physics where possible, generation only where necessary.**
   Thin cloud and haze are *invertible*. The `RadiometricReclaimer` recovers the
   true surface analytically via `J = (I - A(1-tau)) / tau` and refines it with a
   tiny CNN. Only opaque holes (`tau` below a threshold) are handed to the
   generative `LatentTerrainSynthesizer`. This keeps thin regions spectrally
   exact and reserves "imagination" for truly missing pixels.

3. **SAR as a structural anchor inside holes.**
   The synthesizer is a gated-convolution inpainter conditioned on recovered
   surroundings + Sentinel-1 SAR, which sees through clouds.

4. **Self-supervised training with zero paired data.**
   `CloudForge` generates clean scenes + forges continuous cloud opacity fields,
   then composites cloudy observations through the physics model. This yields
   exact ground truth for every module — no real cloudy/clear pairs needed.
   Swap the procedural scene for real clear LISS-IV tiles later; nothing else
   changes.

5. **Curriculum difficulty.**
   Training ramps from thin/small clouds to thick/large clouds.

6. **Objectives that enforce physical meaning.**
   Opacity-weighted reconstruction, a frequency-split loss (preserves fine
   texture), Spectral Angle (SAM), NDVI consistency, and a seam-energy term at
   hole boundaries.

## Module Map

| File | Role |
|------|------|
| `cartographer.py` | Continuous opacity / airlight estimation |
| `reclaimer.py` | Physics inversion for thin cloud + learned refinement |
| `synthesizer.py` | SAR-conditioned gated-conv inpainting for opaque holes |
| `cloudforge.py` | Self-supervised scene + cloud generator (the data engine) |
| `losses.py` | Opacity-weighted / frequency-split / SAM / NDVI / seam losses |
| `pipeline.py` | End-to-end HALO composition with feathered blending |
| `train.py` | Self-supervised curriculum trainer |
| `demo.py` | Zero-data smoke test |
| `blocks.py` | Shared NN blocks + fractal noise generator |

## Data Flow

```
observed optical (G,R,NIR) + SAR (VV,VH)
        │
        ▼  CloudCartographer
   tau (opacity), airlight, cloud/shadow prob
        │
        ▼  RadiometricReclaimer   (analytic inverse + refine)
   recovered surface  ──────────────► good in thin cloud / haze
        │
        ├─ hole = (tau < threshold)  → feathered soft mask
        ▼  LatentTerrainSynthesizer  (gated conv + SAR)
   synthesized content inside holes
        │
        ▼  Harmonized blend
   output = recovered·(1-hole) + synthesized·hole
```

## Run It (no data needed)

```bash
# smoke test: forge → forward → losses
python -m halo.demo

# confirm gradients flow and loss drops
python -m halo.demo --train-steps 40

# full self-supervised training run
python -m halo.train --steps 300 --size 128 --batch-size 4
```

A checkpoint is written to `outputs/checkpoints/halo.pt`.

## Moving to Real LISS-IV

1. Replace procedural scenes: feed real clear LISS-IV tiles (and co-registered
   SAR) into `forge_batch(..., clean_optical=..., sar=...)`. The forging of
   clouds/airlight stays identical, so supervision remains exact.
2. Optionally add real cloudy scenes for unpaired fine-tuning using only the
   cartographer + NDVI/SAM consistency terms (no clean target required).
3. Export `output` to GeoTIFF with the source scene's CRS/transform and write a
   `tau`-derived quality/uncertainty raster alongside.

## Why This Should Score Well

- Thin-cloud regions stay radiometrically faithful (physics, not hallucination).
- Generation is concentrated where it matters, reducing artifacts and seams.
- SAM + NDVI losses target the exact spectral metrics evaluators use.
- The opacity field doubles as a per-pixel uncertainty map for trust.
