"""Educational content for architecture and dataset pages."""

FUSION_ARCHITECTURE = """
### DCMF-UNet — Dual-Encoder Cross-Modal Fusion U-Net

**Problem framing:** Cloud removal on LISS-IV is *image-to-image translation* on multispectral data (Green, Red, NIR @ ~5.8 m).
Thick clouds fully hide the surface in optical bands, but **Sentinel-1 SAR (VV/VH) penetrates cloud** and preserves
structure. The model fuses both modalities so reconstruction inside cloud pixels is anchored to real geometry.

---

#### End-to-end data flow

```mermaid
flowchart TB
    subgraph inputs [Inputs per patch]
        O[LISS-IV cloudy<br/>3 bands G/R/NIR]
        S[Sentinel-1 SAR<br/>VV + VH]
        M[Cloud mask<br/>1 = reconstruct]
    end

    subgraph encoders [Dual encoders]
        OE[Optical encoder<br/>mask-aware gated convs]
        SE[SAR encoder<br/>standard conv blocks]
    end

    subgraph fusion [Multi-scale fusion]
        FG1[CrossModalFusionGate<br/>scale 1]
        FG2[CrossModalFusionGate<br/>scale 2]
        FG3[CrossModalFusionGate<br/>scale 3]
        BN[SE bottleneck]
    end

    subgraph decode [Decoder]
        DEC[U-Net decoder<br/>gated conv + skips]
        GEN[Generated optical 3-band]
    end

    OUT[Cloud-free LISS-IV<br/>clear pixels preserved]

    O --> OE
    M --> OE
    S --> SE
    OE --> FG1
    SE --> FG1
    OE --> FG2
    SE --> FG2
    OE --> FG3
    SE --> FG3
    FG3 --> BN --> DEC --> GEN
    O --> OUT
    GEN --> OUT
    M --> OUT
```

---

#### Why two encoders?

| Design choice | Rationale |
|---|---|
| Separate optical & SAR encoders | Optical and radar have different statistics; fusing too early blurs both |
| Mask concatenated to optical input | Gated convolutions learn which optical pixels are trustworthy |
| Fusion gate at **every** scale | Clouds are multi-scale; SAR structure must guide fine and coarse features |
| Gate conditioned on cloud mask | Gate opens toward SAR **inside** clouds, keeps optical **outside** |
| Clear-pixel preservation | `output = optical × (1−mask) + generated × mask` — no hallucination on clear land |

---

#### CrossModalFusionGate (core innovation)

At each encoder level the gate computes:

1. **Merge** — concatenate optical and SAR features, project to shared channels.
2. **Gate** — spatial map in `[0,1]` from `[optical, SAR, mask]`.
3. **Fuse** — `fused = optical × (1−gate) + merge × gate`.
4. **Refine** — small conv block on the fused tensor.

In cloud pixels the gate learns to trust SAR; in clear pixels it keeps optical features.

---

#### Discriminator (PatchGAN 70×70)

Conditional PatchGAN receives **`[reconstructed optical | SAR]`** (5 channels).
It scores **local realism** and forces generated content to match radar structure —
critical for vegetation edges, rivers, and urban boundaries under cloud.

---

#### Loss function

```
L_total = λ1·Masked L1
        + λ2·Perceptual (VGG16 features)
        + λ3·LSGAN adversarial
        + λ4·Spectral (SAM + per-band mean/std)
        + λ5·(1 − SSIM)
```

| Loss | What it enforces |
|---|---|
| Masked L1 | Pixel accuracy, higher weight on cloud pixels |
| Perceptual | Texture and spatial structure |
| LSGAN | Realistic local patches |
| Spectral / SAM | Multispectral fidelity (evaluators use SAM) |
| SSIM | Structural similarity |

---

#### Model size & inference

| Component | Parameters |
|---|---|
| DCMF-UNet generator | ~8 M |
| PatchGAN discriminator | ~2.8 M |
| Patch size (train) | 256×256 |
| Inference | Overlapping tiles + feather blend; GeoTIFF CRS preserved |
"""

HALO_ARCHITECTURE = """
### HALO — Hybrid Atmospheric-Latent Optical Reconstruction

**Core idea:** Not all cloud pixels are equally lost. Thin cloud/haze is **physically invertible**;
only **opaque holes** need generative synthesis. HALO uses a continuous opacity field τ instead of a hard mask.

---

#### Pipeline

```mermaid
flowchart LR
    IN[Cloudy LISS-IV + SAR] --> C[Cloud Cartographer<br/>τ, airlight, cloud prob]
    C --> R[Radiometric Reclaimer<br/>J = I−A·1−τ / τ]
    C --> H{τ < threshold?}
    R --> H
    H -->|opaque hole| S[Latent Terrain Synthesizer<br/>gated conv + SAR]
    R --> B[Harmonized blend]
    S --> B
    B --> OUT[Cloud-free output]
```

---

#### Modules

| Module | Role | Why it matters |
|---|---|---|
| **Cloud Cartographer** | Predicts transmittance τ∈[0,1], airlight A, cloud/shadow probability | Continuous opacity > binary mask; supplies physics variables |
| **Radiometric Reclaimer** | Analytic inverse `J=(I−A(1−τ))/τ` + tiny CNN refine | Thin cloud stays spectrally exact — no hallucination |
| **Latent Terrain Synthesizer** | Gated-conv inpainter, SAR-conditioned | Fills only where surface is truly unrecoverable |
| **Cloud Forge** | Self-supervised data engine | Forges clouds on procedural/real clear scenes — **no paired data required** |

---

#### Training (self-supervised)

1. Generate or load a **clean** multispectral scene + SAR.
2. Forge continuous cloud opacity τ and airlight A.
3. Composite cloudy observation: **`I = J·τ + A·(1−τ)`**.
4. Supervise every module with exact ground truth.
5. **Curriculum:** thin/small clouds → thick/large clouds.

---

#### HALO vs Fusion — when to use which

| Aspect | DCMF-UNet (Fusion) | HALO |
|---|---|---|
| Primary strength | SAR-guided generative inpainting | Physics + selective generation |
| Training data | Paired cloudy/clear + SAR (or synthetic pairs) | Can train with zero real pairs (Cloud Forge) |
| Best for | Thick cloud, strong SAR available | Mixed thin/thick cloud, spectral fidelity |
| Uncertainty | Via mask | τ field = per-pixel confidence map |
"""

DATASET_GUIDE = """
### Dataset strategy for ISRO Problem Statement 2

Cloud removal is **supervised image-to-image translation**. You need aligned inputs and targets.

---

#### Primary dataset — LISS-IV (Bhoonidhi)

| Property | Detail |
|---|---|
| Sensor | Resourcesat-2 / 2A LISS-IV MX70 L2 |
| Resolution | ~5.8 m |
| Bands | Green, Red, NIR (3 channels) |
| Collections | `ResourceSat-2_LISS4-MX70_L2`, `ResourceSat-2A_LISS4-MX70_L2` |
| Study area | North Eastern Region (NER) India — 8 states |
| API | [Bhoonidhi API](https://bhoonidhi.nrsc.gov.in/bhoonidhi-api/) — requires IP whitelisting |

**What to collect:**

1. **Cloudy input** — LISS-IV scene with significant cloud cover (training input).
2. **Cloud-free target** — same area, minimal cloud (supervision target).
3. Prefer same path/row, same season, gap < 30 days, minimal land-cover change.

---

#### Auxiliary dataset — Sentinel-1 SAR (strongly recommended)

| Property | Detail |
|---|---|
| Bands | VV, VH (2 channels) |
| Why | Microwave penetrates cloud; provides structure when optical is blind |
| Temporal window | ±6 to ±12 days of LISS-IV acquisition |
| Source | Copernicus Open Access Hub / Google Earth Engine |
| Preprocessing | Co-register to LISS-IV 5.8 m grid (see `fusion/preprocess/coregister.py`) |

**Without SAR**, thick-cloud reconstruction relies purely on surrounding context — results degrade significantly.

---

#### Optional auxiliary data

| Dataset | Use |
|---|---|
| Sentinel-2 | Cloud masks (SCL band), temporal reference, pretraining |
| DEM (SRTM/ASTER) | Mountain shadow vs cloud disambiguation |
| Temporal LISS-IV / S2 stacks | Multi-date fusion for seasonal consistency |

---

#### Recommended date windows (NER India)

| Purpose | Months | Reason |
|---|---|---|
| Cloud-free references | Nov – Feb | Post-monsoon / winter, lower cloud |
| Cloudy training scenes | May – Sep | Monsoon, thick cloud diversity |
| Validation / test | Mar–Apr, Oct | Transition seasons, generalization |
| Overall window | 2018 – 2025+ | Multi-year seasonal diversity |

---

#### On-disk folder layout (fusion training)

```
data/
├── raw/liss4/                    # Bhoonidhi downloads (.zip → extract GeoTIFF)
│   └── 2026-03/North_Eastern_Region/
├── catalog/                      # STAC search JSON + CSV
└── fusion/                       # Training patches (aligned)
    ├── lissiv_cloudy/<id>.tif   # Input: cloudy optical [G,R,NIR]
    ├── lissiv_clear/<id>.tif    # Target: cloud-free optical
    ├── sentinel1/<id>.npy       # SAR [VV,VH] co-registered
    └── masks/<id>.png           # Binary mask (white = cloud pixel)
```

**Critical rule:** every `<id>` must exist in **all four folders** with the **same spatial extent**.

---

#### Preprocessing pipeline

```mermaid
flowchart LR
    A[Raw LISS-IV GeoTIFF] --> B[Atmospheric correction]
    B --> C[Co-register SAR to LISS-IV grid]
    C --> D[Cloud mask<br/>Fmask / s2cloudless / heuristic]
    D --> E[Extract 256×256 patches<br/>50% overlap]
    E --> F[data/fusion/]
```

| Step | Tool | Notes |
|---|---|---|
| Warp / resample | GDAL / Rasterio | Same CRS, transform, pixel size |
| SAR co-registration | `fusion/preprocess/coregister.py` | Sub-pixel alignment matters |
| Cloud mask | `fusion/preprocess/cloud_mask.py` or Fmask | White = reconstruct |
| Patch extract | `fusion/preprocess/patch_extract.py` | 256×256, 50% overlap |
| Augmentation | Flip, rotate, radiometric jitter | Albumentations optional |

---

#### Pairing strategy

| Strategy | Pairs needed | Difficulty |
|---|---|---|
| Fully supervised | Cloudy + clear same location | Best quality; needs good pairs |
| Synthetic bootstrap | None (forge clouds on clear tiles) | `python -m fusion.train --synthetic` |
| Pretrain S2+S1 → finetune LISS-IV | Abundant S2, few LISS-IV | Recommended for hackathon |
| Semi-supervised | Cloudy only + clear from different dates | Harder; temporal consistency loss helps |

**Minimum for demo:** 50–100 paired patches. **Strong submission:** 200+ patches across 3+ NER regions.

---

#### Evaluation (what judges check)

| Metric | Type | Target (reference) |
|---|---|---|
| PSNR | Reference | > 32 dB in cloud region |
| SSIM | Reference | > 0.90 |
| SAM | Spectral | < 5° |
| NDVI RMSE | Downstream | < 0.05 |
| Visual | Qualitative | Texture, color, no seams — primary judge check |

Report metrics on **full image** and **cloud-mask pixels only**.
"""

EVALUATION_GUIDE = """
### Evaluation protocol

Hold out **20–25 cloudy LISS-IV scenes** never seen during training.

For each scene provide:
1. Cloudy input composite (RGB or false-color)
2. Cloud mask
3. Model reconstruction
4. Reference clear scene (if available for quantitative metrics)

Use `fusion/evaluate.py` or the metrics panel in the Live Demo page.
"""
