# LISS-IV Cloud Removal — ISRO PS2

SAR-optical fusion GAN for cloud-free LISS-IV reconstruction (NER India).

## Run app

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Core workflow (4 steps)

| Step | What | Command |
|------|------|---------|
| 1 | Download LISS-IV | `python scripts/download_ner.py --start 2026-01-20 --end 2026-01-29 --region Assam` |
| 2 | Stack BAND2/3/4 (2048 px, ~5 MB/scene) | `python scripts/process_bhoonidhi.py --input data/raw/liss4 --role cloudy` |
| 3 | Train on LISS-IV data | `python -m fusion.train --data-root data/fusion --epochs 10` |
| 4 | Output | `python scripts/generate_viz.py --pairs 2` |

## Output format (ISRO poster)

`scripts/generate_viz.py` writes:

```text
outputs/cloud_free/
  visualization_of_cloud_removal.png   # poster: cloudy | clear rows
  <scene>_cloudy.png
  <scene>_clear.png
  <scene>_pair.png
```

Your raw data path: `data/raw/liss4/` (each folder needs BAND2/3/4.tif).

Full-resolution stacks need ~3.5 GB/scene. Default preprocessing downsamples to 2048 px (~5–15 MB/scene). Use `--full-res` only if you have enough disk space.

## Bhoonidhi extract

`BAND2.tif` = Green, `BAND3.tif` = Red, `BAND4.tif` = NIR

Credentials in `.env`: `BHOONIDHI_USER_ID`, `BHOONIDHI_PASSWORD`

## Model

DCMF-UNet: dual-encoder SAR + optical fusion GAN (`fusion/`)

## Sample Output
<img width="604" height="300" alt="image" src="https://github.com/user-attachments/assets/861e93db-4bd0-4029-9a83-0c0e70a0268e" />


<!--
## Train on Google Colab (GPU)

Use the self-contained folder **`colab_training/`**:

1. Run `python scripts/process_bhoonidhi.py --input data/raw/liss4 --role cloudy` on your PC
2. Upload `colab_training/` + your `lissiv_cloudy/*.tif` files to Colab
3. Open `colab_training/LISS_IV_Cloud_Training.ipynb` and run all cells
4. Download `fusion_generator.pt` → place in `outputs/checkpoints/` on your PC

See `colab_training/README.md` for details.
-->

