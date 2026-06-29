from __future__ import annotations

import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.bhoonidhi import (
    LISS4_COLLECTIONS,
    BhoonidhiClient,
    build_datetime_range,
    summarize_feature,
)
from src.config import RAW_DIR, get_credentials
from ui.scene_viz import build_cloud_probability_rgb, load_upload_rgb, reconstruct_upload
from src.regions import all_region_names, get_region
from ui.styles import inject_styles
from ui.training import run_training_ui


APP_TITLE = "Generative AI-Based Cloud Removal and Reconstruction for LISS-IV Satellite Imagery"
APP_SUBTITLE = "DCMF-UNet · dual-encoder SAR–optical fusion GAN with cloud-mask guided cross-modal gating"
APP_FOOTER = "BAH 2026"
CHECKPOINT_PATH = Path("outputs/checkpoints/fusion_generator.pt")

PAGES = ["Results", "Pipeline", "Download", "Preprocess", "Train"]
DEFAULT_PAGE = "Results"


def page_config() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🛰️", layout="wide")


def header(title: str, sub: str = "") -> None:
    st.markdown(f"## {title}")
    if sub:
        st.caption(sub)


def sidebar() -> str:
    with st.sidebar:
        st.markdown(f'<p class="app-title">{APP_TITLE}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="app-subtitle">{APP_SUBTITLE}</p>', unsafe_allow_html=True)
        st.divider()
        if "nav_page" not in st.session_state:
            st.session_state.nav_page = DEFAULT_PAGE
        page = st.selectbox(
            "Menu",
            PAGES,
            index=PAGES.index(st.session_state.nav_page),
            label_visibility="collapsed",
        )
        st.session_state.nav_page = page
        st.divider()
        st.markdown(f"Model: {'ready' if CHECKPOINT_PATH.exists() else 'not trained'}")
        if CHECKPOINT_PATH.exists():
            st.caption(f"{CHECKPOINT_PATH.name} ({CHECKPOINT_PATH.stat().st_size / 1e6:.1f} MB)")
        st.divider()
        st.markdown(f'<p class="app-footer">{APP_FOOTER}</p>', unsafe_allow_html=True)
    return page


# ------------------------------------------------------------------ Pipeline
def page_pipeline() -> None:
    header("Workflow", "Core steps only — what we actually run")

    st.markdown(
        """
| Step | Action | Command / page |
|------|--------|----------------|
| **1** | Download LISS-IV (Bhoonidhi) | **Download** tab |
| **2** | Stack BAND2/3/4 → G/R/NIR, cloud masks | **Preprocess** tab |
| **3** | Train fusion model | **Train** tab |
| **4** | Cloud-free GeoTIFF + side-by-side PNG | **Results** tab |
        """
    )

    st.code(
        "# Full CLI sequence\n"
        "python scripts/download_ner.py --start 2026-01-20 --end 2026-01-29 --region Assam\n"
        "python scripts/process_bhoonidhi.py --input data/raw/liss4 --role cloudy\n"
        "python -m fusion.train --data-root data/fusion --epochs 10\n"
        "python scripts/generate_viz.py --pairs 2",
        language="bash",
    )

    st.info("Model: **DCMF-UNet** — dual-encoder SAR + optical fusion GAN (~8M params). Input: cloudy G/R/NIR + SAR VV/VH + mask → cloud-free LISS-IV.")


# ------------------------------------------------------------------ Download
@st.cache_resource
def get_client() -> BhoonidhiClient:
    return BhoonidhiClient()


def check_connection() -> dict[str, str]:
    import socket
    import requests

    out = {"public_ip": "", "api_ok": "no", "hint": ""}
    try:
        out["public_ip"] = requests.get("https://api.ipify.org", timeout=10).text.strip()
    except Exception as exc:  # noqa: BLE001
        out["public_ip"] = str(exc)

    def tcp(host: str) -> bool:
        try:
            with socket.create_connection((host, 443), timeout=8):
                return True
        except OSError:
            return False

    api_ok = tcp("bhoonidhi-api.nrsc.gov.in")
    out["api_ok"] = "yes" if api_ok else "no"
    out["hint"] = "API OK" if api_ok else "IP not whitelisted — email bhoonidhi@nrsc.gov.in"
    return out


def page_download() -> None:
    header("Download LISS-IV", "Bhoonidhi API · NER India")

    if st.button("Check connection"):
        d = check_connection()
        c1, c2 = st.columns(2)
        c1.metric("Your IP", d["public_ip"])
        c2.metric("API reachable", d["api_ok"])
        (st.success if d["api_ok"] == "yes" else st.error)(d["hint"])

    client = get_client()
    creds = get_credentials()

    with st.expander("Login", expanded="token" not in st.session_state):
        with st.form("login"):
            uid = st.text_input("User ID", value=creds.user_id)
            pwd = st.text_input("Password", value=creds.password, type="password")
            if st.form_submit_button("Login"):
                try:
                    st.session_state["token"] = client.authenticate(uid, pwd)
                    st.success("OK")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

    token = st.session_state.get("token")
    if not token:
        st.warning("Login first.")
        return

    c1, c2, c3 = st.columns(3)
    region_name = c1.selectbox("Region", all_region_names())
    t0 = c2.date_input("Start", st.session_state.get("t0", date(2026, 1, 20)), key="start_d")
    t1 = c3.date_input("End", st.session_state.get("t1", date(2026, 1, 29)), key="end_d")

    p1, p2, _ = st.columns(3)
    if p1.button("Preset: 10 days (Jan 2026)"):
        st.session_state["t0"] = date(2026, 1, 20)
        st.session_state["t1"] = date(2026, 1, 29)
        st.rerun()
    if p2.button("Preset: last 10 days"):
        st.session_state["t1"] = date.today()
        st.session_state["t0"] = date.today() - timedelta(days=9)
        st.rerun()
    t0 = st.session_state.get("t0", t0)
    t1 = st.session_state.get("t1", t1)

    if st.button("Search", type="primary"):
        try:
            region = get_region(region_name)
            features = client.search_all(
                token=token,
                collections=LISS4_COLLECTIONS,
                datetime_range=build_datetime_range(str(t0), str(t1)),
                bbox=region.bbox,
                max_items=500,
                online_only=True,
            )
            st.session_state["features"] = features
            st.session_state["period"] = str(t0)[:7]
            st.session_state["region"] = region_name
            st.success(f"{len(features)} products")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

    features = st.session_state.get("features", [])
    if not features:
        return

    st.dataframe([summarize_feature(f) for f in features], use_container_width=True, hide_index=True)
    period = st.session_state.get("period", str(t0)[:7])
    safe = st.session_state.get("region", region_name).replace(" ", "_")
    dest = RAW_DIR / period / safe

    if st.button("Download all"):
        dest.mkdir(parents=True, exist_ok=True)
        bar = st.progress(0.0)

        def prog(i, n, _):
            bar.progress(i / max(n, 1))

        results = client.batch_download(token, features, dest, progress_callback=prog)
        ok = sum(1 for r in results if r["status"] == "downloaded")
        st.success(f"Saved {ok} files to {dest}")


# ------------------------------------------------------------------ Preprocess
def page_preprocess() -> None:
    header("Preprocess", "BAND2=Green, BAND3=Red, BAND4=NIR")

    st.markdown(
        """
After extract from ZIP you get `BAND2.tif`, `BAND3.tif`, `BAND4.tif`.

**Note:** Full-resolution stacks need **~3.5 GB per scene**. Default downsamples to 2048 px (~10 MB/scene).
The **Results** tab reads raw bands directly — stacking is only needed for fusion training.
        """
    )

    raw_path = st.text_input("Raw download folder", str(RAW_DIR))
    role = st.selectbox("Role", ["cloudy", "clear"])
    max_edge = st.number_input("Max edge (px)", min_value=512, max_value=8192, value=2048, step=256)
    full_res = st.checkbox("Full resolution (needs ~3.5 GB/scene free disk)", value=False)

    if st.button("Run stack bands", type="primary"):
        cmd = [
            sys.executable,
            "scripts/process_bhoonidhi.py",
            "--input",
            raw_path,
            "--role",
            role,
            "--overwrite",
        ]
        if full_res:
            cmd.append("--full-res")
        else:
            cmd.extend(["--max-edge", str(int(max_edge))])
        with st.spinner("Processing..."):
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
        st.code(r.stdout + r.stderr)
        if r.returncode == 0:
            st.success(f"Output → data/fusion/lissiv_{role}/")
        else:
            st.error("Failed — check path and install rasterio: pip install rasterio")

    st.markdown("**Target folder layout for training:**")
    st.code(
        "data/fusion/\n"
        "  lissiv_cloudy/   # cloudy scenes\n"
        "  lissiv_clear/    # cloud-free targets\n"
        "  sentinel1/       # SAR VV/VH (optional)\n"
        "  masks/           # white = cloud",
        language="text",
    )


# ------------------------------------------------------------------ Train
def page_train() -> None:
    header("Train", "DCMF-UNet SAR-optical fusion on your LISS-IV data")

    c1, c2, c3 = st.columns(3)
    epochs = c1.number_input("Epochs", min_value=1, max_value=100, value=10, step=1)
    batch_size = c2.number_input("Batch size", min_value=1, max_value=8, value=4, step=1)
    patch_size = c3.number_input("Patch size", min_value=128, max_value=512, value=256, step=64)

    st.caption("Uses **cloudy-only** mode when `lissiv_clear/` is missing. Checkpoint saved after every epoch.")

    if CHECKPOINT_PATH.exists():
        size_mb = CHECKPOINT_PATH.stat().st_size / 1e6
        st.info(f"Current checkpoint: `{CHECKPOINT_PATH}` ({size_mb:.1f} MB)")
    else:
        st.warning("No checkpoint yet — train below to create one.")

    st.markdown("**Equivalent CLI:**")
    st.code(
        f"python -m fusion.train --data-root data/fusion --epochs {epochs} "
        f"--batch-size {batch_size} --size {patch_size}",
        language="bash",
    )

    if st.button("Start training", type="primary"):
        run_training_ui(
            epochs=int(epochs),
            batch_size=int(batch_size),
            size=int(patch_size),
            checkpoint=str(CHECKPOINT_PATH),
        )

    st.markdown("**After training:** open **Results** to generate cloudy | clear visualizations.")


# ------------------------------------------------------------------ Results
def render_cloud_removal_pair(cloudy_rgb, clear_rgb, left_label: str = "Cloudy LISS-IV", right_label: str = "Cloud-free reconstruction") -> None:
    st.markdown('<div class="viz-pair">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.image(cloudy_rgb, use_container_width=True)
        st.markdown(f'<p class="viz-label">{left_label}</p>', unsafe_allow_html=True)
    with c2:
        st.image(clear_rgb, use_container_width=True)
        st.markdown(f'<p class="viz-label">{right_label}</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def ensure_checkpoint() -> bool:
    if CHECKPOINT_PATH.exists():
        return True
    st.warning("Train the model first on the **Train** tab.")
    return False


def page_results() -> None:
    st.markdown('<p class="viz-title">Cloud Removal — Final Output</p>', unsafe_allow_html=True)
    st.caption("Nimbo-style removal: probabilistic cloud mask + DCMF-UNet reconstruction")

    f = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    c1, c2, c3, c4 = st.columns(4)
    brightness = c1.slider("Cloud brightness", 0.35, 0.90, 0.50, 0.01)
    whiteness = c2.slider("Cloud whiteness", 0.55, 0.98, 0.82, 0.01)
    prob_thr = c3.slider("Cloud opacity cut", 0.20, 0.80, 0.45, 0.01)
    dilate = c4.slider("Mask expand", 0, 3, 1, 1)

    if not f:
        st.info("Upload a cloudy optical image (PNG, JPG, or GeoTIFF). Train 10+ epochs on Colab for Nimbo-quality detail.")
        if not CHECKPOINT_PATH.exists():
            st.warning("No checkpoint — run **Train** tab or Colab notebook first.")
        return

    rgb = load_upload_rgb(Image.open(f), max_edge=1024)
    from fusion.preprocess.cloud_mask import apply_probability_threshold

    cloud_prob = build_cloud_probability_rgb(rgb, brightness, whiteness, dilate_iters=int(dilate))
    cloud_prob = apply_probability_threshold(cloud_prob, prob_thr)
    cloud_pct = float(cloud_prob.mean() * 100)

    if cloud_pct < 1.0:
        st.error("Cloud mask is empty — lower **Cloud brightness** or **Cloud whiteness** until clouds are highlighted below.")
        mask_rgb = np.stack([cloud_prob] * 3, axis=-1)
        st.image(mask_rgb, caption="Detected cloud mask (white = remove)", use_container_width=True)
        return

    st.caption(f"Cloud cover detected: **{cloud_pct:.0f}%** of image")

    with st.spinner("Removing clouds..."):
        cloudy_rgb, clear_rgb, mask_rgb = reconstruct_upload(rgb, cloud_prob, CHECKPOINT_PATH)

    with st.expander("Cloud probability map"):
        st.image(mask_rgb, caption="Brighter = more opaque cloud (removed)", use_container_width=True)
    render_cloud_removal_pair(cloudy_rgb, clear_rgb)
    if not CHECKPOINT_PATH.exists():
        st.warning("For Nimbo-like sharp forests: train `fusion_generator.pt` 10–20 epochs on Colab with your LISS-IV `.tif` stacks.")


def main() -> None:
    page_config()
    inject_styles()
    page = sidebar()
    {
        "Pipeline": page_pipeline,
        "Download": page_download,
        "Preprocess": page_preprocess,
        "Train": page_train,
        "Results": page_results,
    }[page]()


if __name__ == "__main__":
    main()
