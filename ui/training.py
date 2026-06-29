"""Streamlit training runner with live progress."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from fusion.train import default_train_args, train


def run_training_ui(
    *,
    epochs: int,
    batch_size: int,
    size: int = 256,
    data_root: str = "data/fusion",
    checkpoint: str = "outputs/checkpoints/fusion_generator.pt",
) -> bool:
    args = default_train_args()
    args.epochs = int(epochs)
    args.batch_size = int(batch_size)
    args.size = int(size)
    args.data_root = data_root
    args.checkpoint = checkpoint
    args.log_every = 1

    progress_bar = st.progress(0.0, text="Starting training...")
    status = st.empty()
    log_area = st.empty()
    lines: list[str] = []

    def on_progress(info: dict) -> None:
        pct = float(info.get("progress", 0.0))
        phase = info.get("phase", "step")
        epoch = info.get("epoch", 0)
        total_epochs = info.get("epochs", epochs)
        step = info.get("step", 0)
        total_steps = info.get("total_steps", 1)

        if phase == "start":
            progress_bar.progress(0.0, text=f"Loading model on {info.get('device', 'cpu')}...")
        elif phase == "done":
            progress_bar.progress(1.0, text="Training complete")
        else:
            progress_bar.progress(
                min(pct, 1.0),
                text=f"Epoch {epoch}/{total_epochs} · step {step}/{total_steps}",
            )

        msg = str(info.get("message", ""))
        if msg:
            lines.append(msg)
            log_area.code("\n".join(lines[-40:]), language="text")

        if phase == "epoch_done":
            ckpt = info.get("checkpoint", checkpoint)
            status.success(f"Epoch {epoch} saved → `{ckpt}`")

    try:
        train(args, progress=on_progress)
    except Exception as exc:  # noqa: BLE001
        status.error(f"Training failed: {exc}")
        return False

    ckpt_path = Path(checkpoint)
    if ckpt_path.exists():
        size_mb = ckpt_path.stat().st_size / 1e6
        status.success(f"Model saved: `{ckpt_path}` ({size_mb:.1f} MB)")
        st.session_state["last_checkpoint"] = str(ckpt_path)
        return True

    status.error("Training finished but checkpoint file was not found.")
    return False
