from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from demo import config
from demo.common.audio_processing import compute_mel_spectrogram
from demo.pc.state import AppState


def render(state: AppState) -> None:
    st.subheader("Advanced audio")
    samples = state.get_audio(config.MEL_SECONDS)
    figure, axis = plt.subplots(figsize=(6.0, 2.6))
    result = None
    if samples.size >= config.MEL_N_FFT:
        result = compute_mel_spectrogram(
            samples,
            sample_rate=config.AUDIO_RATE,
            n_fft=config.MEL_N_FFT,
            hop_length=config.MEL_HOP_LENGTH,
            n_mels=config.MEL_N_MELS,
            fmin=config.MEL_FMIN,
            fmax=config.MEL_FMAX,
        )

    if result is None:
        axis.set_xlim(0.0, config.MEL_SECONDS)
        axis.set_ylim(0.0, config.MEL_FMAX)
        axis.text(
            0.5,
            0.5,
            "Waiting for audio stream...",
            transform=axis.transAxes,
            ha="center",
            va="center",
            alpha=0.7,
        )
        mesh = axis.pcolormesh(
            [[0.0, config.MEL_SECONDS], [0.0, config.MEL_SECONDS]],
            [[0.0, 0.0], [config.MEL_FMAX, config.MEL_FMAX]],
            [[-80.0, -80.0], [-80.0, -80.0]],
            shading="auto",
            cmap="magma",
            vmin=-80.0,
            vmax=0.0,
        )
    else:
        times, frequencies, spectrogram_db = result
        mesh = axis.pcolormesh(
            times,
            frequencies,
            spectrogram_db,
            shading="auto",
            cmap="magma",
        )

    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Frequency (Hz)")
    axis.set_ylim(0.0, config.MEL_FMAX)
    figure.colorbar(mesh, ax=axis, format="%+2.0f dB", pad=0.02)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    result = state.get_audio_model_result()
    with st.container(border=True, height=90):
        if result:
            label = result.get("class", "unknown")
            probs = result.get("probabilities", {})
            prob_text = " | ".join(f"{k}: {float(v):.2f}" for k, v in probs.items())
            st.write(f"**Audio model:** {label}")
            st.caption(prob_text or "No probabilities")
        else:
            st.write("**Audio model:** waiting for result...")
            st.caption("EfficientAT (vehicle/drone/footsteps/background)")
