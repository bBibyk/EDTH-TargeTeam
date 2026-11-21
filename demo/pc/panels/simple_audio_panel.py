from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from demo import config
from demo.common.audio_processing import downsample, pcm16_to_float
from demo.pc.state import AppState


def render(state: AppState) -> None:
    st.subheader("Simple audio")
    samples = state.get_audio(config.WAVEFORM_SECONDS)
    figure, axis = plt.subplots(figsize=(6.0, 2.6))

    if samples.size == 0:
        axis.set_xlim(0.0, config.WAVEFORM_SECONDS)
        axis.set_ylim(-1.0, 1.0)
        axis.text(
            0.5,
            0.5,
            "Waiting for audio stream...",
            transform=axis.transAxes,
            ha="center",
            va="center",
            alpha=0.7,
        )
    else:
        values = downsample(pcm16_to_float(samples), max_points=2000)
        duration = samples.size / config.AUDIO_RATE
        times = np.linspace(0.0, duration, values.size, endpoint=False)
        axis.plot(times, values, color="#4c9be8", linewidth=0.9)
        axis.set_xlim(0.0, max(duration, config.WAVEFORM_SECONDS))

    axis.set_ylim(-1.0, 1.0)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Amplitude")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    with st.container(border=True, height=90):
        if samples.size == 0:
            st.write("**Wake:** waiting for audio...")
            st.caption("RMS threshold detection")
        else:
            level = float(np.sqrt(np.mean(pcm16_to_float(samples) ** 2)))
            threshold = config.AUDIO_RMS_WAKE_THRESHOLD
            awake = level >= threshold
            label = "WAKE" if awake else "idle"
            st.write(f"**Wake:** {label}  (rms={level:.3f}, threshold={threshold:.2f})")
            st.caption("RMS threshold detection")
