from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from demo import config
from demo.common.audio_processing import compute_mel_spectrogram
from demo.pc.state import AppState


def render(state: AppState) -> None:
    st.subheader("Advanced audio")
    samples = state.get_audio(config.MEL_SECONDS)

    if samples.size < config.MEL_N_FFT:
        st.info("Waiting for RTP audio stream...")
    else:
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
            st.info("Not enough audio data for a spectrogram yet.")
        else:
            times, frequencies, spectrogram_db = result
            figure, axis = plt.subplots(figsize=(6.0, 2.6))
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

    with st.container(border=True):
        st.caption("Model info here")
