from __future__ import annotations

import atexit

import streamlit as st

from demo import config
from demo.common.control import ControlClient
from demo.pc.panels import (
    advanced_audio_panel,
    control_panel,
    simple_audio_panel,
    video_panel,
)
from demo.pc.rtp_receiver import RtpAudioReceiver, RtpVideoReceiver
from demo.pc.state import AppState


@st.cache_resource
def _get_runtime() -> tuple[AppState, ControlClient]:
    state = AppState()
    video_receiver = RtpVideoReceiver(state.set_frame)
    audio_receiver = RtpAudioReceiver(state.append_audio)
    video_receiver.start()
    audio_receiver.start()

    atexit.register(video_receiver.stop)
    atexit.register(audio_receiver.stop)

    control_client = ControlClient(
        config.RBP_HOST,
        config.CONTROL_PORT,
        timeout=config.CONTROL_TIMEOUT_SECONDS,
    )
    return state, control_client


@st.fragment(run_every=config.UI_REFRESH_SECONDS)
def _render_live_zones(state: AppState) -> None:
    video_panel.render(state)
    simple_column, advanced_column = st.columns(2)
    with simple_column:
        simple_audio_panel.render(state)
    with advanced_column:
        advanced_audio_panel.render(state)


def run() -> None:
    st.set_page_config(
        page_title="Drone Demo",
        layout="wide",
    )

    state, control_client = _get_runtime()
    status = state.status()

    with st.sidebar:
        control_panel.render(state, control_client)
        st.divider()
        st.header("Stream")
        st.write(f"Video RTP: `{config.PC_HOST}:{config.VIDEO_RTP_PORT}`")
        st.write(f"Audio RTP: `{config.PC_HOST}:{config.AUDIO_RTP_PORT}`")
        st.write(f"Control TCP: `{config.RBP_HOST}:{config.CONTROL_PORT}`")
        st.divider()
        st.header("Capture")
        st.write(f"Video device: `{config.VIDEO_DEVICE}`")
        st.write(f"Audio device: `{config.AUDIO_DEVICE}`")
        st.write(
            f"Video: `{config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT}@{config.VIDEO_FPS}`"
        )
        st.write(f"Audio: `{config.AUDIO_RATE} Hz`, `{config.AUDIO_CHANNELS} ch`")
        st.divider()
        st.caption("Start the Raspberry side first, then this dashboard.")

    st.title("Drone Demo")
    st.caption(
        "Simulated drone video, audio, and command link. "
        "No real detection is performed; model outputs are placeholders."
    )

    video_label = "live" if status["video_live"] else "waiting"
    audio_label = "live" if status["audio_live"] else "waiting"
    st.markdown(f"**Video:** {video_label} | **Audio:** {audio_label}")

    _render_live_zones(state)
