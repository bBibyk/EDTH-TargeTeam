from __future__ import annotations

import atexit

import streamlit as st

from demo import config
from demo.common.control import ControlClient
from demo.pc.audio_result_receiver import AudioResultReceiver
from demo.pc.detection_receiver import DetectionReceiver
from demo.pc.gst_receiver import GstAudioReceiver, GstVideoReceiver
from demo.pc.mjpeg_server import MjpegStreamServer
from demo.pc.panels import (
    advanced_audio_panel,
    control_panel,
    simple_audio_panel,
)
from demo.pc.state import AppState


@st.cache_resource
def _get_runtime() -> tuple[AppState, ControlClient, MjpegStreamServer]:
    state = AppState()
    video_receiver = GstVideoReceiver(state.set_frame)
    audio_receiver = GstAudioReceiver(state.append_audio)
    video_receiver.start()
    audio_receiver.start()

    mjpeg_server = MjpegStreamServer("0.0.0.0", config.VIDEO_HTTP_PORT, state)
    mjpeg_server.start()

    detection_receiver = DetectionReceiver(
        "0.0.0.0", config.DETECTION_PORT, state.set_detections
    )
    detection_receiver.start()

    audio_result_receiver = AudioResultReceiver(
        "0.0.0.0", config.AUDIO_RESULT_PORT, state.set_audio_model_result
    )
    audio_result_receiver.start()

    atexit.register(video_receiver.stop)
    atexit.register(audio_receiver.stop)
    atexit.register(mjpeg_server.stop)
    atexit.register(detection_receiver.stop)
    atexit.register(audio_result_receiver.stop)

    control_client = ControlClient(
        config.RBP_HOST,
        config.CONTROL_PORT,
        timeout=config.CONTROL_TIMEOUT_SECONDS,
    )
    return state, control_client, mjpeg_server


@st.fragment(run_every=max(config.UI_REFRESH_SECONDS * 5, 0.1))
def _render_left(state: AppState) -> None:
    simple_audio_panel.render(state)
    st.divider()
    advanced_audio_panel.render(state)


def run() -> None:
    st.set_page_config(
        page_title="Drone Demo",
        layout="wide",
    )

    state, control_client, mjpeg_server = _get_runtime()
    stream_url = mjpeg_server.url(config.PC_PUBLIC_HOST)

    with st.sidebar:
        control_panel.render(state, control_client)
        st.divider()
        st.header("Stream")
        st.write(f"Video HTTP: `{stream_url}`")
        st.write(f"Video RTP: `{config.PC_HOST}:{config.VIDEO_RTP_PORT}`")
        st.write(f"Audio RTP: `{config.PC_HOST}:{config.AUDIO_RTP_PORT}`")
        st.write(f"Detection TCP: `0.0.0.0:{config.DETECTION_PORT}`")
        st.write(f"Audio model TCP: `0.0.0.0:{config.AUDIO_RESULT_PORT}`")
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
    st.caption("Sound-first pipeline: RMS wake -> mel/vehicle model -> video+YOLO.")

    left, right = st.columns([1, 1], gap="large")
    with left:
        _render_left(state)
    with right:
        st.subheader("Video")
        if not mjpeg_server.is_running:
            st.error(f"MJPEG server unavailable: {mjpeg_server.error}")
        else:
            st.markdown(
                f'<img src="{stream_url}" style="width:100%; border-radius:8px;" />',
                unsafe_allow_html=True,
            )
        with st.container(border=True):
            st.caption("Model info here")
