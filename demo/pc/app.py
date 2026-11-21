from __future__ import annotations

import atexit

import streamlit as st

from demo import config
from demo.common.control import ControlClient
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
    process_frame = None
    if config.PC_DETECTION:
        # Boxes drawn on the PC: detection runs on every received frame.
        from demo.raspberry.detector.overlay import DetectionOverlay

        overlay = DetectionOverlay(
            config.YOLO_MODEL_PATH,
            classes=config.DETECTION_CLASSES,
            imgsz=config.YOLO_IMGSZ,
            conf=config.YOLO_CONF,
        )
        overlay.start()
        atexit.register(overlay.stop)
        process_frame = overlay.process
    video_receiver = GstVideoReceiver(state.set_frame, process_frame)
    audio_receiver = GstAudioReceiver(state.append_audio)
    video_receiver.start()
    audio_receiver.start()

    mjpeg_server = MjpegStreamServer("0.0.0.0", config.VIDEO_HTTP_PORT, state)
    mjpeg_server.start()

    atexit.register(video_receiver.stop)
    atexit.register(audio_receiver.stop)
    atexit.register(mjpeg_server.stop)

    control_client = ControlClient(
        config.RBP_HOST,
        config.CONTROL_PORT,
        timeout=config.CONTROL_TIMEOUT_SECONDS,
    )
    return state, control_client, mjpeg_server


@st.fragment(run_every=max(config.UI_REFRESH_SECONDS * 5, 0.1))
def _render_audio(state: AppState) -> None:
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

    state, control_client, mjpeg_server = _get_runtime()
    stream_url = mjpeg_server.url(config.PC_PUBLIC_HOST)

    with st.sidebar:
        control_panel.render(state, control_client)
        st.divider()
        st.header("Stream")
        st.write(f"Video HTTP: `{stream_url}`")
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

    @st.fragment(run_every=1.0)
    def _render_status() -> None:
        s = state.status()
        video_label = "live" if s["video_live"] else "waiting"
        audio_label = "live" if s["audio_live"] else "waiting"
        st.markdown(f"**Video:** {video_label} | **Audio:** {audio_label}")
        if mjpeg_server.is_running and not s["video_live"]:
            st.info("Waiting for video stream...")

    _render_status()

    st.subheader("Video")
    if not mjpeg_server.is_running:
        st.error(f"MJPEG server unavailable: {mjpeg_server.error}")
    else:
        st.markdown(
            f'<img src="{stream_url}" style="width:60%; max-width:900px; border-radius:8px;" />',
            unsafe_allow_html=True,
        )
    with st.container(border=True):
        if config.PC_DETECTION:
            st.caption(
                f"Detection on PC: `{config.YOLO_MODEL_PATH.name}` @ {config.YOLO_IMGSZ}, "
                f"classes: {', '.join(config.DETECTION_CLASSES)}"
            )
        else:
            st.caption("Model info here")
    _render_audio(state)
