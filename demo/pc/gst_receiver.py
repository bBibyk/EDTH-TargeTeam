from __future__ import annotations

import threading
from collections.abc import Callable

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

from demo import config


def _sample_to_bytes(sample: Gst.Sample) -> bytes:
    buffer = sample.get_buffer()
    ok, map_info = buffer.map(Gst.MapFlags.READ)
    if not ok:
        return b""
    try:
        return bytes(map_info.data)
    finally:
        buffer.unmap(map_info)


class GstReceiverPipeline:
    def __init__(
        self,
        name: str,
        description: str,
        on_sample: Callable[[Gst.Sample], None],
    ) -> None:
        self.name = name
        self.description = description
        self._on_sample = on_sample
        self._pipeline: Gst.Pipeline | None = None
        self._loop: GLib.MainLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._error = ""

    @property
    def is_running(self) -> bool:
        pipeline = self._pipeline
        if pipeline is None:
            return False
        return pipeline.get_state(0).state in (Gst.State.PLAYING, Gst.State.PAUSED)

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    def start(self) -> None:
        if self.is_running:
            return
        Gst.init(None)
        with self._lock:
            self._error = ""
        pipeline = Gst.parse_launch(self.description)
        self._pipeline = pipeline

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_message)

        appsink = pipeline.get_by_name(f"{self.name}-appsink")
        if appsink is not None:
            appsink.connect("new-sample", self._on_new_sample)

        loop = GLib.MainLoop()
        self._loop = loop
        self._thread = threading.Thread(
            target=loop.run, name=f"gst-{self.name}", daemon=True
        )
        self._thread.start()
        pipeline.set_state(Gst.State.PLAYING)

    def stop(self) -> None:
        pipeline = self._pipeline
        loop = self._loop
        thread = self._thread
        self._pipeline = None
        self._loop = None
        self._thread = None

        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
        if loop is not None and loop.is_running():
            loop.quit()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def _on_new_sample(self, sink: Gst.Element) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if sample is not None:
            try:
                self._on_sample(sample)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._error = str(exc)
        return Gst.FlowReturn.OK

    def _on_message(self, _bus: Gst.Bus, msg: Gst.Message) -> None:
        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            detail = err.message
            if debug:
                detail = f"{detail} ({debug})"
            with self._lock:
                self._error = detail
            loop = self._loop
            if loop is not None and loop.is_running():
                loop.quit()
        elif msg.type == Gst.MessageType.EOS:
            loop = self._loop
            if loop is not None and loop.is_running():
                loop.quit()


def build_video_receiver_pipeline() -> str:
    return (
        f"udpsrc port={config.VIDEO_RTP_PORT} "
        f'caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000" ! '
        f"rtpjitterbuffer latency={config.GST_JITTERBUFFER_MS} drop-on-latency=true ! "
        f"rtph264depay ! h264parse ! avdec_h264 max-threads=1 ! "
        f"videoconvert ! jpegenc quality={config.JPEG_QUALITY} ! "
        f"appsink name=video-appsink emit-signals=true max-buffers=1 drop=true sync=false"
    )


def build_audio_receiver_pipeline() -> str:
    return (
        f"udpsrc port={config.AUDIO_RTP_PORT} "
        f'caps="application/x-rtp,media=audio,encoding-name=OPUS,payload=97,clock-rate=48000" ! '
        f"rtpjitterbuffer latency={config.GST_JITTERBUFFER_MS} ! "
        f"rtpopusdepay ! opusdec ! audioconvert ! audioresample ! "
        f"audio/x-raw,format=S16LE,rate={config.AUDIO_RATE},channels=1 ! "
        f"appsink name=audio-appsink emit-signals=true max-buffers=50 drop=true sync=false"
    )


class GstVideoReceiver(GstReceiverPipeline):
    def __init__(self, on_frame: Callable[[bytes], None]) -> None:
        super().__init__("video", build_video_receiver_pipeline(), self._handle_sample)
        self._on_frame = on_frame

    def _handle_sample(self, sample: Gst.Sample) -> None:
        data = _sample_to_bytes(sample)
        if data:
            self._on_frame(data)


class GstAudioReceiver(GstReceiverPipeline):
    def __init__(self, on_audio: Callable[[np.ndarray], None]) -> None:
        super().__init__("audio", build_audio_receiver_pipeline(), self._handle_sample)
        self._on_audio = on_audio

    def _handle_sample(self, sample: Gst.Sample) -> None:
        data = _sample_to_bytes(sample)
        if not data:
            return
        samples = np.frombuffer(data, dtype=np.int16).copy()
        self._on_audio(samples)
