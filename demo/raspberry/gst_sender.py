from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

from demo import config


@dataclass
class PipelineStatus:
    running: bool
    error: str = ""


class GstPipeline:
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self._pipeline: Gst.Pipeline | None = None
        self._loop: GLib.MainLoop | None = None
        self._thread: threading.Thread | None = None
        self._error = ""
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        pipeline = self._pipeline
        if pipeline is None:
            return False
        state = pipeline.get_state(0).state
        return state in (Gst.State.PLAYING, Gst.State.PAUSED)

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    def start(self) -> None:
        if self.is_running:
            return
        with self._lock:
            self._error = ""
        pipeline = Gst.parse_launch(self.description)
        self._pipeline = pipeline
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_message)

        loop = GLib.MainLoop()
        self._loop = loop
        self._thread = threading.Thread(
            target=loop.run, name=f"gst-{self.name}", daemon=True
        )
        self._thread.start()

        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            with self._lock:
                self._error = f"{self.name} failed to start"
            self.stop()

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

    def status(self) -> PipelineStatus:
        return PipelineStatus(running=self.is_running, error=self.error)

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
            with self._lock:
                self._error = f"{self.name} received EOS"
            loop = self._loop
            if loop is not None and loop.is_running():
                loop.quit()


def _configure_camera() -> None:
    if not config.CAMERA_FORCE_FPS:
        return
    try:
        subprocess.run(
            [
                "v4l2-ctl",
                "-d",
                config.VIDEO_DEVICE,
                "-c",
                "exposure_dynamic_framerate=0",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return


def build_video_sender_pipeline(fake: bool = False) -> str:
    sink = (
        f"rtph264pay config-interval=1 pt=96 ! "
        f"udpsink host={config.PC_HOST} port={config.VIDEO_RTP_PORT} "
        f"sync=false async=false"
    )
    if fake:
        source = (
            f"videotestsrc is-live=true pattern=smpte ! "
            f"video/x-raw,width={config.VIDEO_WIDTH},height={config.VIDEO_HEIGHT},"
            f"framerate={config.VIDEO_FPS}/1,format=I420"
        )
    else:
        source = (
            f"v4l2src device={config.VIDEO_DEVICE} do-timestamp=true ! "
            f"image/jpeg,width={config.VIDEO_WIDTH},height={config.VIDEO_HEIGHT},"
            f"framerate={config.VIDEO_FPS}/1 ! jpegdec ! videoconvert ! "
            f"video/x-raw,format=I420"
        )
    encode = (
        f"x264enc tune=zerolatency speed-preset=ultrafast "
        f"bitrate={config.X264_BITRATE_KBPS} key-int-max={config.VIDEO_FPS} "
        f"bframes=0 byte-stream=true"
    )
    return f"{source} ! {encode} ! {sink}"


def build_audio_sender_pipeline(fake: bool = False) -> str:
    sink = (
        f"rtpopuspay pt=97 ! udpsink host={config.PC_HOST} "
        f"port={config.AUDIO_RTP_PORT} sync=false async=false"
    )
    if fake:
        source = (
            "audiotestsrc is-live=true wave=sine freq=440 ! "
            "audio/x-raw,rate=48000,channels=1"
        )
    elif config.AUDIO_INPUT_FORMAT == "pulse":
        source = f"pulsesrc device={config.AUDIO_DEVICE} do-timestamp=true"
    elif config.AUDIO_INPUT_FORMAT == "alsa":
        source = f"alsasrc device={config.AUDIO_DEVICE} do-timestamp=true"
    else:
        source = "autoaudiosrc do-timestamp=true"

    process = "audioconvert ! audioresample ! audio/x-raw,rate=48000,channels=1"
    encode = f"opusenc bitrate={config.OPUS_BITRATE} audio-type=voice"
    return f"{source} ! {process} ! {encode} ! {sink}"


class GstSender:
    def __init__(self, fake: bool = False) -> None:
        self.fake = fake
        self.video: GstPipeline | None = None
        self.audio: GstPipeline | None = None

    def start(self) -> bool:
        Gst.init(None)
        if not self.fake:
            _configure_camera()
        self.video = GstPipeline("video-sender", build_video_sender_pipeline(self.fake))
        self.audio = GstPipeline("audio-sender", build_audio_sender_pipeline(self.fake))
        self.video.start()
        self.audio.start()
        return self.video.is_running and self.audio.is_running

    def status(self) -> dict[str, object]:
        return {
            "video_running": bool(self.video and self.video.is_running),
            "audio_running": bool(self.audio and self.audio.is_running),
            "video_error": self.video.error if self.video else "",
            "audio_error": self.audio.error if self.audio else "",
        }

    def stop(self) -> None:
        if self.video is not None:
            self.video.stop()
        if self.audio is not None:
            self.audio.stop()
