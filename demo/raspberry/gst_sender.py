from __future__ import annotations

import queue
import subprocess
import threading
from dataclasses import dataclass

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

from demo import config
from demo.raspberry.detection_sender import DetectionSender


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

    def get_element(self, name: str) -> Gst.Element | None:
        if self._pipeline is None:
            return None
        return self._pipeline.get_by_name(name)

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
    return (
        f"{source} ! tee name=t "
        f"t. ! queue ! {encode} ! {sink} "
        f"t. ! queue leaky=downstream max-size-buffers=1 ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink name=yolo-appsink emit-signals=true max-buffers=1 drop=true sync=false"
    )


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
        self._detector = None
        self._detection_sender: DetectionSender | None = None
        self._detection_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._detection_thread: threading.Thread | None = None
        self._detection_stop = threading.Event()
        self._frame_count = 0

    def start(self) -> bool:
        Gst.init(None)
        if not self.fake:
            _configure_camera()
        self._start_detector()
        self.video = GstPipeline("video-sender", build_video_sender_pipeline(self.fake))
        self.audio = GstPipeline("audio-sender", build_audio_sender_pipeline(self.fake))
        self.video.start()
        self.audio.start()
        self._attach_yolo_sink()
        return self.video.is_running and self.audio.is_running

    def status(self) -> dict[str, object]:
        return {
            "video_running": bool(self.video and self.video.is_running),
            "audio_running": bool(self.audio and self.audio.is_running),
            "video_error": self.video.error if self.video else "",
            "audio_error": self.audio.error if self.audio else "",
            "yolo_enabled": self._detector is not None,
        }

    def stop(self) -> None:
        self._detection_stop.set()
        if self._detection_thread is not None and self._detection_thread.is_alive():
            self._detection_thread.join(timeout=2.0)
            self._detection_thread = None
        if self._detection_sender is not None:
            self._detection_sender.close()
            self._detection_sender = None
        if self.video is not None:
            self.video.stop()
        if self.audio is not None:
            self.audio.stop()

    def _start_detector(self) -> None:
        model_path = config.YOLO_MODEL_PATH
        if not model_path.exists():
            print(f"YOLO model not found: {model_path}", flush=True)
            return
        try:
            from demo.raspberry.yolo_detector import YoloDetector

            self._detector = YoloDetector(
                model_path,
                config.YOLO_CONFIDENCE,
                config.YOLO_IOU,
                config.YOLO_MAX_DET,
                config.YOLO_DEVICE,
            )
            self._detection_sender = DetectionSender(
                config.PC_HOST, config.DETECTION_PORT
            )
            self._detection_stop.clear()
            self._detection_thread = threading.Thread(
                target=self._detection_loop, name="yolo-worker", daemon=True
            )
            self._detection_thread.start()
            print(f"YOLO enabled: {model_path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"YOLO disabled: {exc}", flush=True)
            self._detector = None
            self._detection_sender = None

    def _attach_yolo_sink(self) -> None:
        if self.video is None or self._detector is None:
            return
        appsink = self.video.get_element("yolo-appsink")
        if appsink is None:
            return
        appsink.connect("new-sample", self._on_yolo_sample)

    def _on_yolo_sample(self, sink: Gst.Element) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        self._frame_count += 1
        if self._frame_count % config.YOLO_FRAME_INTERVAL != 0:
            return Gst.FlowReturn.OK
        buffer = sample.get_buffer()
        ok, map_info = buffer.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK
        try:
            frame = np.frombuffer(map_info.data, dtype=np.uint8).copy()
        finally:
            buffer.unmap(map_info)
        expected = config.VIDEO_WIDTH * config.VIDEO_HEIGHT * 3
        if frame.size != expected:
            return Gst.FlowReturn.OK
        frame = frame.reshape((config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 3))
        if self._detection_queue.full():
            try:
                self._detection_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._detection_queue.put_nowait(frame)
        except queue.Full:
            pass
        return Gst.FlowReturn.OK

    def _detection_loop(self) -> None:
        while not self._detection_stop.is_set():
            try:
                frame = self._detection_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._detector is None or self._detection_sender is None:
                continue
            try:
                detections = self._detector.infer(frame)
                self._detection_sender.send(
                    detections, (config.VIDEO_WIDTH, config.VIDEO_HEIGHT)
                )
            except Exception as exc:  # noqa: BLE001
                print(f"YOLO inference error: {exc}", flush=True)
