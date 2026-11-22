from __future__ import annotations

import queue
import subprocess
import threading
import time
from dataclasses import dataclass

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

from demo import config
from demo.raspberry.audio_result_sender import AudioResultSender
from demo.raspberry.detection_sender import DetectionSender

STM32_APPSRC_NAME = "stm32-audio-src"
STM32_RATE = 32000
VIDEO_RETRY_SECONDS = 5.0  # after the camera fails to start, wait this long before trying again


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
        source = "audiotestsrc is-live=true wave=sine freq=440"
    elif config.AUDIO_INPUT_FORMAT == "pulse":
        source = f"pulsesrc device={config.AUDIO_DEVICE} do-timestamp=true"
    elif config.AUDIO_INPUT_FORMAT == "alsa":
        source = f"alsasrc device={config.AUDIO_DEVICE} do-timestamp=true"
    elif config.AUDIO_INPUT_FORMAT == "stm32":
        # Fed by GstSender._push_stm32_audio from the STM32 board (stm32/mic_stream firmware).
        source = (
            f"appsrc name={STM32_APPSRC_NAME} is-live=true format=time do-timestamp=false "
            f'caps="audio/x-raw,format=S16LE,rate={STM32_RATE},channels=1,layout=interleaved"'
        )
    else:
        source = "autoaudiosrc do-timestamp=true"

    process = "audioconvert ! audioresample ! audio/x-raw,rate=48000,channels=1"
    encode = f"opusenc bitrate={config.OPUS_BITRATE} audio-type=voice"
    model_tap = (
        "queue leaky=downstream max-size-buffers=200 ! "
        "audioconvert ! audioresample ! "
        "audio/x-raw,format=S16LE,rate=16000,channels=1 ! "
        "appsink name=audio-model-appsink emit-signals=true max-buffers=200 drop=true sync=false"
    )
    return (
        f"{source} ! tee name=ta "
        f"ta. ! queue ! {process} ! {encode} ! {sink} "
        f"ta. ! {model_tap}"
    )


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
        self._audio_model = None
        self._audio_result_sender: AudioResultSender | None = None
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._audio_thread: threading.Thread | None = None
        self._audio_stop = threading.Event()
        self._video_active = threading.Event()
        self._last_vehicle_time = 0.0
        self._wake_lock = threading.Lock()
        self._video_failed_at = -VIDEO_RETRY_SECONDS
        self._stm32_mic = None
        self._stm32_appsrc: Gst.Element | None = None
        self._stm32_pts: int | None = None

    def start(self) -> bool:
        Gst.init(None)
        if not self.fake:
            _configure_camera()
        self._start_detector()
        self._start_audio_model()
        self.audio = GstPipeline("audio-sender", build_audio_sender_pipeline(self.fake))
        self.audio.start()
        self._attach_audio_model_sink()
        self._start_stm32_mic()
        return self.audio.is_running

    def status(self) -> dict[str, object]:
        return {
            "video_running": bool(self.video and self.video.is_running),
            "audio_running": bool(self.audio and self.audio.is_running),
            "video_error": self.video.error if self.video else "",
            "audio_error": self.audio.error if self.audio else "",
            "yolo_enabled": self._detector is not None,
            "audio_model_enabled": self._audio_model is not None,
            "video_active": self._video_active.is_set(),
            "audio_source": "audiotestsrc" if self.fake else config.AUDIO_INPUT_FORMAT,
            "stm32_connected": bool(self._stm32_mic and self._stm32_mic.connected),
        }

    def stop(self) -> None:
        self._detection_stop.set()
        self._audio_stop.set()
        if self._stm32_mic is not None:
            self._stm32_mic.stop()
            self._stm32_mic = None
        self._stm32_appsrc = None
        if self._detection_thread is not None and self._detection_thread.is_alive():
            self._detection_thread.join(timeout=2.0)
            self._detection_thread = None
        if self._audio_thread is not None and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=2.0)
            self._audio_thread = None
        if self._detection_sender is not None:
            self._detection_sender.close()
            self._detection_sender = None
        if self._audio_result_sender is not None:
            self._audio_result_sender.close()
            self._audio_result_sender = None
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
        if not self._video_active.is_set():
            return Gst.FlowReturn.OK
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

    def _start_audio_model(self) -> None:
        self._audio_result_sender = AudioResultSender(
            config.PC_HOST, config.AUDIO_RESULT_PORT
        )
        model_path = config.AUDIO_EVENT_MODEL_PATH
        mel_path = config.AUDIO_EVENT_MEL_BASIS_PATH
        if not model_path.exists():
            print(f"Audio model not found: {model_path}", flush=True)
        else:
            try:
                from demo.raspberry.audio_model import DroneAudioModel

                self._audio_model = DroneAudioModel(
                    model_path,
                    mel_path,
                    threads=config.AUDIO_WAKE_THREADS,
                )
                print(f"Audio model enabled: {model_path}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"Audio model disabled: {exc}", flush=True)
                self._audio_model = None
        self._audio_stop.clear()
        self._audio_thread = threading.Thread(
            target=self._audio_model_loop, name="audio-model-worker", daemon=True
        )
        self._audio_thread.start()

    def _start_video_if_needed(self) -> None:
        with self._wake_lock:
            now = time.monotonic()
            if self._video_active.is_set():
                self._last_vehicle_time = now
                return
            if now - self._video_failed_at < VIDEO_RETRY_SECONDS:
                return  # the camera just failed: don't retry on every audio window
            if not self.fake:
                _configure_camera()
            video = GstPipeline("video-sender", build_video_sender_pipeline(self.fake))
            video.start()
            if not video.is_running:
                self._video_failed_at = now
                print(
                    f"Video could not start ({video.error or 'camera missing?'}); "
                    f"audio keeps running, retry in {VIDEO_RETRY_SECONDS:.0f} s",
                    flush=True,
                )
                video.stop()
                return
            self.video = video  # only publish a running pipeline (main.py watches it)
            self._attach_yolo_sink()
            self._video_active.set()
            self._last_vehicle_time = now
            print("Video activated by audio model", flush=True)

    def handle_video_failure(self) -> None:
        """Video died (camera unplugged, ...): drop it, keep audio running, retry later."""
        with self._wake_lock:
            video, self.video = self.video, None
            self._video_active.clear()
            self._video_failed_at = time.monotonic()
        if video is not None:
            error = video.error
            video.stop()
            print(f"Video stopped: {error or 'pipeline not running'}", flush=True)

    def _maybe_stop_video(self) -> None:
        with self._wake_lock:
            if not self._video_active.is_set():
                return
            if time.monotonic() - self._last_vehicle_time < config.VIDEO_WAKE_HOLD_SECONDS:
                return
            self._video_active.clear()
            video, self.video = self.video, None  # unpublish first, so main.py doesn't see a dying pipeline
            if video is not None:
                video.stop()
            print("Video deactivated (no vehicle)", flush=True)

    def _start_stm32_mic(self) -> None:
        if self.fake or config.AUDIO_INPUT_FORMAT != "stm32" or self.audio is None:
            return
        appsrc = self.audio.get_element(STM32_APPSRC_NAME)
        if appsrc is None:
            print("STM32 audio source is missing from the audio pipeline", flush=True)
            return
        from demo.raspberry.audio_wake.stm32_mic import Stm32Microphone

        self._stm32_appsrc = appsrc
        self._stm32_pts = None
        self._stm32_mic = Stm32Microphone(
            self._push_stm32_audio,
            port=config.STM32_MIC_PORT,
            baud=config.STM32_MIC_BAUD,
            as_int16=True,
            reconnect=True,  # keeps retrying if the board is unplugged or being reflashed
        )
        self._stm32_mic.start()

    def _push_stm32_audio(self, block: np.ndarray) -> None:
        appsrc = self._stm32_appsrc
        if appsrc is None or block.size == 0:
            return
        if self._stm32_pts is None:  # start at the pipeline's current running time
            clock = appsrc.get_clock()
            self._stm32_pts = max(0, clock.get_time() - appsrc.get_base_time()) if clock else 0
        buffer = Gst.Buffer.new_wrapped(block.astype("<i2").tobytes())
        buffer.pts = self._stm32_pts
        buffer.duration = block.size * Gst.SECOND // STM32_RATE
        self._stm32_pts += buffer.duration
        appsrc.emit("push-buffer", buffer)

    def _attach_audio_model_sink(self) -> None:
        if self.audio is None or self._audio_model is None:
            return
        appsink = self.audio.get_element("audio-model-appsink")
        if appsink is None:
            return
        appsink.connect("new-sample", self._on_audio_model_sample)

    def _on_audio_model_sample(self, sink: Gst.Element) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        buffer = sample.get_buffer()
        ok, map_info = buffer.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK
        try:
            data = np.frombuffer(map_info.data, dtype=np.int16).copy()
        finally:
            buffer.unmap(map_info)
        if data.size == 0:
            return Gst.FlowReturn.OK
        try:
            self._audio_queue.put_nowait(data)
        except queue.Full:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._audio_queue.put_nowait(data)
            except queue.Full:
                pass
        return Gst.FlowReturn.OK

    def _audio_model_loop(self) -> None:
        model_sr = int(getattr(self._audio_model, "sample_rate", 16000))
        window_samples = int(getattr(self._audio_model, "window_samples", model_sr * 5))
        window = np.zeros(window_samples, dtype=np.float32)
        filled = 0
        since_infer = 0
        infer_every = max(1, int(config.AUDIO_WAKE_HOP_SECONDS * model_sr))
        rms_threshold = config.AUDIO_RMS_WAKE_THRESHOLD
        wake_classes = tuple(getattr(config, "AUDIO_WAKE_CLASSES", ("drone",)))
        wake_threshold = None
        if config.AUDIO_WAKE_THRESHOLD:
            wake_threshold = float(config.AUDIO_WAKE_THRESHOLD)
        elif self._audio_model is not None:
            wake_threshold = float(getattr(self._audio_model, "threshold", 0.5))
        while not self._audio_stop.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                self._maybe_stop_video()
                continue
            samples = chunk.astype(np.float32) / 32768.0
            n = samples.size
            if n >= window.size:
                window[:] = samples[-window.size :]
                filled = window.size
            else:
                window = np.roll(window, -n)
                window[-n:] = samples
                filled = min(window.size, filled + n)
            since_infer += n
            if filled < window.size or since_infer < infer_every:
                continue
            since_infer = 0
            rms = float(np.sqrt(np.mean(samples ** 2)))
            if rms < rms_threshold:
                self._maybe_stop_video()
                if self._audio_result_sender is not None:
                    self._audio_result_sender.send(
                        {"class": "background", "wake": False, "rms": rms, "probabilities": {}}
                    )
                continue
            if self._audio_result_sender is None:
                continue
            if self._audio_model is None:
                self._audio_result_sender.send(
                    {"class": "model_disabled", "wake": True, "rms": rms, "probabilities": {}}
                )
                continue
            try:
                result = self._audio_model.predict(window.copy())
                result["rms"] = rms
                probs = result.get("probabilities", {})
                wake_score = float(sum(float(probs.get(c, 0.0)) for c in wake_classes))
                result["wake_score"] = wake_score
                result["wake_classes"] = list(wake_classes)
                if wake_threshold is not None:
                    result["wake_threshold"] = wake_threshold
                wake_triggered = wake_threshold is None or wake_score >= wake_threshold
                result["wake"] = wake_triggered
                if wake_triggered:
                    self._start_video_if_needed()
                else:
                    self._maybe_stop_video()
                self._audio_result_sender.send(result)
            except Exception as exc:  # noqa: BLE001
                print(f"Audio model error: {exc}", flush=True)
