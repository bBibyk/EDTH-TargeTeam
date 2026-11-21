from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from demo import config


class _RtpProcessReceiver:
    def __init__(
        self,
        name: str,
        sdp_path: Path,
        log_path: Path,
        args_builder: Callable[[Path], list[str]],
        on_data: Callable[[np.ndarray], None],
    ) -> None:
        self.name = name
        self._sdp_path = sdp_path
        self._log_path = log_path
        self._args_builder = args_builder
        self._on_data = on_data
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._log_file = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"rtp-{self.name}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._terminate_process()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if not self._sdp_ready():
                self._stop_event.wait(0.2)
                continue

            self._start_process()
            try:
                self._consume()
            finally:
                self._terminate_process()

            if not self._stop_event.is_set():
                self._stop_event.wait(0.5)

    def _sdp_ready(self) -> bool:
        try:
            return self._sdp_path.exists() and self._sdp_path.stat().st_size > 0
        except OSError:
            return False

    def _start_process(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_path.open("ab")
        args = self._args_builder(self._sdp_path)
        with self._lock:
            self._process = subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=self._log_file,
            )

    def _terminate_process(self) -> None:
        with self._lock:
            process = self._process
            self._process = None

        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
            if process.stdout is not None:
                process.stdout.close()

        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _consume(self) -> None:
        raise NotImplementedError


class RtpVideoReceiver(_RtpProcessReceiver):
    def __init__(self, on_frame: Callable[[np.ndarray], None]) -> None:
        super().__init__(
            "video",
            config.VIDEO_SDP_PATH,
            config.VIDEO_LOG_PATH,
            build_video_receiver_args,
            on_frame,
        )

    def _consume(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return

        frame_bytes = config.VIDEO_FRAME_BYTES
        while not self._stop_event.is_set() and process.poll() is None:
            data = process.stdout.read(frame_bytes)
            if len(data) < frame_bytes:
                break
            frame = np.frombuffer(data, dtype=np.uint8).reshape(
                (config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 3)
            )
            self._on_data(frame)


class RtpAudioReceiver(_RtpProcessReceiver):
    def __init__(self, on_audio: Callable[[np.ndarray], None]) -> None:
        super().__init__(
            "audio",
            config.AUDIO_SDP_PATH,
            config.AUDIO_LOG_PATH,
            build_audio_receiver_args,
            on_audio,
        )

    def _consume(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return

        chunk_bytes = config.AUDIO_CHUNK_BYTES
        while not self._stop_event.is_set() and process.poll() is None:
            data = process.stdout.read(chunk_bytes)
            if len(data) < chunk_bytes:
                break
            samples = np.frombuffer(data, dtype=np.int16).copy()
            self._on_data(samples)


def build_video_receiver_args(sdp_path: Path) -> list[str]:
    return [
        config.FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-protocol_whitelist",
        "file,udp,rtp",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-max_delay",
        "0",
        "-analyzeduration",
        "0",
        "-i",
        str(sdp_path),
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]


def build_audio_receiver_args(sdp_path: Path) -> list[str]:
    return [
        config.FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-protocol_whitelist",
        "file,udp,rtp",
        "-fflags",
        "nobuffer",
        "-max_delay",
        "0",
        "-analyzeduration",
        "0",
        "-i",
        str(sdp_path),
        "-vn",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(config.AUDIO_RATE),
        "-",
    ]
