from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from demo import config
from demo.common.ring_buffer import AudioRingBuffer


class AppState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._latest_frame: bytes | None = None
        self._last_frame_time = 0.0
        self._last_audio_time = 0.0
        self._audio = AudioRingBuffer(config.AUDIO_BUFFER_SAMPLES)
        self._detections: list[dict[str, Any]] = []
        self._detection_frame_size: tuple[int, int] = (0, 0)
        self._last_detection_time = 0.0
        self.last_command = ""
        self.last_response: dict[str, Any] | None = None
        self.last_error = ""

    def set_frame(self, frame: bytes) -> None:
        with self._lock:
            self._latest_frame = frame
            self._last_frame_time = time.monotonic()

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._latest_frame

    def append_audio(self, samples: np.ndarray) -> None:
        self._audio.write(samples)
        with self._lock:
            self._last_audio_time = time.monotonic()

    def get_audio(self, seconds: float) -> np.ndarray:
        sample_count = int(seconds * config.AUDIO_RATE)
        return self._audio.snapshot(sample_count)

    def set_detections(
        self, detections: list[dict[str, Any]], frame_size: tuple[int, int]
    ) -> None:
        with self._lock:
            self._detections = detections
            self._detection_frame_size = frame_size
            self._last_detection_time = time.monotonic()

    def get_detections(self) -> tuple[list[dict[str, Any]], tuple[int, int]]:
        with self._lock:
            if time.monotonic() - self._last_detection_time > 2.0:
                return [], (0, 0)
            return list(self._detections), self._detection_frame_size

    def video_is_live(self) -> bool:
        with self._lock:
            return (
                self._latest_frame is not None
                and time.monotonic() - self._last_frame_time
                <= config.STREAM_STALE_SECONDS
            )

    def audio_is_live(self) -> bool:
        with self._lock:
            return (
                time.monotonic() - self._last_audio_time <= config.STREAM_STALE_SECONDS
            )

    def set_command_result(
        self,
        command: str,
        response: dict[str, Any] | None = None,
        error: str = "",
    ) -> None:
        with self._lock:
            self.last_command = command
            self.last_response = response
            self.last_error = error

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "video_live": self.video_is_live(),
                "audio_live": self.audio_is_live(),
                "last_command": self.last_command,
                "last_response": self.last_response,
                "last_error": self.last_error,
            }
