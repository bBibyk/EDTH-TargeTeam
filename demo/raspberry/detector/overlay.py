"""YOLO26n detection overlay for the live video stream.

`DetectionOverlay.process(frame)` is called for every camera frame (BGR numpy):
it hands the frame to a background YOLO thread (newest frame wins, older ones are
dropped) and draws the latest boxes on the frame in place. Video therefore keeps
the camera frame rate even when YOLO is slower; boxes refresh at YOLO's rate.

Detection only runs while `is_active()` is true (audio wake gate), so the Pi
does no inference while idle.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

BOX_COLOR = (0, 255, 0)  # BGR
IDLE_COLOR = (160, 160, 160)
MAX_BOX_AGE_SECONDS = 0.5  # do not draw boxes older than this


@dataclass
class Detection:
    label: str
    score: float
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionOverlay:
    def __init__(
        self,
        model_path: Path,
        classes: Sequence[str] = ("person",),
        imgsz: int = 640,
        conf: float = 0.25,
        is_active: Callable[[], bool] = lambda: True,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(str(model_path), task="detect")
        by_name = {name: idx for idx, name in self._model.names.items()}
        unknown = [c for c in classes if c not in by_name and not str(c).isdigit()]
        if unknown:
            raise ValueError(f"unknown classes {unknown}; available: {list(by_name)}")
        self._class_ids = [int(c) if str(c).isdigit() else by_name[c] for c in classes] or None
        self._imgsz = imgsz
        self._conf = conf
        self._is_active = is_active

        # First predict() is slow (lazy setup): do it now, not on the first wake.
        self._model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), imgsz=imgsz, verbose=False)

        self._cond = threading.Condition()
        self._pending: np.ndarray | None = None
        self._detections: list[Detection] = []
        self._detections_at = 0.0
        self._fps = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="yolo-overlay", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- called from the video thread ---------------------------------------
    def process(self, frame: np.ndarray) -> np.ndarray:
        active = self._is_active()
        if active:
            with self._cond:
                self._pending = frame.copy()
                self._cond.notify()
        self.draw(frame, active)
        return frame

    def draw(self, frame: np.ndarray, active: bool) -> None:
        with self._cond:
            detections = list(self._detections)
            fresh = time.monotonic() - self._detections_at < MAX_BOX_AGE_SECONDS
            fps = self._fps
        if active and fresh:
            for det in detections:
                cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), BOX_COLOR, 2)
                text = f"{det.label} {det.score:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ty = max(det.y1, th + 4)
                cv2.rectangle(frame, (det.x1, ty - th - 4), (det.x1 + tw + 4, ty), BOX_COLOR, -1)
                cv2.putText(frame, text, (det.x1 + 2, ty - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        status = f"DETECTING {fps:.0f} FPS" if active else "DETECTOR IDLE"
        (sw, sh), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (4, 4), (12 + sw, 12 + sh), (0, 0, 0), -1)
        cv2.putText(frame, status, (8, 8 + sh), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    BOX_COLOR if active else IDLE_COLOR, 1, cv2.LINE_AA)

    # -- YOLO thread ----------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            with self._cond:
                while self._pending is None and not self._stop.is_set():
                    self._cond.wait(timeout=0.5)
                frame, self._pending = self._pending, None
            if frame is None:
                continue
            started = time.monotonic()
            result = self._model.predict(
                frame, imgsz=self._imgsz, conf=self._conf,
                classes=self._class_ids, verbose=False,
            )[0]
            detections = []
            if result.boxes is not None:
                boxes = result.boxes
                for (x1, y1, x2, y2), cls, score in zip(
                    boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()
                ):
                    detections.append(Detection(result.names[int(cls)], float(score),
                                                int(x1), int(y1), int(x2), int(y2)))
            elapsed = max(time.monotonic() - started, 1e-6)
            with self._cond:
                self._detections = detections
                self._detections_at = time.monotonic()
                self._fps = 0.8 * self._fps + 0.2 / elapsed if self._fps else 1.0 / elapsed
