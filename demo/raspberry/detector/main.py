"""Visual detector (YOLO26n), gated by the audio wake signal.

Sleeps (camera kept open, frames discarded, no inference) until a WAKE datagram
arrives, then runs YOLO26n on every frame for WAKE_HOLD_SECONDS after the last wake.

    python -m demo.raspberry.detector.main                      # camera from config
    python -m demo.raspberry.detector.main --source clip.mp4    # video file
    python -m demo.raspberry.detector.main --always-on --show   # no gating, preview window

Default model is the NCNN export at imgsz 320 (fastest on the Pi 5 CPU). Re-export with:
    yolo export model=models/vision/yolo26n.pt format=ncnn imgsz=320
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from demo import config
from demo.common.wake import WakeGate, WakeListener


def _parse_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def _is_live(source: int | str) -> bool:
    return isinstance(source, int) or str(source).startswith(("/dev/", "rtsp://", "udp://"))


def on_detections(result: Any) -> None:
    """Hook: what to do with a frame's detections (currently: log them)."""
    if result.boxes is None or len(result.boxes) == 0:
        return
    names = result.names
    parts = [
        f"{names[int(c)]} {float(p):.2f}"
        for c, p in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())
    ]
    print(f"detections: {', '.join(parts)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO26n detector woken by the audio model")
    parser.add_argument("--model", type=Path, default=config.YOLO_MODEL_PATH)
    parser.add_argument("--source", default=config.DETECTOR_SOURCE)
    parser.add_argument("--imgsz", type=int, default=config.YOLO_IMGSZ)
    parser.add_argument("--conf", type=float, default=config.YOLO_CONF)
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="Only keep these classes (names or COCO ids), e.g. --classes person",
    )
    parser.add_argument("--hold", type=float, default=config.WAKE_HOLD_SECONDS)
    parser.add_argument("--host", default=config.WAKE_BIND_HOST)
    parser.add_argument("--port", type=int, default=config.WAKE_PORT)
    parser.add_argument("--always-on", action="store_true", help="Ignore wake gating")
    parser.add_argument("--show", action="store_true", help="Show annotated frames")
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    import numpy as np

    model = YOLO(str(args.model), task="detect")
    class_ids = None
    if args.classes:
        by_name = {name: idx for idx, name in model.names.items()}
        try:
            class_ids = [int(c) if c.isdigit() else by_name[c] for c in args.classes]
        except KeyError as exc:
            print(f"Unknown class {exc}; available: {', '.join(model.names.values())}", flush=True)
            return 1
    # The first predict() is slow (lazy setup); do it now, not on the first wake.
    model.predict(np.zeros((args.imgsz, args.imgsz, 3), np.uint8), imgsz=args.imgsz, verbose=False)
    source = _parse_source(str(args.source))
    live = _is_live(source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"Cannot open video source {source!r}", flush=True)
        return 1

    gate = WakeGate(args.hold)
    listener = WakeListener(args.host, args.port, gate.wake)
    listener.start()

    stop = threading.Event()

    def _handle_signal(signum: int, frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(
        f"Detector: {args.model.name} on {source!r}, listening for wake on "
        f"udp://{args.host}:{args.port}{' (always on)' if args.always_on else ''}",
        flush=True,
    )

    was_active = False
    frames = 0
    started = 0.0
    try:
        while not stop.is_set():
            active = args.always_on or gate.awake
            if active and not was_active:
                frames, started = 0, time.monotonic()
            elif was_active and not active:
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"Detector idle: {frames} frames in {elapsed:.1f}s "
                    f"({frames / elapsed:.1f} FPS) during last wake.",
                    flush=True,
                )
            was_active = active

            if not active:
                if live:
                    capture.grab()  # keep the camera buffer fresh, skip decoding
                else:
                    time.sleep(0.05)  # do not consume a file while idle
                continue

            ok, frame = capture.read()
            if not ok:
                print("End of video source.", flush=True)
                break
            result = model.predict(
                frame, imgsz=args.imgsz, conf=args.conf, classes=class_ids, verbose=False
            )[0]
            frames += 1
            on_detections(result)
            if args.show:
                cv2.imshow("detector", result.plot())
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        listener.stop()
        capture.release()
        if args.show:
            cv2.destroyAllWindows()
        print("Detector stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
