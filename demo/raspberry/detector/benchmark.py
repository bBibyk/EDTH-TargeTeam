"""Compare YOLO26n inference speed across export formats on this machine.

    python -m demo.raspberry.detector.benchmark                 # all formats found
    python -m demo.raspberry.detector.benchmark --imgsz 320 --runs 100

Run it on the Pi 5 itself: format rankings on x86 do not carry over to ARM.
Times are end-to-end predict() calls (preprocess + inference + postprocess).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from demo import config


def main() -> int:
    vision = config.MODELS_DIR / "vision"
    parser = argparse.ArgumentParser(description="Benchmark YOLO26n export formats")
    parser.add_argument(
        "--models", nargs="+", type=Path,
        default=[vision / "yolo26n.pt", vision / "yolo26n.onnx", vision / "yolo26n_ncnn_model"],
    )
    parser.add_argument("--imgsz", type=int, default=config.YOLO_IMGSZ)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    from ultralytics import YOLO

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 3), np.uint8)

    print(f"imgsz={args.imgsz}, frame {config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT}, {args.runs} runs")
    for path in args.models:
        if not path.exists():
            print(f"{path.name:24s} missing, skipped")
            continue
        model = YOLO(str(path), task="detect")
        for _ in range(args.warmup):
            model.predict(frame, imgsz=args.imgsz, verbose=False)
        times = []
        for _ in range(args.runs):
            started = time.perf_counter()
            model.predict(frame, imgsz=args.imgsz, verbose=False)
            times.append((time.perf_counter() - started) * 1000.0)
        median = statistics.median(times)
        print(f"{path.name:24s} median {median:6.1f} ms  ({1000.0 / median:5.1f} FPS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
