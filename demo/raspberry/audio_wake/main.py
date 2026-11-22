"""Audio wake program: microphone -> audio wake model -> WAKE signal to the detector.

Keeps a sliding window (the length the model is trained on: 5 s at 16 kHz for the
drone_audio_detector model), scores it every hop (default 0.5 s) and sends a UDP
wake datagram whenever the score crosses the threshold. Each datagram re-arms the
detector for WAKE_HOLD_SECONDS.

    python -m demo.raspberry.audio_wake.main --verbose            # live microphone
    python -m demo.raspberry.audio_wake.main --wav clip.wav       # replay a file (testing)
    python -m demo.raspberry.audio_wake.main --wake-classes drone,vehicle
"""
from __future__ import annotations

import argparse
import queue
import signal
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from demo import config
from demo.common.ring_buffer import AudioRingBuffer
from demo.common.wake import WakeSender
from demo.raspberry.audio_wake.model import load_audio_wake_model


def _read_with_ffmpeg(path: Path, target_rate: int) -> np.ndarray:
    """Any format ffmpeg knows (flac, mp3, m4a, 24-bit wav, video, ...) -> mono float32."""
    try:
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(target_rate),
             "-f", "s16le", "-"],
            capture_output=True, check=True,
        ).stdout
    except FileNotFoundError as exc:
        raise ValueError(f"{path}: not a 16-bit PCM wav, and ffmpeg is not installed") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"{path}: ffmpeg could not read it: {exc.stderr.decode().strip()}") from exc
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _read_wav(path: Path, target_rate: int) -> np.ndarray:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            raw = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError):
        return _read_with_ffmpeg(path, target_rate)  # e.g. a FLAC file named .wav
    if width != 2 or rate != target_rate:
        return _read_with_ffmpeg(path, target_rate)  # proper resampling
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return data.reshape(-1, channels).mean(axis=1)


def _wav_feeder(
    path: Path, rate: int, chunk: int, out: queue.Queue, stop: threading.Event, loop: bool,
    min_samples: int = 0,
) -> None:
    data = _read_wav(path, rate)
    if 0 < len(data) < min_samples:
        repeats = -(-min_samples // len(data))
        print(
            f"{path.name} is {len(data) / rate:.2f} s, shorter than the model window: "
            f"playing it {repeats} times in a row",
            flush=True,
        )
        data = np.tile(data, repeats)
    period = chunk / rate
    while not stop.is_set():
        for start in range(0, len(data), chunk):
            if stop.is_set():
                return
            out.put(data[start : start + chunk])
            time.sleep(period)  # real-time pacing
        if not loop:
            out.put(None)
            return


def main() -> int:
    parser = argparse.ArgumentParser(description="Audio wake -> detector wake signal")
    parser.add_argument("--model", type=Path, default=config.AUDIO_WAKE_MODEL_PATH)
    parser.add_argument(
        "--wav", type=Path,
        help="Replay an audio file instead of the microphone (wav; flac/mp3/... via ffmpeg)",
    )
    parser.add_argument("--loop", action="store_true", help="Loop the --wav file")
    parser.add_argument("--device", default=config.AUDIO_WAKE_DEVICE, help="sounddevice input")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--wake-classes", default=",".join(config.AUDIO_WAKE_CLASSES),
        help="Comma-separated classes whose probabilities form the wake score",
    )
    parser.add_argument(
        "--window", type=float, default=config.AUDIO_WAKE_WINDOW_SECONDS,
        help="Window length in s (ignored when the model has a fixed window)",
    )
    parser.add_argument("--hop", type=float, default=config.AUDIO_WAKE_HOP_SECONDS)
    parser.add_argument(
        "--min-consecutive", type=int, default=1,
        help="Positive windows in a row required before sending a wake",
    )
    parser.add_argument("--host", default=config.WAKE_HOST)
    parser.add_argument("--port", type=int, default=config.WAKE_PORT)
    parser.add_argument("--verbose", action="store_true", help="Print every score")
    args = parser.parse_args()

    wake_classes = tuple(c.strip() for c in args.wake_classes.split(",") if c.strip())
    model = load_audio_wake_model(
        args.model, threads=config.AUDIO_WAKE_THREADS, wake_classes=wake_classes
    )
    rate = model.sample_rate
    window_seconds = getattr(model, "window_seconds", None) or args.window
    if args.threshold is not None:
        threshold = args.threshold
    elif config.AUDIO_WAKE_THRESHOLD:
        threshold = float(config.AUDIO_WAKE_THRESHOLD)
    else:
        threshold = model.threshold

    window = int(window_seconds * rate)
    hop = int(args.hop * rate)
    ring = AudioRingBuffer(window, dtype=np.float32)
    chunks: queue.Queue = queue.Queue()
    stop = threading.Event()
    sender = WakeSender(args.host, args.port)

    def _handle_signal(signum: int, frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    stream = None
    if args.wav is not None:
        threading.Thread(
            target=_wav_feeder,
            args=(args.wav, rate, hop, chunks, stop, args.loop, window + 4 * hop),
            daemon=True,
        ).start()
        source = str(args.wav)
    elif args.device and str(args.device).startswith("stm32"):
        # STM32 board over the ST-LINK USB serial port: "stm32" or "stm32:/dev/ttyACM1"
        from demo.raspberry.audio_wake.stm32_mic import Stm32Microphone

        port = str(args.device).partition(":")[2] or config.STM32_MIC_PORT
        target_rate = rate

        def _push_stm32(block: np.ndarray) -> None:
            data = np.asarray(block, dtype=np.float32).reshape(-1)
            if Stm32Microphone.sample_rate != target_rate:
                x_old = np.arange(data.size, dtype=np.float32)
                x_new = np.arange(
                    0, data.size, Stm32Microphone.sample_rate / target_rate, dtype=np.float32
                )
                data = np.interp(x_new, x_old, data).astype(np.float32)
            chunks.put(data)

        stream = Stm32Microphone(_push_stm32, port=port, baud=config.STM32_MIC_BAUD)
        stream.start()
        source = f"STM32 microphone ({stream.port})"
    else:
        import sounddevice as sd

        def _callback(indata, frames, time_info, status) -> None:
            if status:
                print(f"audio status: {status}", flush=True)
            chunks.put(indata[:, 0].copy())

        stream = sd.InputStream(
            samplerate=rate, channels=1, dtype="float32",
            blocksize=hop, device=args.device, callback=_callback,
        )
        stream.start()
        source = f"microphone ({args.device or 'default'})"

    print(
        f"Audio wake: {source} @ {rate} Hz, model={model.name}, thr={threshold:.3f}, "
        f"window={window_seconds}s hop={args.hop}s -> udp://{args.host}:{args.port}",
        flush=True,
    )

    pending = 0
    streak = 0
    try:
        while not stop.is_set():
            try:
                chunk = chunks.get(timeout=0.5)
            except queue.Empty:
                continue
            if chunk is None:
                break
            ring.write(chunk)
            pending += len(chunk)
            if len(ring) < window or pending < hop:
                continue
            pending = 0

            started = time.perf_counter()
            score = model.predict(ring.snapshot())
            latency_ms = (time.perf_counter() - started) * 1000.0
            streak = streak + 1 if score >= threshold else 0
            if args.verbose:
                probs = getattr(model, "last_probabilities", None)
                detail = "  " + " ".join(f"{k}={v:.2f}" for k, v in probs.items()) if probs else ""
                print(f"score={score:.3f} ({latency_ms:.0f} ms){detail}", flush=True)
            if streak >= args.min_consecutive:
                sender.send(score, model=model.name)
                print(f"WAKE sent (score={score:.3f}, {latency_ms:.0f} ms)", flush=True)
    finally:
        stop.set()
        if stream is not None:
            stream.stop()
            if hasattr(stream, "close"):
                stream.close()
        sender.close()
        print("Audio wake stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
