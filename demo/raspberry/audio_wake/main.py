"""Audio wake program: microphone -> audio wake model -> WAKE signal to the detector.

Keeps a sliding window (default 5 s, the length the model is trained on), scores it
every hop (default 0.5 s) and sends a UDP wake datagram whenever the score crosses
the threshold. Each datagram re-arms the detector for WAKE_HOLD_SECONDS.

    python -m demo.raspberry.audio_wake.main                 # live microphone
    python -m demo.raspberry.audio_wake.main --wav clip.wav  # replay a file (testing)
"""
from __future__ import annotations

import argparse
import queue
import signal
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


def _read_wav(path: Path, target_rate: int) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        raw = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM wav is supported")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    data = data.reshape(-1, channels).mean(axis=1)
    if rate != target_rate:
        positions = np.arange(0, len(data), rate / target_rate)
        data = np.interp(positions, np.arange(len(data)), data).astype(np.float32)
    return data


def _wav_feeder(
    path: Path, rate: int, chunk: int, out: queue.Queue, stop: threading.Event, loop: bool
) -> None:
    data = _read_wav(path, rate)
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
    parser.add_argument("--model", type=Path, default=config.AUDIO_MODEL_PATH)
    parser.add_argument("--wav", type=Path, help="Replay a wav file instead of the microphone")
    parser.add_argument("--loop", action="store_true", help="Loop the --wav file")
    parser.add_argument("--device", default=config.AUDIO_WAKE_DEVICE, help="sounddevice input")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--window", type=float, default=config.AUDIO_WAKE_WINDOW_SECONDS)
    parser.add_argument("--hop", type=float, default=config.AUDIO_WAKE_HOP_SECONDS)
    parser.add_argument(
        "--min-consecutive", type=int, default=1,
        help="Positive windows in a row required before sending a wake",
    )
    parser.add_argument("--host", default=config.WAKE_HOST)
    parser.add_argument("--port", type=int, default=config.WAKE_PORT)
    parser.add_argument("--verbose", action="store_true", help="Print every score")
    args = parser.parse_args()

    model = load_audio_wake_model(args.model, threads=config.AUDIO_WAKE_THREADS)
    rate = model.sample_rate
    if args.threshold is not None:
        threshold = args.threshold
    elif config.AUDIO_WAKE_THRESHOLD:
        threshold = float(config.AUDIO_WAKE_THRESHOLD)
    else:
        threshold = model.threshold

    window = int(args.window * rate)
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
            args=(args.wav, rate, hop, chunks, stop, args.loop),
            daemon=True,
        ).start()
        source = str(args.wav)
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
        f"window={args.window}s hop={args.hop}s -> udp://{args.host}:{args.port}",
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
                print(f"score={score:.3f} ({latency_ms:.0f} ms)", flush=True)
            if streak >= args.min_consecutive:
                sender.send(score, model=model.name)
                print(f"WAKE sent (score={score:.3f}, {latency_ms:.0f} ms)", flush=True)
    finally:
        stop.set()
        if stream is not None:
            stream.stop()
            stream.close()
        sender.close()
        print("Audio wake stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
