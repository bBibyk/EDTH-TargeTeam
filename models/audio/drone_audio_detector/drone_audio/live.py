"""Real-time drone / vehicle / other monitor.

Listens to the microphone (or replays a wav at real-time speed), keeps a sliding 5 s window,
scores it every hop and shows the live probability of each class plus the drone verdict.

    python -m drone_audio.live                    # default microphone
    python -m drone_audio.live --list-devices
    python -m drone_audio.live --device 2 --hop 0.25
    python -m drone_audio.live --wav clip.wav --loop
    python -m drone_audio.live --plain --csv scores.csv

Deps: numpy, onnxruntime, sounddevice (microphone only).
"""
from __future__ import annotations

import argparse
import csv
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drone_audio.native import CLASSES, load_native_model  # noqa: E402  (CLASSES: drone / vehicle / other)

SR = 16000  # the model's input rate
DEFAULT_MODEL = ROOT / "model"

RED, GREEN, DIM, BOLD, RESET = "\033[31m", "\033[32m", "\033[2m", "\033[1m", "\033[0m"


def read_wav(path: Path) -> np.ndarray:
    """16-bit PCM wav -> mono float32 in [-1, 1] at SR."""
    try:
        with wave.open(str(path), "rb") as handle:
            rate, channels, width = handle.getframerate(), handle.getnchannels(), handle.getsampwidth()
            raw = handle.readframes(handle.getnframes())
    except (OSError, wave.Error) as exc:
        raise SystemExit(f"{path}: not a readable PCM wav ({exc}); convert it with "
                         f"ffmpeg -i in -ac 1 -ar 16000 -sample_fmt s16 out.wav") from None
    if width != 2:
        raise SystemExit(f"{path}: only 16-bit PCM wav is supported")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    data = data.reshape(-1, channels).mean(axis=1)
    if rate != SR:
        positions = np.arange(0, len(data), rate / SR)
        data = np.interp(positions, np.arange(len(data)), data).astype(np.float32)
    return data


def sounddevice_ok() -> bool:
    try:
        import sounddevice  # noqa: F401  (raises OSError if the PortAudio lib is missing)
    except (ImportError, OSError):
        return False
    return True


def pick_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if sounddevice_ok():
        return "sounddevice"
    for tool, backend in (("parecord", "parecord"), ("arecord", "arecord")):
        if shutil.which(tool):
            return backend
    raise SystemExit("No microphone backend: install PortAudio (apt install libportaudio2) for sounddevice, "
                     "or pulseaudio-utils (parecord) / alsa-utils (arecord).")


def list_devices(backend: str) -> None:
    if backend == "sounddevice":
        import sounddevice as sd

        print(sd.query_devices())
    elif backend == "parecord":
        print("PulseAudio/PipeWire sources (pass the name to --device):", flush=True)
        subprocess.run(["pactl", "list", "short", "sources"], check=False)
    else:
        print("ALSA capture devices (pass e.g. hw:1,0 or plughw:1,0 to --device):", flush=True)
        subprocess.run(["arecord", "-l"], check=False)


def recorder_cmd(backend: str, device: str | None) -> list[str]:
    if backend == "parecord":
        cmd = ["parecord", "--raw", "--format=s16le", f"--rate={SR}", "--channels=1", "--latency-msec=50"]
        return cmd + ([f"--device={device}"] if device else [])
    cmd = ["arecord", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(SR), "-c", "1"]
    return cmd + (["-D", device] if device else [])


def subprocess_feeder(proc: subprocess.Popen, chunk: int, out: queue.Queue, stop: threading.Event) -> None:
    nbytes = chunk * 2
    while not stop.is_set():
        raw = proc.stdout.read(nbytes)
        if not raw:
            print(f"\nrecorder exited (code {proc.poll()})", file=sys.stderr, flush=True)
            out.put(None)
            return
        if len(raw) % 2:
            raw = raw[:-1]
        out.put(np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)


def wav_feeder(data: np.ndarray, chunk: int, out: queue.Queue, stop: threading.Event, loop: bool) -> None:
    period = chunk / SR
    next_t = time.perf_counter()
    while not stop.is_set():
        for start in range(0, len(data), chunk):
            if stop.is_set():
                return
            out.put(data[start : start + chunk])
            next_t += period
            time.sleep(max(0.0, next_t - time.perf_counter()))  # real-time pacing without drift
        if not loop:
            out.put(None)
            return


class Window:
    """Fixed-size sliding window of the most recent samples."""

    def __init__(self, size: int) -> None:
        self.buf = np.zeros(size, np.float32)
        self.filled = 0

    def push(self, x: np.ndarray) -> None:
        n = len(x)
        if n >= len(self.buf):
            self.buf[:] = x[-len(self.buf) :]
        else:
            self.buf[:-n] = self.buf[n:]
            self.buf[-n:] = x
        self.filled = min(len(self.buf), self.filled + n)

    @property
    def full(self) -> bool:
        return self.filled >= len(self.buf)


def bar(p: float, width: int, color: str = "") -> str:
    n = int(round(p * width))
    return f"{color}{'█' * n}{RESET}{DIM}{'·' * (width - n)}{RESET}"


def dbfs(x: np.ndarray) -> float:
    return 20.0 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9)


class Display:
    def __init__(self, plain: bool, threshold: float, width: int = 40) -> None:
        self.plain = plain or not sys.stdout.isatty()
        self.thr = threshold
        self.width = width
        self.lines = 0

    def _redraw(self, text: list[str]) -> None:
        if self.lines:
            sys.stdout.write(f"\033[{self.lines}F")  # cursor up to the start of the previous block
        sys.stdout.write("".join(f"\033[2K{line}\n" for line in text))
        sys.stdout.flush()
        self.lines = len(text)

    def filling(self, frac: float, level: float) -> None:
        if self.plain:
            return
        self._redraw([f"Filling window {bar(frac, self.width)} {frac * 100:3.0f}%   input {level:6.1f} dBFS"])

    def show(self, t: float, p_drone: float, p3: np.ndarray, ms: float, level: float, streak: int) -> None:
        drone = p_drone >= self.thr
        top = CLASSES[int(np.argmax(p3))]
        if self.plain:
            probs = " ".join(f"{c}={p:.2f}" for c, p in zip(CLASSES, p3))
            flag = "DRONE" if drone else "-----"
            print(f"[{t:7.1f}s] {flag} p_drone={p_drone:.3f} {probs} top={top} {ms:.0f}ms {level:.1f}dBFS", flush=True)
            return
        verdict = f"{BOLD}{RED}DRONE DETECTED{RESET}" if drone else f"{GREEN}no drone{RESET}"
        mark = int(round(self.thr * self.width))
        lines = [
            f"{BOLD}t={t:7.1f}s{RESET}  {verdict}  {DIM}(streak {streak}){RESET}",
            f"  P(drone)           {bar(p_drone, self.width, RED if drone else '')} {p_drone:5.3f}",
            f"  {'':18} {' ' * mark}^ thr {self.thr:.3f}",
        ]
        for name, p in zip(CLASSES, p3):
            arrow = "◀" if name == top else " "
            lines.append(f"  {name:18} {bar(float(p), self.width)} {p:5.3f} {arrow}")
        lines.append(f"  {DIM}inference {ms:5.0f} ms   input {level:6.1f} dBFS{RESET}")
        self._redraw(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Real-time drone / vehicle / other audio monitor")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="model dir (model.json + onnx), default ./model")
    ap.add_argument("--wav", type=Path, help="replay a 16-bit wav at real-time speed instead of the mic")
    ap.add_argument("--loop", action="store_true", help="loop the --wav file")
    ap.add_argument("--device", default=None, help="sounddevice input name or index")
    ap.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    ap.add_argument("--backend", choices=("auto", "sounddevice", "parecord", "arecord"), default="auto",
                    help="microphone backend (auto: sounddevice, else parecord, else arecord)")
    ap.add_argument("--window", type=float, default=5.0, help="window seconds (model is trained on 5)")
    ap.add_argument("--hop", type=float, default=0.5, help="seconds between scores")
    ap.add_argument("--threshold", type=float, default=None, help="override the model's threshold")
    ap.add_argument("--threads", type=int, default=4, help="onnxruntime intra-op threads")
    ap.add_argument("--emb-avg", type=int, default=1, help="average the last N embeddings (steadier, slower)")
    ap.add_argument("--plain", action="store_true", help="one line per score instead of the live view")
    ap.add_argument("--csv", type=Path, help="append every score to this csv file")
    args = ap.parse_args()

    backend = None if args.wav is not None else pick_backend(args.backend)
    if args.list_devices:
        list_devices(backend or pick_backend(args.backend))
        return 0

    wav_data = read_wav(args.wav) if args.wav is not None else None
    print(f"Loading {args.model} ...", flush=True)
    model = load_native_model(args.model, threads=args.threads, emb_avg=args.emb_avg)
    threshold = model.thr if args.threshold is None else args.threshold
    window_n, hop_n = int(args.window * SR), max(1, int(args.hop * SR))
    model(np.zeros(window_n, np.float32))  # warm-up so the first live score isn't slow
    model.hist.clear()

    chunks: queue.Queue = queue.Queue()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    stream = proc = None
    if wav_data is not None:
        threading.Thread(target=wav_feeder, args=(wav_data, hop_n, chunks, stop, args.loop), daemon=True).start()
        source = str(args.wav)
    elif backend in ("parecord", "arecord"):
        proc = subprocess.Popen(recorder_cmd(backend, args.device), stdout=subprocess.PIPE)
        threading.Thread(target=subprocess_feeder, args=(proc, hop_n, chunks, stop), daemon=True).start()
        source = f"microphone via {backend} ({args.device or 'default'})"
    else:
        import sounddevice as sd

        def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
            if status:
                print(f"audio status: {status}", file=sys.stderr, flush=True)
            chunks.put(indata[:, 0].copy())

        device = int(args.device) if args.device is not None and args.device.isdigit() else args.device
        stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=hop_n,
                                device=device, callback=callback)
        stream.start()
        source = f"microphone ({args.device or 'default'})"

    print(f"Source: {source} @ {SR} Hz | window {args.window}s hop {args.hop}s | "
          f"k={model.k} thr={threshold:.3f} emb_avg={args.emb_avg} | Ctrl+C to stop", flush=True)

    csv_file = writer = None
    if args.csv:
        new = not args.csv.exists()
        csv_file = args.csv.open("a", newline="")
        writer = csv.writer(csv_file)
        if new:
            writer.writerow(["wall_time", "stream_s", "p_drone", *CLASSES, "drone", "latency_ms"])

    win = Window(window_n)
    display = Display(args.plain, threshold)
    received = pending = streak = 0
    try:
        while not stop.is_set():
            try:
                chunk = chunks.get(timeout=0.5)
            except queue.Empty:
                continue
            if chunk is None:
                break
            # If inference ever falls behind, catch up on everything queued and score only the newest window.
            while True:
                win.push(chunk)
                received += len(chunk)
                pending += len(chunk)
                try:
                    chunk = chunks.get_nowait()
                except queue.Empty:
                    break
                if chunk is None:
                    stop.set()
                    break
            level = dbfs(win.buf[-hop_n:])
            if not win.full:
                display.filling(win.filled / window_n, level)
                continue
            if pending < hop_n:
                continue
            pending = 0

            t0 = time.perf_counter()
            p_drone, p3 = model(win.buf.copy())
            ms = (time.perf_counter() - t0) * 1000.0
            drone = p_drone >= threshold
            streak = streak + 1 if drone else 0
            display.show(received / SR, p_drone, p3, ms, level, streak)
            if writer:
                writer.writerow([f"{time.time():.3f}", f"{received / SR:.2f}", f"{p_drone:.4f}",
                                 *(f"{p:.4f}" for p in p3), int(drone), f"{ms:.1f}"])
                csv_file.flush()
    finally:
        stop.set()
        if stream is not None:
            stream.stop()
            stream.close()
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        if csv_file:
            csv_file.close()
        print("Stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
