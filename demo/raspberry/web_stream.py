"""Live camera view in a web browser, served by the Pi itself (no PC app needed).

The USB camera already outputs JPEG frames (MJPEG). ffmpeg passes them through
unchanged (no re-encoding, almost no CPU), and the same MJPEG HTTP server the PC
app uses (demo/pc/mjpeg_server.py) pushes the latest frame to every browser.

    ./demo/scripts/run_web_stream.sh                     # settings from demo/.env
    ./demo/scripts/run_web_stream.sh --width 1280 --height 720

Then open http://<pi-address>:<VIDEO_HTTP_PORT>/ from any device on the same network.
Only one program can use the camera at a time: stop run_rbp.sh / run_detector.sh first.
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import threading
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from demo import config
from demo.pc.mjpeg_server import MjpegStreamServer
from demo.pc.state import AppState

SOI, EOI = b"\xff\xd8", b"\xff\xd9"  # JPEG start / end markers


def _force_fixed_fps(device: str) -> None:
    """Stop the camera lowering its frame rate in low light (same as gst_sender)."""
    if not config.CAMERA_FORCE_FPS:
        return
    try:
        subprocess.run(
            ["v4l2-ctl", "-d", device, "-c", "exposure_dynamic_framerate=0"],
            check=False, capture_output=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def _camera_command(device: str, width: int, height: int, fps: int) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "v4l2", "-input_format", "mjpeg",
        "-video_size", f"{width}x{height}", "-framerate", str(fps),
        "-i", device,
        "-c:v", "copy", "-f", "mjpeg", "-",  # camera JPEGs, untouched, to stdout
    ]


def _pump_frames(proc: subprocess.Popen, state: AppState) -> None:
    """Split ffmpeg's byte stream into JPEG frames and publish the newest one."""
    buf = b""
    while chunk := proc.stdout.read1(65536):
        buf += chunk
        while True:
            start = buf.find(SOI)
            if start < 0:
                buf = buf[-1:]  # keep a possible half marker
                break
            end = buf.find(EOI, start + 2)
            if end < 0:
                buf = buf[start:]
                break
            state.set_frame(buf[start:end + 2])
            buf = buf[end + 2:]


def _keep_tail(stream, tail: list[str]) -> None:
    """Drain ffmpeg's stderr (so it never blocks) and keep the last lines."""
    for line in iter(stream.readline, b""):
        tail.append(line.decode(errors="replace").rstrip())
        del tail[:-5]


def _local_addresses() -> list[str]:
    try:
        out = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", "scope", "global"],
            capture_output=True, text=True, timeout=2,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    addrs = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) > 3 and not parts[1].startswith(("docker", "veth", "br-")):
            addrs.append(parts[3].split("/")[0])
    return addrs


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the Pi camera to web browsers")
    parser.add_argument("--device", default=config.VIDEO_DEVICE)
    parser.add_argument("--width", type=int, default=config.VIDEO_WIDTH)
    parser.add_argument("--height", type=int, default=config.VIDEO_HEIGHT)
    parser.add_argument("--fps", type=int, default=config.VIDEO_FPS)
    parser.add_argument("--host", default=config.RBP_BIND_HOST)
    parser.add_argument("--port", type=int, default=config.VIDEO_HTTP_PORT)
    args = parser.parse_args()

    stop = threading.Event()

    def _handle_signal(signum: int, frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    state = AppState()
    server = MjpegStreamServer(args.host, args.port, state)
    server.start()
    if not server.is_running:
        print(f"Cannot start the web server: {server.error}", flush=True)
        return 1

    print(
        f"Camera {args.device} {args.width}x{args.height}@{args.fps} -> web browser",
        flush=True,
    )
    for addr in _local_addresses() or ["<pi-address>"]:
        print(f"  open http://{addr}:{server.actual_port}/", flush=True)

    _force_fixed_fps(args.device)
    try:
        while not stop.is_set():
            proc = subprocess.Popen(
                _camera_command(args.device, args.width, args.height, args.fps),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            tail: list[str] = []
            threading.Thread(target=_pump_frames, args=(proc, state), daemon=True).start()
            threading.Thread(target=_keep_tail, args=(proc.stderr, tail), daemon=True).start()
            while not stop.is_set() and proc.poll() is None:
                stop.wait(0.5)
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            if not stop.is_set():
                reason = " | ".join(tail) or f"exit code {proc.returncode}"
                print(f"Camera stopped ({reason}). Retrying in 2 s...", flush=True)
                stop.wait(2.0)
    finally:
        server.stop()
        print("Web stream stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
