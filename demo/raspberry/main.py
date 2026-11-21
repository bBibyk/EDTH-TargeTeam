from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from demo import config
from demo.common.control import ControlServer
from demo.raspberry.gst_sender import GstSender


def _on_command(command: str) -> dict[str, object]:
    print(f"received {command} command", flush=True)
    return {"ok": True}


def _print_sender_failure(sender: GstSender) -> None:
    if sender.video is not None and sender.video.error:
        print(f"Video sender error: {sender.video.error}", flush=True)
    if sender.audio is not None and sender.audio.error:
        print(f"Audio sender error: {sender.audio.error}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Raspberry drone-side GStreamer sender")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Use synthetic video and audio instead of webcam/microphone",
    )
    args = parser.parse_args()

    stop_event = threading.Event()

    def _handle_signal(signum: int, frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    sender = GstSender(fake=args.fake)
    control_server = ControlServer(
        config.RBP_BIND_HOST, config.CONTROL_PORT, _on_command
    )

    try:
        print(
            f"Starting GStreamer senders ({'synthetic' if args.fake else 'hardware'})",
            flush=True,
        )
        video_source = "videotestsrc" if args.fake else config.VIDEO_DEVICE
        audio_source = "audiotestsrc" if args.fake else config.AUDIO_DEVICE
        print(
            f"Video: {video_source} -> rtp://{config.PC_HOST}:{config.VIDEO_RTP_PORT}",
            flush=True,
        )
        print(
            f"Audio: {audio_source} -> rtp://{config.PC_HOST}:{config.AUDIO_RTP_PORT}",
            flush=True,
        )
        try:
            sender.start()
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to start GStreamer senders: {exc}", flush=True)
            return 1
        try:
            control_server.start()
        except OSError as exc:
            print(f"Failed to start control server: {exc}", flush=True)
            return 1

        status = sender.status()
        print(f"Sender status: {status}", flush=True)
        print(
            f"Control server listening on {config.RBP_BIND_HOST}:{config.CONTROL_PORT}",
            flush=True,
        )
        _print_sender_failure(sender)
        print("Raspberry simulation ready. Press Ctrl+C to stop.", flush=True)

        while not stop_event.wait(0.5):
            if sender.video is not None and not sender.video.is_running:
                print("Video sender exited unexpectedly.", flush=True)
                _print_sender_failure(sender)
                stop_event.set()
            if sender.audio is not None and not sender.audio.is_running:
                print("Audio sender exited unexpectedly.", flush=True)
                _print_sender_failure(sender)
                stop_event.set()
    finally:
        print("Stopping Raspberry simulation...", flush=True)
        control_server.stop()
        sender.stop()
        print("Stopped.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
