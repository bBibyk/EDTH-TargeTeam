"""Wake signal between the audio wake program and the visual detector.

One UDP datagram per positive audio window, JSON encoded:

    {"type": "wake", "source": "audio", "score": 0.91, "ts": 1732190000.123}

UDP is fire-and-forget: the audio side re-sends on every positive window, so a
lost datagram only delays the wake by one hop. Each wake (re)arms the detector
for WAKE_HOLD_SECONDS.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Callable
from typing import Any


class WakeSender:
    def __init__(self, host: str, port: int) -> None:
        self._address = (host, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, score: float, source: str = "audio", **extra: Any) -> None:
        message = {
            "type": "wake",
            "source": source,
            "score": float(score),
            "ts": time.time(),
            **extra,
        }
        self._socket.sendto(json.dumps(message).encode("utf-8"), self._address)

    def close(self) -> None:
        self._socket.close()


class WakeListener:
    """Receives wake datagrams on a background thread and calls `on_wake(message)`."""

    def __init__(
        self,
        host: str,
        port: int,
        on_wake: Callable[[dict[str, Any]], None],
    ) -> None:
        self._address = (host, port)
        self._on_wake = on_wake
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._socket is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(self._address)
        sock.settimeout(0.5)
        self._socket = sock
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="wake-listener", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                data, _peer = self._socket.recvfrom(4096)
            except TimeoutError:
                continue
            except OSError:
                break
            try:
                message = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(message, dict) and message.get("type") == "wake":
                self._on_wake(message)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None
