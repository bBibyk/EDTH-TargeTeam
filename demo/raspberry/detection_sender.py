from __future__ import annotations

import json
import socket
import threading
from typing import Any


class DetectionSender:
    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._lock = threading.Lock()
        self._socket: socket.socket | None = None

    def close(self) -> None:
        with self._lock:
            sock = self._socket
            self._socket = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def send(
        self, detections: list[dict[str, Any]], frame_size: tuple[int, int]
    ) -> None:
        payload = {
            "frame_w": frame_size[0],
            "frame_h": frame_size[1],
            "detections": detections,
        }
        line = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        with self._lock:
            sock = self._ensure_socket()
            if sock is None:
                return
            try:
                sock.sendall(line)
            except OSError:
                self._close_locked()
                sock = self._ensure_socket()
                if sock is None:
                    return
                try:
                    sock.sendall(line)
                except OSError:
                    self._close_locked()

    def _ensure_socket(self) -> socket.socket | None:
        if self._socket is not None:
            return self._socket
        try:
            sock = socket.create_connection((self._host, self._port), timeout=1.0)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(1.0)
            self._socket = sock
            return sock
        except OSError:
            self._socket = None
            return None

    def _close_locked(self) -> None:
        sock = self._socket
        self._socket = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
