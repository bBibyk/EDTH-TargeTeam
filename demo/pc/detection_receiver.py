from __future__ import annotations

import json
import socket
import threading
from collections.abc import Callable
from typing import Any


class DetectionReceiver:
    def __init__(
        self,
        host: str,
        port: int,
        on_detections: Callable[[list[dict[str, Any]], tuple[int, int]], None],
    ) -> None:
        self._host = host
        self._port = port
        self._on_detections = on_detections
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self._host, self._port))
        server.listen(2)
        server.settimeout(1.0)
        self._socket = server
        self._thread = threading.Thread(
            target=self._serve, name="detection-receiver", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        sock = self._socket
        self._socket = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _serve(self) -> None:
        while not self._stop_event.is_set():
            sock = self._socket
            if sock is None:
                break
            try:
                conn, _addr = sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with conn:
                conn.settimeout(1.0)
                self._handle_client(conn)

    def _handle_client(self, conn: socket.socket) -> None:
        buffer = b""
        while not self._stop_event.is_set():
            try:
                chunk = conn.recv(65536)
            except TimeoutError:
                continue
            except OSError:
                break
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                self._handle_line(line)

    def _handle_line(self, line: bytes) -> None:
        if not line.strip():
            return
        try:
            payload = json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        detections = payload.get("detections")
        if not isinstance(detections, list):
            return
        frame_w = int(payload.get("frame_w", 0) or 0)
        frame_h = int(payload.get("frame_h", 0) or 0)
        self._on_detections(detections, (frame_w, frame_h))
