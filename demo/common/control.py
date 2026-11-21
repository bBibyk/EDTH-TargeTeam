from __future__ import annotations

import json
import socket
import socketserver
import threading
import time
from collections.abc import Callable
from typing import Any


class _ControlHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        for raw_line in self.rfile:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                response: dict[str, Any] = {"error": "invalid json"}
            else:
                command = str(message.get("cmd", "")).strip().upper()
                if not command:
                    response = {"error": "missing cmd"}
                else:
                    try:
                        result = self.server.on_command(command) or {}
                        response = {"ack": command, **result}
                    except Exception as exc:  # noqa: BLE001
                        response = {"error": str(exc)}

            self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))
            self.wfile.flush()


class _ControlTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[socketserver.BaseRequestHandler],
        on_command: Callable[[str], dict[str, Any] | None],
    ) -> None:
        self.on_command = on_command
        super().__init__(server_address, handler_class)


class ControlServer:
    def __init__(
        self,
        host: str,
        port: int,
        on_command: Callable[[str], dict[str, Any] | None],
    ) -> None:
        self._host = host
        self._port = port
        self._on_command = on_command
        self._server: _ControlTCPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        return self._host, self._port

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = _ControlTCPServer(
            (self._host, self._port), _ControlHandler, self._on_command
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="control-server",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


class ControlClient:
    def __init__(self, host: str, port: int, timeout: float = 2.0) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout

    def send(self, command: str, retries: int = 1) -> dict[str, Any]:
        normalized = command.strip().upper()
        payload = (json.dumps({"cmd": normalized}) + "\n").encode("utf-8")
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            try:
                with socket.create_connection(
                    (self._host, self._port), timeout=self._timeout
                ) as connection:
                    connection.settimeout(self._timeout)
                    connection.sendall(payload)
                    with connection.makefile("rb") as reader:
                        response = reader.readline()
                    if not response:
                        raise RuntimeError("empty control response")
                    return json.loads(response.decode("utf-8"))
            except (OSError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(0.1)

        raise ConnectionError(f"control command failed: {last_error}")
