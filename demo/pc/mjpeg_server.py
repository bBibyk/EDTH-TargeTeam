from __future__ import annotations

import io
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageDraw, ImageFont

from demo.pc.state import AppState


def _draw_detections(
    frame: bytes, detections: list[dict], frame_size: tuple[int, int]
) -> bytes:
    if not detections:
        return frame
    try:
        image = Image.open(io.BytesIO(frame)).convert("RGB")
    except OSError:
        return frame
    draw = ImageDraw.Draw(image)
    src_w, src_h = frame_size
    img_w, img_h = image.size
    sx = img_w / src_w if src_w > 0 else 1.0
    sy = img_h / src_h if src_h > 0 else 1.0
    font = ImageFont.load_default()
    for det in detections:
        xyxy = det.get("xyxy")
        if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy
        label = str(det.get("label", "obj"))
        conf = float(det.get("conf", 0.0))
        text = f"{label} {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        text_box = draw.textbbox((x1, y1), text, font=font)
        tw = text_box[2] - text_box[0]
        th = text_box[3] - text_box[1]
        ty = max(0, y1 - th - 4)
        draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 4], fill=(0, 255, 0))
        draw.text((x1 + 2, ty + 2), text, fill=(0, 0, 0), font=font)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=80)
    return output.getvalue()


class MjpegStreamServer:
    def __init__(self, host: str, port: int, state: AppState) -> None:
        self._host = host
        self._port = port
        self._actual_port = port
        self._state = state
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._error = ""

    @property
    def actual_port(self) -> int:
        return self._actual_port

    @property
    def error(self) -> str:
        return self._error

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def url(self, host: str | None = None) -> str:
        return f"http://{host or self._host}:{self._actual_port}/stream.mjpg"

    def start(self) -> None:
        if self.is_running:
            return
        state = self._state

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path not in ("/", "/stream.mjpg"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.end_headers()
                try:
                    while True:
                        frame = state.get_frame()
                        if frame is None:
                            time.sleep(0.05)
                            continue
                        detections, frame_size = state.get_detections()
                        frame = _draw_detections(frame, detections, frame_size)
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(frame)}\r\n\r\n".encode()
                        )
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        time.sleep(1 / 30)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return

            def log_message(self, format: str, *args: object) -> None:
                return

        class Server(ThreadingHTTPServer):
            allow_reuse_address = True

        last_error: OSError | None = None
        for port in range(self._port, self._port + 20):
            try:
                server = Server((self._host, port), Handler)
                self._server = server
                self._actual_port = port
                self._error = ""
                break
            except OSError as exc:
                last_error = exc
        else:
            self._error = str(last_error) if last_error else "could not bind port"
            return

        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="mjpeg-http", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
