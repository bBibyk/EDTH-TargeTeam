"""Read the STM32 microphone (stm32/mic_stream firmware) from the ST-LINK USB serial port.

The board streams raw 32 kHz mono int16 audio in packets, 1 Mbaud, little-endian:

    A5 5A | seq u16 | n u16 | n x int16 | checksum u16 = seq + n + samples (as u16), mod 2^16

At boot it also prints one text line, `$MIC,32000,1,S16LE,<version>`; text is skipped.

    python -m demo.raspberry.audio_wake.stm32_mic                       # live stats per second
    python -m demo.raspberry.audio_wake.stm32_mic --seconds 30 --record clip.wav
    python -m demo.raspberry.audio_wake.main --device stm32 --verbose   # feed the audio model
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import os
import select
import struct
import sys
import termios
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

SAMPLE_RATE = 32000
DEFAULT_BAUD = 1_000_000
MAGIC = b"\xa5\x5a"
MAX_SAMPLES_PER_PACKET = 2048
RESYNC_GAP_PACKETS = 125  # a jump of more than 1 s is a resync (stale data, board reset), not loss


def find_port() -> str:
    """The ST-LINK virtual COM port of the board (stable by-id name if present)."""
    for pattern in ("/dev/serial/by-id/*STLink*", "/dev/serial/by-id/*STM32*", "/dev/ttyACM0"):
        found = sorted(glob.glob(pattern))
        if found:
            return found[0]
    return "/dev/ttyACM0"


def _open_raw(port: str, baud: int) -> int:
    speed = getattr(termios, f"B{baud}", None)
    if speed is None:
        raise ValueError(f"baud rate {baud} is not supported by termios")
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        # Exclusive: a second reader would silently steal half of the packets.
        fcntl.ioctl(fd, termios.TIOCEXCL)
    except OSError:
        pass
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0                                                   # iflag: raw
    attrs[1] = 0                                                   # oflag
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL        # cflag: 8N1
    attrs[3] = 0                                                   # lflag
    attrs[4] = attrs[5] = speed
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    termios.tcflush(fd, termios.TCIOFLUSH)
    return fd


class PacketParser:
    """Turns the byte stream into int16 sample blocks; counts gaps and corrupt packets."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._last_seq: int | None = None
        self.packets = 0
        self.lost_packets = 0
        self.resyncs = 0
        self.bad_packets = 0
        self.skipped_bytes = 0
        self.text_lines: list[str] = []

    def feed(self, data: bytes) -> list[np.ndarray]:
        self._buf += data
        blocks: list[np.ndarray] = []
        while True:
            start = self._buf.find(MAGIC)
            if start < 0:
                self._skip_text(len(self._buf) - 1)
                break
            if start > 0:
                self._skip_text(start)
            if len(self._buf) < 6:
                break
            seq, n = struct.unpack_from("<HH", self._buf, 2)
            if not 0 < n <= MAX_SAMPLES_PER_PACKET:
                self._drop(1)
                continue
            size = 6 + 2 * n + 2
            if len(self._buf) < size:
                break
            samples = np.frombuffer(bytes(self._buf[6:6 + 2 * n]), dtype="<i2")
            (checksum,) = struct.unpack_from("<H", self._buf, 6 + 2 * n)
            if int((seq + n + int(samples.astype(np.uint16).sum(dtype=np.uint64))) & 0xFFFF) != checksum:
                self._drop(1)  # corrupt or false sync: resync on the next magic
                continue
            del self._buf[:size]
            if self._last_seq is not None:
                gap = (seq - self._last_seq - 1) & 0xFFFF
                if gap > RESYNC_GAP_PACKETS:
                    self.resyncs += 1
                else:
                    self.lost_packets += gap
            self._last_seq = seq
            self.packets += 1
            blocks.append(samples)
        return blocks

    def _drop(self, count: int) -> None:
        self.bad_packets += 1
        self.skipped_bytes += count
        del self._buf[:count]

    def _skip_text(self, count: int) -> None:
        if count <= 0:
            return
        chunk = bytes(self._buf[:count])
        for line in chunk.split(b"\n"):
            text = line.strip().decode("ascii", errors="ignore")
            if text.startswith("$"):
                self.text_lines.append(text)
                del self.text_lines[:-10]
        self.skipped_bytes += count
        del self._buf[:count]


class Stm32Microphone:
    """Background reader: calls on_audio(block) for every packet.

    block is float32 in [-1, 1], or the raw int16 samples with as_int16=True.
    With reconnect=True it keeps retrying (board unplugged, being reflashed, ...)
    instead of stopping at the first error.
    """

    sample_rate = SAMPLE_RATE

    def __init__(
        self,
        on_audio: Callable[[np.ndarray], None],
        port: str | None = None,
        baud: int = DEFAULT_BAUD,
        as_int16: bool = False,
        reconnect: bool = False,
    ) -> None:
        self._fixed_port = port
        self.port = port or find_port()
        self.baud = baud
        self._on_audio = on_audio
        self._as_int16 = as_int16
        self._reconnect = reconnect
        self.parser = PacketParser()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self.samples = 0
        self.started_at = 0.0
        self.error: str | None = None
        self.connected = False

    def start(self) -> None:
        if not self._reconnect:
            self._fd = _open_raw(self.port, self.baud)  # fail loudly right away
        self.started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="stm32-mic", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._close()

    def _close(self) -> None:
        self.connected = False
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def _run(self) -> None:
        reported = ""
        while not self._stop.is_set():
            try:
                if self._fd is None:
                    self.port = self._fixed_port or find_port()
                    self._fd = _open_raw(self.port, self.baud)
                self._read_until_error()
            except OSError as exc:  # missing, unplugged, busy, ...
                self.error = str(exc)
            self._close()
            if not self._reconnect or self._stop.is_set():
                return
            if self.error != reported:
                print(f"STM32 mic: {self.error} - retrying every 2 s", flush=True)
                reported = self.error or ""
            self._stop.wait(2.0)

    def _read_until_error(self) -> None:
        assert self._fd is not None
        # The ST-LINK keeps a few stale bytes from before the port was opened: drop them.
        until = time.monotonic() + 0.1
        while time.monotonic() < until:
            if select.select([self._fd], [], [], 0.02)[0]:
                os.read(self._fd, 65536)
        self.parser = PacketParser()
        self.connected = True
        if self._reconnect:
            print(f"STM32 mic connected: {self.port} @ {self.baud} baud", flush=True)
        while not self._stop.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.2)
            if not ready:
                continue
            data = os.read(self._fd, 65536)
            if not data:
                raise OSError("serial port closed")
            for block in self.parser.feed(data):
                self.samples += len(block)
                self._on_audio(block if self._as_int16 else block.astype(np.float32) / 32768.0)


def _dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check / record the STM32 microphone stream")
    parser.add_argument("--port", default=os.getenv("STM32_MIC_PORT") or None)
    parser.add_argument("--baud", type=int, default=int(os.getenv("STM32_MIC_BAUD", DEFAULT_BAUD)))
    parser.add_argument("--seconds", type=float, default=0, help="Stop after this long (0 = Ctrl+C)")
    parser.add_argument("--record", type=Path, help="Write the audio to this 16-bit WAV file")
    args = parser.parse_args()

    lock = threading.Lock()
    second: list[np.ndarray] = []
    recorded: list[np.ndarray] = []

    def on_audio(block: np.ndarray) -> None:
        with lock:
            second.append(block)
            if args.record:
                recorded.append(block)

    mic = Stm32Microphone(on_audio, port=args.port, baud=args.baud)
    mic.start()
    print(f"Reading {mic.port} at {mic.baud} baud (Ctrl+C to stop)", flush=True)
    t0 = time.monotonic()
    try:
        while not args.seconds or time.monotonic() - t0 < args.seconds:
            time.sleep(1.0)
            with lock:
                chunk = np.concatenate(second) if second else np.zeros(0, np.float32)
                second.clear()
            elapsed = time.monotonic() - mic.started_at
            p = mic.parser
            level = f"rms {_dbfs(chunk):6.1f} dBFS  peak {20 * np.log10(np.abs(chunk).max() + 1e-12):6.1f} dBFS" if len(chunk) else "no audio"
            print(
                f"{elapsed:5.0f}s  {mic.samples / max(elapsed, 1e-9):7.0f} samples/s  {level}  "
                f"packets {p.packets}  lost {p.lost_packets}  bad {p.bad_packets}  resyncs {p.resyncs}",
                flush=True,
            )
            if mic.error:
                print(f"error: {mic.error}", flush=True)
                break
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
    if mic.parser.text_lines:
        print("board said:", " | ".join(mic.parser.text_lines), flush=True)
    if args.record:
        audio = np.concatenate(recorded) if recorded else np.zeros(0, np.float32)
        with wave.open(str(args.record), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes((np.clip(audio, -1, 32767 / 32768) * 32768).astype("<i2").tobytes())
        print(f"wrote {args.record} ({len(audio) / SAMPLE_RATE:.1f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
