from __future__ import annotations

import os
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
ROOT_DIR = DEMO_DIR.parent
RUNTIME_DIR = Path(os.getenv("DEMO_RUNTIME_DIR", str(DEMO_DIR / "runtime")))

GST_VIDEO_LOG_PATH = RUNTIME_DIR / "gst_video.log"
GST_AUDIO_LOG_PATH = RUNTIME_DIR / "gst_audio.log"


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


RBP_HOST = os.getenv("RBP_HOST", "127.0.0.1")
RBP_BIND_HOST = os.getenv("RBP_BIND_HOST", "0.0.0.0")
PC_HOST = os.getenv("PC_HOST", "127.0.0.1")
PC_PUBLIC_HOST = os.getenv("PC_PUBLIC_HOST", "127.0.0.1")

CONTROL_PORT = _env_int("CONTROL_PORT", 9200)
VIDEO_RTP_PORT = _env_int("VIDEO_RTP_PORT", 9100)
AUDIO_RTP_PORT = _env_int("AUDIO_RTP_PORT", 9102)
VIDEO_HTTP_PORT = _env_int("VIDEO_HTTP_PORT", 8080)

VIDEO_DEVICE = os.getenv("VIDEO_DEVICE", "/dev/video4")
AUDIO_DEVICE = os.getenv("AUDIO_DEVICE", "default")
AUDIO_INPUT_FORMAT = os.getenv("AUDIO_INPUT_FORMAT", "pulse")

VIDEO_WIDTH = _env_int("VIDEO_WIDTH", 640)
VIDEO_HEIGHT = _env_int("VIDEO_HEIGHT", 480)
VIDEO_FPS = _env_int("VIDEO_FPS", 30)

AUDIO_RATE = _env_int("AUDIO_RATE", 16000)
AUDIO_CHANNELS = _env_int("AUDIO_CHANNELS", 1)

GST_JITTERBUFFER_MS = _env_int("GST_JITTERBUFFER_MS", 50)
JPEG_QUALITY = _env_int("JPEG_QUALITY", 70)
X264_BITRATE_KBPS = _env_int("X264_BITRATE_KBPS", 2500)
OPUS_BITRATE = _env_int("OPUS_BITRATE", 32000)
CAMERA_FORCE_FPS = os.getenv("CAMERA_FORCE_FPS", "1") == "1"

VIDEO_FRAME_BYTES = VIDEO_WIDTH * VIDEO_HEIGHT * 3
AUDIO_CHUNK_SAMPLES = _env_int("AUDIO_CHUNK_SAMPLES", 1024)
AUDIO_CHUNK_BYTES = AUDIO_CHUNK_SAMPLES * 2

AUDIO_BUFFER_SECONDS = _env_float("AUDIO_BUFFER_SECONDS", 5.0)
AUDIO_BUFFER_SAMPLES = int(AUDIO_RATE * AUDIO_BUFFER_SECONDS)
WAVEFORM_SECONDS = _env_float("WAVEFORM_SECONDS", 1.0)
MEL_SECONDS = _env_float("MEL_SECONDS", 2.0)

MEL_N_FFT = _env_int("MEL_N_FFT", 1024)
MEL_HOP_LENGTH = _env_int("MEL_HOP_LENGTH", 256)
MEL_N_MELS = _env_int("MEL_N_MELS", 128)
MEL_FMIN = _env_float("MEL_FMIN", 20.0)
MEL_FMAX = _env_float("MEL_FMAX", min(8000.0, AUDIO_RATE / 2.0))

UI_REFRESH_SECONDS = _env_float("UI_REFRESH_SECONDS", 0.1)
STREAM_STALE_SECONDS = _env_float("STREAM_STALE_SECONDS", 2.0)
CONTROL_TIMEOUT_SECONDS = _env_float("CONTROL_TIMEOUT_SECONDS", 2.0)

TONE_SAMPLE_RATE = _env_int("TONE_SAMPLE_RATE", 44100)


def ensure_runtime_dir() -> Path:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    return RUNTIME_DIR
