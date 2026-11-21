from __future__ import annotations

import subprocess
import time
from pathlib import Path

from demo import config


class FFmpegProcess:
    def __init__(self, name: str, args: list[str], log_path: Path) -> None:
        self.name = name
        self.args = args
        self.log_path = log_path
        self._process: subprocess.Popen[bytes] | None = None
        self._log_file = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self.is_running:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("wb")
        self._process = subprocess.Popen(
            self.args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=self._log_file,
        )

    def stop(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        self._process = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def log_tail(self, max_bytes: int = 2000) -> str:
        if not self.log_path.exists():
            return ""
        data = self.log_path.read_bytes()
        return data[-max_bytes:].decode("utf-8", errors="replace")


def _video_input_args(fake: bool) -> list[str]:
    if fake:
        return [
            "-re",
            "-f",
            "lavfi",
            "-i",
            (
                f"testsrc2=size={config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT}:"
                f"rate={config.VIDEO_FPS}"
            ),
        ]

    return [
        "-f",
        config.VIDEO_INPUT_FORMAT,
        "-i",
        config.VIDEO_DEVICE,
    ]


def _video_codec_args() -> list[str]:
    if config.VIDEO_CODEC == "libx264":
        return [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-x264-params",
            "repeat-headers=1",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(config.VIDEO_FPS),
            "-keyint_min",
            str(config.VIDEO_FPS),
            "-bf",
            "0",
        ]
    return ["-c:v", config.VIDEO_CODEC, "-q:v", "5"]


def _audio_input_args(fake: bool) -> list[str]:
    if fake:
        return [
            "-re",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate={config.AUDIO_RATE}",
        ]

    return ["-f", config.AUDIO_INPUT_FORMAT, "-i", config.AUDIO_DEVICE]


def _audio_codec_args() -> list[str]:
    if config.AUDIO_CODEC == "libopus":
        return [
            "-c:a",
            "libopus",
            "-b:a",
            config.AUDIO_BITRATE,
            "-application",
            "lowdelay",
            "-frame_duration",
            "20",
            "-vbr",
            "off",
            "-ar",
            "48000",
            "-ac",
            "1",
        ]
    if config.AUDIO_CODEC == "pcm_mulaw":
        return ["-c:a", "pcm_mulaw", "-ar", "8000", "-ac", "1"]
    return ["-c:a", config.AUDIO_CODEC]


def build_video_sender_args(fake: bool = False) -> list[str]:
    scale_filter = (
        f"scale={config.VIDEO_WIDTH}:{config.VIDEO_HEIGHT}:"
        "force_original_aspect_ratio=decrease,"
        f"pad={config.VIDEO_WIDTH}:{config.VIDEO_HEIGHT}:"
        "(ow-iw)/2:(oh-ih)/2:color=black"
    )
    return [
        config.FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "warning",
        *_video_input_args(fake),
        "-vf",
        scale_filter,
        "-r",
        str(config.VIDEO_FPS),
        *_video_codec_args(),
        "-an",
        "-f",
        "rtp",
        "-sdp_file",
        str(config.VIDEO_SDP_PATH),
        f"rtp://{config.PC_HOST}:{config.VIDEO_RTP_PORT}?pkt_size=1200",
    ]


def build_audio_sender_args(fake: bool = False) -> list[str]:
    return [
        config.FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "warning",
        *_audio_input_args(fake),
        *_audio_codec_args(),
        "-vn",
        "-f",
        "rtp",
        "-sdp_file",
        str(config.AUDIO_SDP_PATH),
        f"rtp://{config.PC_HOST}:{config.AUDIO_RTP_PORT}?pkt_size=1200",
    ]


class RtpSender:
    def __init__(self, fake: bool = False) -> None:
        self.fake = fake
        self.video: FFmpegProcess | None = None
        self.audio: FFmpegProcess | None = None

    def start(self) -> bool:
        config.ensure_runtime_dir()
        for sdp_path in (config.VIDEO_SDP_PATH, config.AUDIO_SDP_PATH):
            sdp_path.unlink(missing_ok=True)

        self.video = FFmpegProcess(
            "video", build_video_sender_args(self.fake), config.VIDEO_LOG_PATH
        )
        self.audio = FFmpegProcess(
            "audio", build_audio_sender_args(self.fake), config.AUDIO_LOG_PATH
        )
        self.video.start()
        self.audio.start()
        return self.wait_for_sdp()

    def wait_for_sdp(self) -> bool:
        deadline = time.monotonic() + config.SDP_WAIT_SECONDS
        while time.monotonic() < deadline:
            if config.VIDEO_SDP_PATH.exists() and config.AUDIO_SDP_PATH.exists():
                return True
            if (
                self.video is not None
                and self.audio is not None
                and not self.video.is_running
                and not self.audio.is_running
            ):
                break
            time.sleep(0.1)
        return False

    def status(self) -> dict[str, object]:
        return {
            "video_running": bool(self.video and self.video.is_running),
            "audio_running": bool(self.audio and self.audio.is_running),
            "video_sdp": config.VIDEO_SDP_PATH.exists(),
            "audio_sdp": config.AUDIO_SDP_PATH.exists(),
        }

    def stop(self) -> None:
        if self.video is not None:
            self.video.stop()
        if self.audio is not None:
            self.audio.stop()
