from __future__ import annotations

import io
import threading
import wave

import numpy as np

from demo import config

_TONE_LOCK = threading.Lock()


def _apply_envelope(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    fade_samples = min(int(sample_rate * 0.01), signal.size // 2)
    if fade_samples <= 0:
        return signal
    envelope = np.ones(signal.size, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    envelope[:fade_samples] = ramp
    envelope[-fade_samples:] = ramp[::-1]
    return signal * envelope


def _sine(frequency: float, duration: float, sample_rate: int) -> np.ndarray:
    times = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
    return np.sin(2.0 * np.pi * frequency * times).astype(np.float32)


def _sweep(
    start_frequency: float,
    end_frequency: float,
    duration: float,
    sample_rate: int,
) -> np.ndarray:
    count = int(duration * sample_rate)
    times = np.arange(count, dtype=np.float32) / sample_rate
    slope = (end_frequency - start_frequency) / max(duration, 1e-6)
    phase = 2.0 * np.pi * (start_frequency * times + 0.5 * slope * times * times)
    return np.sin(phase).astype(np.float32)


def _silence(duration: float, sample_rate: int) -> np.ndarray:
    return np.zeros(int(duration * sample_rate), dtype=np.float32)


def _normalize(signal: np.ndarray, peak: float = 0.9) -> np.ndarray:
    maximum = float(np.max(np.abs(signal)))
    if maximum < 1e-6:
        return signal
    return (signal / maximum * peak).astype(np.float32)


def _cabin_chime(
    frequency: float, duration: float, sample_rate: int, decay: float = 3.5
) -> np.ndarray:
    count = int(duration * sample_rate)
    times = np.arange(count, dtype=np.float32) / sample_rate
    envelope = np.exp(-decay * times).astype(np.float32)
    harmonics = (
        0.62 * np.sin(2.0 * np.pi * frequency * times)
        + 0.24 * np.sin(2.0 * np.pi * frequency * 2.0 * times)
        + 0.14 * np.sin(2.0 * np.pi * frequency * 3.0 * times)
    ).astype(np.float32)
    return harmonics * envelope


def _missile_growl(duration: float, sample_rate: int) -> np.ndarray:
    count = int(duration * sample_rate)
    times = np.arange(count, dtype=np.float32) / sample_rate
    carrier = 210.0
    modulation = 0.45 + 0.55 * (0.5 + 0.5 * np.sin(2.0 * np.pi * 15.0 * times))
    base = (
        0.55 * np.sin(2.0 * np.pi * carrier * times)
        + 0.25 * np.sin(2.0 * np.pi * (carrier * 1.01) * times)
        + 0.20 * np.sin(2.0 * np.pi * carrier * 2.02 * times)
    ).astype(np.float32)
    return (base * modulation).astype(np.float32)


def _missile_lock(duration: float, sample_rate: int) -> np.ndarray:
    count = int(duration * sample_rate)
    times = np.arange(count, dtype=np.float32) / sample_rate
    pulse = 0.72 + 0.28 * np.sin(2.0 * np.pi * 7.0 * times)
    signal = (
        0.58 * np.sin(2.0 * np.pi * 1760.0 * times)
        + 0.22 * np.sin(2.0 * np.pi * 3520.0 * times)
        + 0.20 * np.sin(2.0 * np.pi * 880.0 * times)
    ).astype(np.float32)
    return (signal * pulse).astype(np.float32)


def _cinema_wake_up(sample_rate: int) -> np.ndarray:
    duration = 0.65
    count = int(duration * sample_rate)
    times = np.arange(count, dtype=np.float32) / sample_rate

    main_sweep = _sweep(320.0, 1480.0, duration, sample_rate)
    octave_sweep = _sweep(640.0, 2960.0, duration, sample_rate)
    sub_punch = _sweep(150.0, 65.0, min(0.18, duration), sample_rate)
    sub_padded = np.pad(sub_punch, (0, max(0, count - sub_punch.size)))

    swell = (0.55 + 0.45 * (times / max(duration, 1e-6))).astype(np.float32)
    combined = (
        0.62 * main_sweep * swell
        + 0.24 * octave_sweep * swell
        + 0.35 * sub_padded
    ).astype(np.float32)
    return _normalize(combined, peak=0.88)


def _cinema_confirmation(sample_rate: int) -> np.ndarray:
    bip_duration = 0.065
    gap_duration = 0.055
    freq = 2050.0

    single_bip = (
        0.78 * _sine(freq, bip_duration, sample_rate)
        + 0.22 * _sine(freq * 2.0, bip_duration, sample_rate)
    ).astype(np.float32)
    single_bip = _apply_envelope(single_bip, sample_rate)
    gap = _silence(gap_duration, sample_rate)

    signal = np.concatenate(
        (
            single_bip,
            gap,
            single_bip,
            gap,
            single_bip,
        )
    ).astype(np.float32)
    return _normalize(signal, peak=0.88)


def _cinema_capture(sample_rate: int) -> np.ndarray:
    # Target locked & homing in — accelerating lock tone + pitch-only close-in zip
    lock_duration = 0.58
    lock_count = int(lock_duration * sample_rate)
    t_lock = np.arange(lock_count, dtype=np.float32) / sample_rate
    progress = t_lock / max(lock_duration, 1e-6)

    pulse_phase = 2.0 * np.pi * (8.0 * t_lock + 0.5 * 18.0 * progress * t_lock)
    urgency_pulse = (0.45 + 0.55 * (0.5 + 0.5 * np.sin(pulse_phase))).astype(np.float32)

    carrier_sweep = _sweep(1320.0, 2380.0, lock_duration, sample_rate)
    harmonic_sweep = _sweep(2640.0, 4760.0, lock_duration, sample_rate)
    homing_tone = ((0.72 * carrier_sweep + 0.28 * harmonic_sweep) * urgency_pulse).astype(
        np.float32
    )

    # Pure pitch zip at the end without low-end thump or noise
    zip_duration = 0.14
    zip_pitch = (
        0.75 * _sweep(2380.0, 3400.0, zip_duration * 0.55, sample_rate)
    )
    zip_drop = 0.75 * _sweep(3400.0, 720.0, zip_duration * 0.45, sample_rate)
    pitch_tail = np.concatenate((zip_pitch, zip_drop)).astype(np.float32)

    combined = np.concatenate((homing_tone, pitch_tail)).astype(np.float32)
    return _normalize(combined, peak=0.90)


def build_tone(name: str) -> tuple[np.ndarray, int]:
    sample_rate = config.TONE_SAMPLE_RATE
    normalized = name.strip().lower()

    if normalized == "wake_up":
        signal = _cinema_wake_up(sample_rate)
    elif normalized == "confirmation":
        signal = _cinema_confirmation(sample_rate)
    elif normalized == "capture":
        signal = _cinema_capture(sample_rate)
    else:
        raise ValueError(f"unknown tone: {name}")

    return _apply_envelope(signal, sample_rate), sample_rate


def tone_wav_bytes(name: str) -> bytes:
    signal, sample_rate = build_tone(name)
    pcm = (np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def play_tone(name: str) -> bool:
    try:
        import sounddevice as sd
    except (ImportError, OSError):
        return False

    signal, sample_rate = build_tone(name)

    def _play() -> None:
        with _TONE_LOCK:
            try:
                sd.stop()
                sd.play(signal, sample_rate)
                sd.wait()
            except (sd.PortAudioError, RuntimeError, ValueError):
                return

    threading.Thread(target=_play, name=f"tone-{name}", daemon=True).start()
    return True