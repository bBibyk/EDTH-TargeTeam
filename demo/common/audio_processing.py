from __future__ import annotations

import numpy as np


def pcm16_to_float(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    return values / 32768.0


def downsample(values: np.ndarray, max_points: int = 2000) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    if values.size <= max_points:
        return values
    step = int(np.ceil(values.size / max_points))
    return values[::step]


def _hz_to_mel(frequency: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    mel_points = np.linspace(
        _hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2, dtype=np.float64
    )
    hz_points = _mel_to_hz(mel_points)
    fft_bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    fft_bins = np.clip(fft_bins, 0, n_fft // 2)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = fft_bins[index : index + 3]
        if center > left:
            filters[index, left:center] = np.linspace(
                0.0, 1.0, center - left, endpoint=False, dtype=np.float32
            )
        if right > center:
            filters[index, center:right] = np.linspace(
                1.0, 0.0, right - center, endpoint=False, dtype=np.float32
            )
    return filters


def compute_mel_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    values = pcm16_to_float(samples)
    if values.size < n_fft:
        return None

    frames = np.lib.stride_tricks.sliding_window_view(values, n_fft)[::hop_length]
    if frames.size == 0:
        return None

    window = np.hanning(n_fft).astype(np.float32)
    spectrum = np.fft.rfft(frames * window, axis=1)
    power = np.abs(spectrum) ** 2

    filters = _mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax)
    mel_power = power @ filters.T
    mel_db = 10.0 * np.log10(np.maximum(mel_power, 1e-10))

    maximum = float(mel_db.max())
    if not np.isfinite(maximum):
        return None
    mel_db = mel_db - maximum

    times = np.arange(mel_db.shape[0]) * hop_length / sample_rate
    mel_points = np.linspace(
        _hz_to_mel(fmin), _hz_to_mel(fmax), n_mels, dtype=np.float64
    )
    frequencies = np.asarray(_mel_to_hz(mel_points), dtype=np.float64)
    return times, frequencies, mel_db.T
