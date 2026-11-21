"""Spectrogram front-end for the audio wake model (numpy + pillow only, runs on the Pi).

Reproduces the Challenge 5 organiser spectrogram that the model is trained on:
Slaney mel, 128 bands 50-8000 Hz, n_fft 1024, hop 256, dB ref=max, top_db 80,
uint8, low frequencies at the bottom, bilinear resize to 224x224.
Output: float32 [3, 224, 224] in [0, 1].

If the audio model is trained with different preprocessing, change it here.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

SR, N_FFT, HOP, N_MELS, FMIN, FMAX, TOP_DB, SIZE = 16000, 1024, 256, 128, 50.0, 8000.0, 80.0, 224


def _hz_to_mel(f):
    f = np.asarray(f, float)
    lin = f / (200.0 / 3)
    lo = 1000.0 / (200.0 / 3)
    return np.where(f >= 1000, lo + np.log(np.maximum(f, 1e-9) / 1000.0) / (np.log(6.4) / 27.0), lin)


def _mel_to_hz(m):
    m = np.asarray(m, float)
    lo = 1000.0 / (200.0 / 3)
    return np.where(m >= lo, 1000.0 * np.exp((m - lo) * (np.log(6.4) / 27.0)), m * 200.0 / 3)


def slaney_mel_fb() -> np.ndarray:
    """Same as librosa.filters.mel(sr=16000, n_fft=1024, n_mels=128, fmin=50, fmax=8000)."""
    fftfreqs = np.linspace(0, SR / 2, N_FFT // 2 + 1)
    pts = _mel_to_hz(np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), N_MELS + 2))
    fdiff = np.diff(pts)
    ramps = pts[:, None] - fftfreqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    w = np.maximum(0, np.minimum(lower, upper))
    w *= (2.0 / (pts[2:N_MELS + 2] - pts[:N_MELS]))[:, None]
    return w.astype(np.float32)


class SpecFrontend:
    def __init__(self) -> None:
        self.fb = slaney_mel_fb()
        n = np.arange(N_FFT)
        self.win = (0.5 - 0.5 * np.cos(2 * np.pi * n / N_FFT)).astype(np.float32)  # periodic hann

    def uint8(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32)
        xp = np.pad(x, N_FFT // 2)  # center=True, constant padding
        nfr = 1 + (len(xp) - N_FFT) // HOP
        idx = np.arange(N_FFT)[None, :] + HOP * np.arange(nfr)[:, None]
        frames = xp[idx] * self.win
        S = np.abs(np.fft.rfft(frames, axis=1)).T.astype(np.float32) ** 2
        M = 10 * np.log10(np.maximum(self.fb @ S, 1e-10))
        M = np.maximum(M - M.max(), -TOP_DB)
        I = ((M + TOP_DB) / TOP_DB * 255).astype(np.uint8)[::-1]
        return np.asarray(Image.fromarray(I).resize((SIZE, SIZE), Image.BILINEAR))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        g = self.uint8(x).astype(np.float32) / 255.0
        return np.repeat(g[None], 3, 0)
