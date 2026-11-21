from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

SR = 32000
WINDOW_SECONDS = 2
WINDOW_SAMPLES = SR * WINDOW_SECONDS
N_MELS = 128
WIN_LENGTH = 800
HOP_LENGTH = 320
N_FFT = 1024
CLASSES = ["background", "vehicle", "drone", "footsteps"]


class EfficientATAudioModel:
    def __init__(self, model_path: str | Path, mel_basis_path: str | Path) -> None:
        self._window = np.hanning(WIN_LENGTH).astype(np.float32)
        self._mel_basis = np.load(str(mel_basis_path)).astype(np.float32, copy=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def _mel(self, waveform: np.ndarray) -> np.ndarray:
        x = np.asarray(waveform, dtype=np.float32)
        if x.shape[0] != WINDOW_SAMPLES:
            raise ValueError(f"Expected {WINDOW_SAMPLES} samples, got {x.shape[0]}")
        x = x[1:] - np.float32(0.97) * x[:-1]
        x = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP_LENGTH]
        padded_window = np.zeros(N_FFT, dtype=np.float32)
        start = (N_FFT - WIN_LENGTH) // 2
        padded_window[start : start + WIN_LENGTH] = self._window
        frames = frames * padded_window[None, :]
        spectrum = np.fft.rfft(frames, n=N_FFT, axis=1)
        power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag).astype(
            np.float32
        )
        power = power.T
        melspec = self._mel_basis @ power
        melspec = np.log(melspec + np.float32(1e-5))
        melspec = (melspec + np.float32(4.5)) / np.float32(5.0)
        return melspec[np.newaxis, np.newaxis, :, :].astype(np.float32)

    def predict(self, waveform: np.ndarray) -> dict[str, Any]:
        mel = self._mel(waveform)
        logits = self._session.run(["logits"], {"mel": mel})[0][0]
        x = np.asarray(logits, dtype=np.float32)
        x = x - np.max(x)
        exp_x = np.exp(x)
        probs = exp_x / np.sum(exp_x)
        index = int(np.argmax(probs))
        return {
            "class": CLASSES[index],
            "class_index": index,
            "probabilities": {name: float(probs[i]) for i, name in enumerate(CLASSES)},
            "vehicle_probability": float(probs[CLASSES.index("vehicle")]),
        }
