"""EfficientAT audio classifier exported by the training team (models/audio/raspberry_pi).

The folder holds `efficientat_audio_v1.onnx` (+ `.onnx.data`), `mel_basis.npy` and
their reference script `inference_pi.py`. The front-end below is the same maths as
`inference_pi.lightweight_mel`: pre-emphasis 0.97, 1024-point STFT (800-sample Hann
window, hop 320, reflect padding), 128 mel bands from `mel_basis.npy`, log, (x + 4.5) / 5.

Contract: mono float32 in [-1, 1] at 32 kHz, exactly 2 s (64 000 samples)
-> input `mel` [1, 1, 128, 200] -> output `logits` [1, 4] for
background / vehicle / drone / footsteps (softmax -> probabilities).

The wake score is the summed probability of the chosen wake classes (default: drone).
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from demo.raspberry.audio_wake.model import AudioWakeModel

SR = 32000
WINDOW_SECONDS = 2.0
WINDOW_SAMPLES = int(SR * WINDOW_SECONDS)
N_FFT, WIN_LENGTH, HOP_LENGTH = 1024, 800, 320
CLASSES = ("background", "vehicle", "drone", "footsteps")
MEL_BASIS_FILE = "mel_basis.npy"


def find_efficientat_onnx(path: Path) -> Path | None:
    """`path` is the model folder or its .onnx file; it needs mel_basis.npy next to it."""
    folder = path if path.is_dir() else path.parent
    if not (folder / MEL_BASIS_FILE).is_file():
        return None
    if path.is_file() and path.suffix == ".onnx":
        return path
    if path.is_dir():
        found = sorted(folder.glob("*.onnx"))
        if len(found) == 1:
            return found[0]
    return None


class EfficientAtModel(AudioWakeModel):
    sample_rate = SR
    window_seconds = WINDOW_SECONDS
    classes = CLASSES

    def __init__(self, onnx_path: Path, wake_classes: Sequence[str], threads: int = 1) -> None:
        import onnxruntime as ort

        unknown = [c for c in wake_classes if c not in CLASSES]
        if unknown or not wake_classes:
            raise ValueError(f"wake classes must be some of {CLASSES}, got {list(wake_classes)}")
        self.wake_classes = tuple(wake_classes)
        self._wake_index = [CLASSES.index(c) for c in self.wake_classes]

        self._mel_basis = np.load(onnx_path.parent / MEL_BASIS_FILE).astype(np.float32)
        window = np.zeros(N_FFT, dtype=np.float32)
        start = (N_FFT - WIN_LENGTH) // 2
        window[start:start + WIN_LENGTH] = np.hanning(WIN_LENGTH).astype(np.float32)
        self._window = window

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        self._output = self._session.get_outputs()[0].name
        self.name = f"{onnx_path.stem}[{'+'.join(self.wake_classes)}]"
        self.last_probabilities: dict[str, float] = {}

    def mel(self, waveform: np.ndarray) -> np.ndarray:
        x = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if len(x) != WINDOW_SAMPLES:
            raise ValueError(f"expected {WINDOW_SAMPLES} samples, got {len(x)}")
        x = x[1:] - np.float32(0.97) * x[:-1]
        x = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP_LENGTH]
        spectrum = np.fft.rfft(frames * self._window[None, :], n=N_FFT, axis=1)
        power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag).astype(np.float32)
        mel = np.log(self._mel_basis @ power.T + np.float32(1e-5))
        mel = (mel + np.float32(4.5)) / np.float32(5.0)
        return mel[np.newaxis, np.newaxis].astype(np.float32, copy=False)

    def probabilities(self, waveform: np.ndarray) -> np.ndarray:
        logits = self._session.run([self._output], {self._input: self.mel(waveform)})[0][0]
        e = np.exp(logits - np.max(logits))
        return e / e.sum()

    def predict(self, samples: np.ndarray) -> float:
        p = self.probabilities(np.asarray(samples, dtype=np.float32)[-WINDOW_SAMPLES:])
        self.last_probabilities = {c: float(v) for c, v in zip(CLASSES, p)}
        return float(p[self._wake_index].sum())
