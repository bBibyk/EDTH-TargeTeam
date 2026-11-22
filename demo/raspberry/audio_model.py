from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

SR = 16000
WINDOW_SECONDS = 5
WINDOW_SAMPLES = SR * WINDOW_SECONDS
CLASSES = ["drone", "vehicle", "other"]


def _load_native(model_path: str | Path, threads: int = 1):
    path = Path(model_path)
    detector_root = path
    for parent in [path, *path.parents]:
        if (parent / "drone_audio" / "native.py").is_file():
            detector_root = parent
            break
    if str(detector_root) not in sys.path:
        sys.path.insert(0, str(detector_root))
    from drone_audio.native import load_native_model

    model_dir = path
    if (path / "model.json").is_file():
        model_dir = path
    elif (path.parent / "model.json").is_file():
        model_dir = path.parent
    return load_native_model(model_dir, threads=threads)


class DroneAudioModel:
    """Adapter around drone_audio_detector/native.py with the legacy predict() dict contract."""

    sample_rate = SR
    window_seconds = WINDOW_SECONDS
    window_samples = WINDOW_SAMPLES
    classes = tuple(CLASSES)

    def __init__(
        self, model_path: str | Path, mel_basis_path: str | Path | None = None, threads: int = 1
    ) -> None:
        self._model = _load_native(model_path, threads=threads)
        self.threshold = float(getattr(self._model, "thr", 0.5))
        self.name = Path(model_path).name

    def predict(self, waveform: np.ndarray) -> dict[str, Any]:
        x = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if x.shape[0] != WINDOW_SAMPLES:
            if x.shape[0] > WINDOW_SAMPLES:
                x = x[-WINDOW_SAMPLES:]
            else:
                x = np.pad(x, (0, WINDOW_SAMPLES - x.shape[0]))
        p_drone, p3 = self._model(x)
        p3 = np.asarray(p3, dtype=np.float32).reshape(-1)
        probs = {name: float(p3[i]) for i, name in enumerate(CLASSES)}
        index = int(np.argmax(p3))
        return {
            "class": CLASSES[index],
            "class_index": index,
            "probabilities": probs,
            "drone_probability": float(p_drone),
            "vehicle_probability": float(probs["vehicle"]),
            "wake": bool(float(p_drone) >= self.threshold),
        }


# Backward-compatible name used by older imports.
EfficientATAudioModel = DroneAudioModel
