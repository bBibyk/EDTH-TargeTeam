"""Audio wake model wrapper.

`AUDIO_MODEL_PATH` can be:
- a drone_audio_detector model dir (`model.json` + .onnx) -> native.py wrapper,
- a legacy EfficientAT folder (.onnx + `mel_basis.npy`) -> efficientat.py,
- a single legacy .onnx with the older spectrogram-image contract.
If nothing is found, a clearly-labelled RMS-energy placeholder is used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from demo.raspberry.audio_wake.frontend import SR


class AudioWakeModel:
    """Scores one mono float32 window in [-1, 1] at `sample_rate` Hz -> wake probability."""

    sample_rate = SR
    threshold = 0.5
    name = "base"

    def predict(self, samples: np.ndarray) -> float:
        raise NotImplementedError


class OnnxAudioWakeModel(AudioWakeModel):
    def __init__(self, path: Path, threads: int = 4) -> None:
        import onnxruntime as ort

        from demo.raspberry.audio_wake.frontend import SpecFrontend

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        outputs = [o.name for o in self._session.get_outputs()]
        self._output = "wake_logit" if "wake_logit" in outputs else outputs[0]
        meta = self._session.get_modelmeta().custom_metadata_map
        if "wake_threshold" in meta:
            self.threshold = float(meta["wake_threshold"])
        self._frontend = SpecFrontend()
        self.name = path.name

    def predict(self, samples: np.ndarray) -> float:
        spec = self._frontend(samples)[None].astype(np.float32)
        logit = float(np.asarray(self._session.run([self._output], {self._input: spec})[0]).reshape(-1)[0])
        return float(1.0 / (1.0 + np.exp(-logit)))


class PlaceholderEnergyModel(AudioWakeModel):
    """NOT a drone detector: fires on loud sound. Only for testing the plumbing."""

    name = "placeholder-energy"

    def __init__(self, rms_dbfs_at_half: float = -30.0) -> None:
        self._mid = rms_dbfs_at_half

    def predict(self, samples: np.ndarray) -> float:
        tail = samples[-self.sample_rate // 2 :]  # last 0.5 s, so it reacts quickly
        rms = float(np.sqrt(np.mean(np.square(tail)) + 1e-12))
        dbfs = 20.0 * np.log10(rms)
        return float(1.0 / (1.0 + np.exp(-(dbfs - self._mid) / 3.0)))


def _find_drone_audio_detector_root(path: Path) -> Path | None:
    candidates = [path, *path.parents]
    for candidate in candidates:
        if (candidate / "drone_audio" / "native.py").is_file():
            return candidate
    return None


def _is_drone_audio_detector_model(path: Path) -> bool:
    if path.is_dir() and (path / "model.json").is_file():
        return True
    return path.is_file() and path.name == "model.json"


class DroneAudioDetectorWakeModel(AudioWakeModel):
    sample_rate = 16000
    window_seconds = 5.0

    def __init__(self, path: Path, threads: int = 4, wake_classes: tuple[str, ...] = ("drone",)) -> None:
        model_dir = path if path.is_dir() else path.parent
        root = _find_drone_audio_detector_root(model_dir)
        if root is None:
            raise FileNotFoundError(f"drone_audio/native.py not found near {path}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from drone_audio.native import load_native_model

        self._model = load_native_model(model_dir, threads=threads)
        meta_path = model_dir / "model.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        self.classes = tuple(meta.get("classes", ("drone", "vehicle", "other")))
        unknown = [c for c in wake_classes if c not in self.classes]
        if unknown or not wake_classes:
            raise ValueError(f"wake classes must be some of {self.classes}, got {list(wake_classes)}")
        self.wake_classes = tuple(wake_classes)
        self.threshold = float(getattr(self._model, "thr", meta.get("thr", 0.5)))
        self.name = f"drone_audio_detector[{'+'.join(self.wake_classes)}]"
        self.last_probabilities: dict[str, float] = {}

    def predict(self, samples: np.ndarray) -> float:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        needed = int(self.sample_rate * self.window_seconds)
        if x.shape[0] > needed:
            x = x[-needed:]
        elif x.shape[0] < needed:
            x = np.pad(x, (0, needed - x.shape[0]))
        p_drone, p3 = self._model(x)
        probs = [float(v) for v in np.asarray(p3, dtype=np.float32).reshape(-1)]
        self.last_probabilities = {c: probs[i] for i, c in enumerate(self.classes)}
        return float(sum(self.last_probabilities.get(c, 0.0) for c in self.wake_classes))


def load_audio_wake_model(
    path: Path, threads: int = 4, wake_classes: tuple[str, ...] = ("drone",)
) -> AudioWakeModel:
    from demo.raspberry.audio_wake.efficientat import EfficientAtModel, find_efficientat_onnx

    if _is_drone_audio_detector_model(path):
        return DroneAudioDetectorWakeModel(path, threads=threads, wake_classes=wake_classes)

    efficientat = find_efficientat_onnx(path)  # legacy folder with mel_basis.npy
    if efficientat is not None:
        return EfficientAtModel(efficientat, wake_classes=wake_classes, threads=threads)
    if path.is_file():
        return OnnxAudioWakeModel(path, threads=threads)
    print(
        f"WARNING: no audio model at {path} -> using the RMS-energy placeholder "
        f"(wakes on any loud sound, not on drones).",
        flush=True,
    )
    return PlaceholderEnergyModel()
