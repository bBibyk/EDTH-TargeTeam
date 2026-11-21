"""Audio wake model wrapper.

`AUDIO_MODEL_PATH` (default `models/audio/raspberry_pi`) can be:
- a folder with an .onnx + `mel_basis.npy` -> the EfficientAT classifier (efficientat.py),
- a single .onnx with the older spectrogram-image contract (see models/audio/README.md).
If neither exists, a clearly-labelled RMS-energy placeholder is used so the
wake -> detector chain can be tested end to end.
"""
from __future__ import annotations

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


def load_audio_wake_model(
    path: Path, threads: int = 4, wake_classes: tuple[str, ...] = ("drone",)
) -> AudioWakeModel:
    from demo.raspberry.audio_wake.efficientat import EfficientAtModel, find_efficientat_onnx

    efficientat = find_efficientat_onnx(path)  # folder with mel_basis.npy (current model)
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
