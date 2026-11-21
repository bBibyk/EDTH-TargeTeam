"""numpy-only Pi pieces for the native-frontend EfficientAT + cosine-kNN wake model (no torch needed).

  NativeFrontend : 16 kHz waveform -> EfficientAT log-mel [1,1,128,T] (twin of effat_native.NativeFrontend,
                   matches torch to ~5e-6)
  KnnModel       : <dir>/knn.npz (bank, labels, weights, standardization, k, temp, thr, net file)
                   + <dir>/<net>.onnx (input "logmel", output "embedding")

Copied from c5lab (pi_native.py); built there by export_native_knn.py. Used by
demo/raspberry/audio_wake/model.py when the model path is a directory holding knn.npz.
"""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np

N_CLASSES = 4  # drone_buzz, vehicle_mechanical, explosion, ambient_outdoor; bucket 4 = unlabeled "other"


def kaldi_mel_banks(n_mels=128, n_fft=1024, sr=32000, fmin=0.0, fmax=15000.0) -> np.ndarray:
    mel = lambda f: 1127.0 * np.log(1.0 + np.asarray(f, float) / 700.0)  # noqa: E731
    fft_mel = mel(sr / n_fft * np.arange(n_fft // 2))
    lo, hi = mel(fmin), mel(fmax)
    delta = (hi - lo) / (n_mels + 1)
    b = np.arange(n_mels)[:, None]
    left, center, right = lo + b * delta, lo + (b + 1) * delta, lo + (b + 2) * delta
    W = np.maximum(0.0, np.minimum((fft_mel[None] - left) / (center - left), (right - fft_mel[None]) / (right - center)))
    return np.pad(W, ((0, 0), (0, 1))).astype(np.float32)


class NativeFrontend:
    def __init__(self):
        self.mel = kaldi_mel_banks()
        n = np.arange(800)
        self.win = np.zeros(1024, np.float32)
        self.win[112:912] = 0.5 - 0.5 * np.cos(2 * np.pi * n / 799)

    def __call__(self, x16: np.ndarray) -> np.ndarray:
        x16 = np.asarray(x16, np.float32)
        n = len(x16)
        pos = np.clip((np.arange(2 * n) + 0.5) / 2.0 - 0.5, 0, n - 1)
        i0 = np.floor(pos).astype(int)
        i1 = np.minimum(i0 + 1, n - 1)
        x = x16[i0] * (1 - (pos - i0)) + x16[i1] * (pos - i0)
        x = x[1:] - 0.97 * x[:-1]
        xp = np.pad(x, 512, mode="reflect")
        nfr = 1 + (len(xp) - 1024) // 320
        idx = np.arange(1024)[None, :] + 320 * np.arange(nfr)[:, None]
        S = np.abs(np.fft.rfft(xp[idx] * self.win, axis=1)).T ** 2
        return ((np.log(self.mel @ S + 1e-5) + 4.5) / 5.0).astype(np.float32)[None, None]


def knn_scores(bank, labels, weights, z, k, temp):
    """bank: standardized + L2-normalized [N,d]; z: standardized + L2-normalized [d] -> 5 vote fractions."""
    sim = bank @ z
    nn = np.argpartition(-sim, k - 1)[:k]
    V = np.zeros(N_CLASSES + 1)
    np.add.at(V, labels[nn], weights[nn] * np.exp(temp * sim[nn]))
    return V / V.sum()


class KnnModel:
    """Callable: 5 s waveform -> (p_drone, p4). emb_avg>1 averages the last N window embeddings (streaming
    analogue of the shift-averaging used in CV)."""

    def __init__(self, model_dir, threads=4, emb_avg=1, session_factory=None):
        d = Path(model_dir)
        h = np.load(d / "knn.npz", allow_pickle=True)
        self.mu, self.sd = h["mean"], h["scale"]
        self.bank, self.labels, self.weights = h["bank"], h["labels"].astype(int), h["weights"]
        self.k, self.temp, self.thr = int(h["k"]), float(h["temp"]), float(h["thr"])
        if session_factory is None:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            session_factory = lambda p: ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])  # noqa
        self.sess = session_factory(d / str(h["net"]))
        self.fe = NativeFrontend()
        self.hist = collections.deque(maxlen=max(1, emb_avg))

    def embed(self, x):
        return self.sess.run(["embedding"], {"logmel": self.fe(x)})[0][0]

    def __call__(self, x):
        self.hist.append(self.embed(x))
        z = (np.mean(self.hist, 0) - self.mu) / self.sd
        z /= np.linalg.norm(z) + 1e-9
        V = knn_scores(self.bank, self.labels, self.weights, z, self.k, self.temp)
        p4 = V[:N_CLASSES] + 1e-9
        return float(V[0]), p4 / p4.sum()
