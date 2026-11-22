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

CLASSES = ("drone", "vehicle", "other")  # what every model reports (p3); older models are folded into these
FINE_CLASSES = ("drone_buzz", "vehicle_mechanical", "explosion", "ambient_outdoor", "other")  # kNN banks, 5-class models
_COARSE = {"drone_buzz": "drone", "vehicle_mechanical": "vehicle", "explosion": "other", "ambient_outdoor": "other"}


def fold_matrix(classes) -> np.ndarray:
    """[len(classes), 3] 0/1 matrix: p3 = p @ M (explosion / ambient / other all -> other)."""
    M = np.zeros((len(classes), len(CLASSES)))
    for j, c in enumerate(classes):
        M[j, CLASSES.index(_COARSE.get(c, c if c in CLASSES else "other"))] = 1.0
    return M


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
    V = np.zeros(len(FINE_CLASSES))
    np.add.at(V, labels[nn], weights[nn] * np.exp(temp * sim[nn]))
    return V / V.sum()


class KnnModel:
    """Callable: 5 s waveform -> (p_drone, p3 over CLASSES). emb_avg>1 averages the last N window embeddings (streaming
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
        return float(V[0]), V @ fold_matrix(FINE_CLASSES)


class FinetunedModel:
    """Fine-tuned backbone (model_training/kaggle/finetune_mn30.py, mn30 or CED): <dir>/model.json {"type": "finetuned"}
    + <dir>/<net>.onnx (input "logmel", output "probs" [1, len(classes)]). Same callable contract as KnnModel:
    5 s waveform -> (p_drone, p3). emb_avg>1 averages the last N probability vectors."""

    k = "finetuned"

    def __init__(self, model_dir, threads=4, emb_avg=1, session_factory=None):
        import json

        d = Path(model_dir)
        meta = json.load(open(d / "model.json"))
        self.thr = float(meta["thr"])
        if session_factory is None:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            session_factory = lambda p: ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])  # noqa
        self.sess = session_factory(d / meta["net"])
        # "effat_mel128" (mn30, default for older model.json) or "ced_mel64" (CED); CedFrontend is defined below
        self.fe = CedFrontend() if meta.get("frontend") == "ced_mel64" else NativeFrontend()
        self.arch = meta.get("arch", "mn30")
        self.fold = fold_matrix(meta.get("classes", FINE_CLASSES))  # 3-class models: identity
        self.hist = collections.deque(maxlen=max(1, emb_avg))

    def __call__(self, x):
        self.hist.append(self.sess.run(["probs"], {"logmel": self.fe(x)})[0][0])
        p = np.mean(self.hist, 0)
        return float(p[0]), p @ self.fold


def load_native_model(model_dir, threads=4, emb_avg=1):
    """Directory with knn.npz -> KnnModel; with model.json {"type": "finetuned"} -> FinetunedModel."""
    import json

    d = Path(model_dir)
    if (d / "model.json").exists() and json.load(open(d / "model.json")).get("type") == "finetuned":
        return FinetunedModel(d, threads=threads, emb_avg=emb_avg)
    return KnnModel(d, threads=threads, emb_avg=emb_avg)


def htk_mel_banks(n_mels=64, n_fft=512, sr=16000, fmin=0.0, fmax=None) -> np.ndarray:
    """numpy twin of torchaudio.functional.melscale_fbanks(mel_scale="htk", norm=None) -> [n_mels, n_fft//2+1]"""
    fmax = sr / 2 if fmax is None else fmax
    freqs = np.linspace(0, sr // 2, n_fft // 2 + 1)
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)  # noqa: E731
    m_pts = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - freqs[:, None]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return np.maximum(0.0, np.minimum(down, up)).T.astype(np.float32)


class CedFrontend:
    """numpy twin of CED's FrontEnd (torchaudio MelSpectrogram 16 kHz, n_fft=win=512 periodic hann, hop 160,
    center reflect, power 2, 64 HTK mels, then AmplitudeToDB(top_db=120)): 16 kHz waveform -> [1, 64, T]."""

    def __init__(self, n_mels=64):
        self.mel = htk_mel_banks(n_mels=n_mels)
        n = np.arange(512)
        self.win = (0.5 - 0.5 * np.cos(2 * np.pi * n / 512)).astype(np.float32)

    def __call__(self, x16: np.ndarray) -> np.ndarray:
        x = np.pad(np.asarray(x16, np.float32), 256, mode="reflect")
        nfr = 1 + (len(x) - 512) // 160
        idx = np.arange(512)[None, :] + 160 * np.arange(nfr)[:, None]
        S = np.abs(np.fft.rfft(x[idx] * self.win, axis=1)).T ** 2
        db = 10.0 * np.log10(np.maximum(self.mel @ S, 1e-10))
        db = np.maximum(db, db.max() - 120.0)
        return db.astype(np.float32)[None]
