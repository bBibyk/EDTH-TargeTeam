import os
import sys
import numpy as np
import onnxruntime as ort


# ============================================================
# CONFIGURATION
# ============================================================

SR = 32000
WINDOW_SECONDS = 2
WINDOW_SAMPLES = SR * WINDOW_SECONDS

N_MELS = 128
WIN_LENGTH = 800
HOP_LENGTH = 320
N_FFT = 1024

CLASSES = [
    "background",
    "vehicle",
    "drone",
    "footsteps",
]

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = BASE_DIR

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "efficientat_audio_v1.onnx",
)

MEL_BASIS_PATH = os.path.join(
    MODEL_DIR,
    "mel_basis.npy",
)


# ============================================================
# PREPROCESSING
# ============================================================

WINDOW = np.hanning(
    WIN_LENGTH
).astype(np.float32)

MEL_BASIS = np.load(
    MEL_BASIS_PATH
).astype(
    np.float32,
    copy=False,
)


def lightweight_mel(waveform):

    x = np.asarray(
        waveform,
        dtype=np.float32,
    )

    if x.ndim != 1:
        raise ValueError(
            f"Waveform must be 1-D, got {x.shape}"
        )

    if len(x) != WINDOW_SAMPLES:
        raise ValueError(
            f"Expected {WINDOW_SAMPLES} samples, "
            f"got {len(x)}"
        )

    # Pre-emphasis
    x = (
        x[1:]
        - np.float32(0.97) * x[:-1]
    )

    # Equivalent to torch.stft(center=True)
    x = np.pad(
        x,
        (
            N_FFT // 2,
            N_FFT // 2,
        ),
        mode="reflect",
    )

    frames = (
        np.lib.stride_tricks
        .sliding_window_view(
            x,
            N_FFT,
        )[::HOP_LENGTH]
    )

    padded_window = np.zeros(
        N_FFT,
        dtype=np.float32,
    )

    start = (
        N_FFT - WIN_LENGTH
    ) // 2

    padded_window[
        start:start + WIN_LENGTH
    ] = WINDOW

    frames = (
        frames
        * padded_window[None, :]
    )

    spectrum = np.fft.rfft(
        frames,
        n=N_FFT,
        axis=1,
    )

    power = (
        spectrum.real * spectrum.real
        + spectrum.imag * spectrum.imag
    ).astype(np.float32)

    power = power.T

    melspec = (
        MEL_BASIS @ power
    )

    melspec = np.log(
        melspec
        + np.float32(1e-5)
    )

    melspec = (
        melspec
        + np.float32(4.5)
    ) / np.float32(5.0)

    return np.asarray(
        melspec[
            np.newaxis,
            np.newaxis,
            :,
            :
        ],
        dtype=np.float32,
    )


# ============================================================
# ONNX
# ============================================================

options = ort.SessionOptions()

options.intra_op_num_threads = 1
options.inter_op_num_threads = 1

SESSION = ort.InferenceSession(
    MODEL_PATH,
    sess_options=options,
    providers=["CPUExecutionProvider"],
)


# ============================================================
# INFERENCE
# ============================================================

def softmax(logits):

    x = np.asarray(
        logits,
        dtype=np.float32,
    )

    x = x - np.max(x)

    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x)


def predict(waveform):

    mel = lightweight_mel(
        waveform
    )

    logits = SESSION.run(
        ["logits"],
        {
            "mel": mel,
        },
    )[0][0]

    probabilities = softmax(
        logits
    )

    index = int(
        np.argmax(probabilities)
    )

    return {
        "class": CLASSES[index],
        "class_index": index,
        "probabilities": {
            name: float(probabilities[i])
            for i, name in enumerate(CLASSES)
        },
        "vehicle_probability": float(
            probabilities[
                CLASSES.index("vehicle")
            ]
        ),
    }


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    waveform = np.zeros(
        WINDOW_SAMPLES,
        dtype=np.float32,
    )

    result = predict(
        waveform
    )

    print("Model:", MODEL_PATH)
    print("Mel basis:", MEL_BASIS_PATH)
    print("Input samples:", WINDOW_SAMPLES)
    print("Classes:", CLASSES)
    print()
    print("Self-test:")
    print(result)
