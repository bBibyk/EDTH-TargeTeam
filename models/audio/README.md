# Audio model

The current model is in **`raspberry_pi/`**. It came from `raspberry_pi_audio_v1.tar.gz`.
`demo/raspberry/audio_wake/main.py` loads it through
`demo/raspberry/audio_wake/efficientat.py`. That front-end is the same maths as
`raspberry_pi/inference_pi.py`, and its output was checked against it on the Pi
(identical mel, probabilities equal within 2e-7).

| File | What it is |
|---|---|
| `efficientat_audio_v1.onnx` + `.onnx.data` | EfficientAT classifier. Keep both files together |
| `mel_basis.npy` | 128 x 513 mel filterbank, 31 Hz – 15 kHz |
| `inference_pi.py` | The training team's reference script. `python inference_pi.py` runs a self-test |
| `SHA256SUMS` | `sha256sum -c SHA256SUMS` (Linux) or `shasum -a 256 -c SHA256SUMS` (macOS) |

## Contract

| | |
|---|---|
| audio in | mono, **32 kHz**, float32 in [-1, 1] (int16 / 32768), exactly **2 s** (64 000 samples) |
| input | `mel`: float32 `[1, 1, 128, 200]` (pre-emphasis 0.97, STFT 1024 / win 800 / hop 320, log-mel, (x + 4.5) / 5) |
| output | `logits`: `[1, 4]` for `background, vehicle, drone, footsteps`. Softmax turns them into probabilities |
| wake score | sum of the probabilities of `AUDIO_WAKE_CLASSES` (default `drone`), compared with `AUDIO_WAKE_THRESHOLD` (default 0.5) |

Because the filterbank goes up to 15 kHz, the audio must really be sampled at 32 kHz.
Audio recorded at 16 kHz and upsampled has nothing above 8 kHz, so the model sees
something it was not trained on.

## Run it on the Pi

```bash
~/yolo26/bin/pip install onnxruntime sounddevice    # plus: sudo apt install libportaudio2
./demo/scripts/run_audio_wake.sh --verbose           # live mic, prints every class's probability
./demo/scripts/run_audio_wake.sh --wav clip.wav      # replay a 16-bit WAV (resampled to 32 kHz)
```

On the Pi 5, one 2 s window takes about 4 ms for the mel and 4–6 ms for the model on one or two cores.
The program scores the last 2 s every `AUDIO_WAKE_HOP_SECONDS` (0.5 s).
The mic is `AUDIO_WAKE_DEVICE`; empty means the system default, through PipeWire.
`AUDIO_WAKE_DEVICE=stm32` (or `--device stm32`) reads the STM32 board, which streams exactly
32 kHz mono int16 over its ST-LINK USB port. See [stm32/mic_stream](../../stm32/mic_stream/README.md).

Known quirks to check with real recordings: pure digital silence comes out as
`footsteps` (0.67), and quiet white noise is only about 0.5–0.7 `background`.

## Older contract (still supported)

Point `AUDIO_MODEL_PATH` at a single `.onnx` that has no `mel_basis.npy` next to it.
The model must take mono 16 kHz audio in a 5 s window. Its input is `spectrogram`, float32
`[1, 3, 224, 224]` in [0, 1], built by `audio_wake/frontend.py`; its output is
`wake_logit`. If `AUDIO_MODEL_PATH` points at nothing, an RMS-loudness placeholder is
used instead. It is only for testing the plumbing.
