# Audio wake model

Put the trained model here as **`audio_wake.onnx`** (or set `AUDIO_MODEL_PATH`).
It is loaded by `demo/raspberry/audio_wake/model.py` and run by
`demo/raspberry/audio_wake/main.py`, which sends the wake signal to the detector.

Until this file exists, the program falls back to an RMS-energy placeholder that
wakes on any loud sound. Use it only to test the plumbing.

## Expected contract (same as the Challenge 5 scorer / c5lab `train.py` export)

| | |
|---|---|
| audio in | mono, 16 kHz, 5 s sliding window |
| input | `spectrogram`: float32 `[1, 3, 224, 224]` in [0, 1], built by `audio_wake/frontend.py` |
| output | `wake_logit` (sigmoid -> wake probability); other outputs are ignored |
| metadata | optional `wake_threshold` (used unless `AUDIO_WAKE_THRESHOLD` / `--threshold` is set) |

If the model is trained with a different input, change `frontend.py` and
`OnnxAudioWakeModel.predict` to match.
