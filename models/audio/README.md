# Audio wake model

## `native_knn_mn30/` — drone detector (default)

Scores each 5 s window (16 kHz mono) and returns **P(drone_buzz)**. The wake signal is sent when
P(drone_buzz) ≥ 0.404, a threshold picked out-of-fold. It is loaded by
`demo/raspberry/audio_wake/model.py` (`NativeKnnWakeModel`) and is the default in
`demo/scripts/run_audio_wake.sh`.

| | |
|---|---|
| pipeline | numpy log-mel frontend (`audio_wake/native.py`) → EfficientAT **mn30** (AudioSet-pretrained, frozen, ONNX FP32) → 5760-d embedding → cosine kNN (k=5) over the 56 labelled challenge clips (`knn.npz`) |
| classes in the bank | drone_buzz, vehicle_mechanical, explosion, ambient_outdoor; only the drone_buzz vote is used for the wake |
| CV (5-fold × 2, seeds 30/40/50) | wake AUC **0.83 ± 0.02**, F1 0.72, 4-class acc 0.73 (single window: 0.80 / 0.70 / 0.71) |
| speed | ~70 ms per window on 4 x86 threads (frontend + net + kNN). Not yet measured on the Pi 5 |
| files | `mn30_native_net.onnx` + `.data0..2` (156 MB of weights split into files under 100 MB for GitHub; keep them together), `knn.npz`, `model.json` (export settings) |
| deps | `numpy`, `onnxruntime` |

Options (env or `demo/.env`):
* `AUDIO_WAKE_THREADS` (default 2): use 4 on a Pi 5 if the audio program has the CPU to itself.
* `AUDIO_WAKE_EMB_AVG` (default 1): averages the last N window embeddings. The score is steadier but reacts more slowly.
* `AUDIO_WAKE_THRESHOLD`: overrides 0.404.

The numbers come from only 56 clips, so treat them as rough estimates. Built in c5lab with
`export_native_knn.py --arch mn30`.

## Other formats

`AUDIO_MODEL_PATH` can also point to an `.onnx` CNN that takes the organiser spectrogram
(`spectrogram` float32 `[1,3,224,224]` in [0,1] → `wake_logit`, optional `wake_threshold` metadata).
If no model exists, an RMS-energy placeholder wakes on any loud sound. Use it only to test the plumbing.
