# Drone audio detector

Listens to 5 s of audio and says how likely it is to be a **drone**, a **vehicle** (cars, trucks, motorbikes,
trains, airplanes, helicopters, military vehicles) or **other** (everything else). It runs on a CPU with only `numpy` and
`onnxruntime`. It needs no PyTorch or GPU, and it runs on a Raspberry Pi.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

./run_live.sh                                    # live microphone view
./run_live.sh --plain                            # one line per score
./run_live.sh --list-devices                     # pick a mic, then --device N
.venv/bin/python -m drone_audio.detect clip.wav  # score a recording, one line per window + summary
```

* Input is any 16-bit PCM wav (any sample rate, mono or stereo). Convert other formats first with
  `ffmpeg -i in.mp3 -ac 1 -ar 16000 -sample_fmt s16 out.wav`.
* On Linux without PortAudio, the live view falls back to `parecord` / `arecord` automatically.
* Useful options: `--threshold 0.4` (more sensitive), `--hop 0.25` (score more often), `--csv scores.csv`.
  `THREADS=4 ./run_live.sh` sets the CPU threads.

## In your own code

```python
import sys; sys.path.insert(0, "path/to/drone_audio_detector")
from drone_audio.native import load_native_model, CLASSES   # CLASSES = ("drone", "vehicle", "other")

model = load_native_model("path/to/drone_audio_detector/model", threads=2)
p_drone, p3 = model(window)   # window: float32 numpy array, 80 000 samples = 5 s at 16 kHz, mono, range [-1, 1]
if p_drone >= model.thr:      # threshold 0.509, picked on the validation set
    print("drone")
```

`p3` holds the three class probabilities and sums to 1. Feed it overlapping windows, for example a new one every
0.5 s, as the live view does.

## The model

| | |
|---|---|
| network | CED-Small audio transformer (22 M parameters), pretrained on AudioSet, whole network fine-tuned |
| input | 5 s, 16 kHz mono → 64-band mel spectrogram (numpy, `drone_audio/native.py`) → `model/ced_small_ft.onnx` |
| speed | ~100 ms per window on a laptop CPU with 2 threads, including the spectrogram |
| checkpoint | epoch 26 of 30 (best validation drone AUC), 3-class head |

Training data: about 18 k five-second windows from 13 public sets. Drones came from ground-recorded delivery flights,
Freesound recordings and drone datasets. Vehicles came from MAD military audio, YouTube vehicle sounds and ESC-50 /
UrbanSound8K engines. Other covered explosions, gunshots, weather, animals, speech, sirens, machinery and forest sounds.
Machinery such as chainsaws, drills, air conditioners and generators counts as **other**, not vehicle.

### Results on held-out data

The test set has whole recordings never seen in training: 55 drone, 600 vehicle and 1800 other windows.

| | drone AUC | drone F1 | drone recall | false drone alarms | 3-class accuracy |
|---|---|---|---|---|---|
| test | 0.986 | 0.889 | 0.87 | 0.21 % | 95.4 % |
| unseen dataset (Svanström: 60 drone, 60 background) | 0.988 | 0.723 | 0.57 | 0 % | 80.8 % |

Test confusion matrix (rows = truth):

| | → drone | → vehicle | → other |
|---|---|---|---|
| drone | 48 | 0 | 7 |
| vehicle | 3 | 543 | 54 |
| other | 4 | 45 | 1751 |

Full numbers are in `model/metrics.json`.

### Limits

* There are few drone test windows (55 + 60), so the drone scores are only accurate to a few points either way.
* On recordings unlike the training data (the Svanström set), it catches only about 6 in 10 drone windows at the
  default threshold, though it ranks them well (AUC 0.99). In a new environment, try `--threshold 0.3`.
* A drone far away, or masked by wind or engines, scores lower. Use a streak of several windows above the
  threshold, not a single window, before acting on it.
* It hasn't been tested in the field. Check it on recordings from your own microphone and site.

## Licences: read before using it outside the hackathon

* **The model weights are GPL-3.0.** They are derived from CED (github.com/RicherMans/CED, GPL-3.0).
* **Some training data is non-commercial only:** ESC-50, UrbanSound8K, FSC22 and the Freesound drone compilation are
  CC BY-NC. Some sets have unclear licences (merged-drone, YouTube vehicle sounds).
* **Some training data forbids military or defence use:** the Al-Emadi DroneAudioDataset and DADS, which contains it.
  These were used for a hackathon prototype only. **Don't use this model for real defence, military or targeting
  work.** Retrain without those sets first.
