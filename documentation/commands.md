# Commands cheat sheet

Run the demo, test the audio model, and manage the Pi and the STM32 board.
How it all works: [architecture/demo-pipeline.md](../architecture/demo-pipeline.md).

| Machine | Runs | Started by |
|---|---|---|
| **STM32** (B-L475E-IOT01A) | Mic → raw 32 kHz audio over its USB (ST-LINK) cable | By itself when powered ([stm32/mic_stream](../stm32/mic_stream/README.md)) |
| **Raspberry Pi** | Reads the STM32 audio, runs the audio model; switches on camera + YOLO when it hears a vehicle/drone/footsteps; streams everything to the Mac | systemd service `edth-rbp` (starts at boot) |
| **Mac** | Web app: video, sound, detections, audio-model results | `./demo/scripts/run_pc.sh` |

Placeholders: `<PI_IP>` = the Pi's address, `<pi-user>` = the Pi's user name.
On the Pi the repository lives in `~/EDTH-TargeTeam`; all Pi commands below start there.

---

## 1. Connect to the Pi

Mac and Pi must be on the same network (same Wi‑Fi or phone hotspot).

```bash
# Easiest: finds the Pi on every network (USB, Wi-Fi, hotspot) and opens SSH
cd ~/Documents/Development/Hardware/RPVideo      # separate RPVideo repo
./mac/find-pi.sh            # only list where the Pi answers
./mac/find-pi.sh --ssh      # ...and connect

# By hand
ssh -i ~/.ssh/id_rsa_raspberry_pi -o HostKeyAlias=raspberrypi.local <pi-user>@<PI_IP>
ssh <pi-user>@raspberrypi.local                   # if the network passes .local names

# Addresses (they come from DHCP and can change)
hostname -I                  # on the Pi
ipconfig getifaddr en0       # on the Mac (Wi-Fi)
```

If an address changes: set `PC_HOST=<Mac IP>` in the Pi's `demo/.env` and
`RBP_HOST=<PI_IP>` in the Mac's `demo/.env`, then restart both sides (section 5).

Copy files:

```bash
# Pi -> Mac (run on the Mac), e.g. a recording
scp -i ~/.ssh/id_rsa_raspberry_pi <pi-user>@<PI_IP>:EDTH-TargeTeam/stm32_drone.wav .

# Mac -> Pi: push code changes (run on the Mac, from the repo root), then restart the service
rsync -a --exclude .git --exclude .venv --exclude __pycache__ --exclude demo/.env \
      --exclude stm32/mic_stream/build --exclude stm32/backup \
      -e "ssh -i ~/.ssh/id_rsa_raspberry_pi" ./ <pi-user>@<PI_IP>:EDTH-TargeTeam/
```

---

## 2. Run the whole system (STM32 sound + camera video → web app)

**Pi.** Plug in the STM32 board and the camera. The service is already set up
to use the STM32 (`AUDIO_INPUT_FORMAT=stm32` in the Pi's `demo/.env`):

```bash
systemctl --user restart edth-rbp
journalctl --user -u edth-rbp -f     # watch: "STM32 mic connected", "Video activated by audio model"
```

**Mac**, from the repo root:

```bash
./demo/scripts/run_pc.sh             # opens http://localhost:8501
```

What you should see:
- the audio panel updates twice a second, even without a camera;
- quiet room = `background` (the model only runs on sound louder than `AUDIO_RMS_WAKE_THRESHOLD`);
- a loud vehicle / drone / footsteps → video + YOLO boxes switch on, and off again 10 s after the last one.

To test without a drone, play the reference sample next to the STM32 (run on the Mac):

```bash
ffmpeg -v error -y -stream_loop 19 -i models/audio/drone_reference.wav -ar 32000 ~/drone_10s.wav
afplay ~/drone_10s.wav               # speaker close to the board and loud enough
```

---

## 3. STM32 sound only, on the Pi

Only one program can read the STM32 at a time, so stop the service first:

```bash
systemctl --user stop edth-rbp
cd ~/EDTH-TargeTeam
```

```bash
# Is the board sending? Expect ~32000 samples/s and "lost 0 bad 0"
~/yolo26/bin/python -m demo.raspberry.audio_wake.stm32_mic

# Audio model on the live STM32 sound, in the terminal (one line every 0.5 s)
./demo/scripts/run_audio_wake.sh --device stm32 --verbose
./demo/scripts/run_audio_wake.sh --device stm32 --verbose --threshold 0.9 --min-consecutive 3   # fewer false alarms

# Record a clip from the board, then run the model on it
~/yolo26/bin/python -m demo.raspberry.audio_wake.stm32_mic --seconds 15 --record stm32_drone.wav
./demo/scripts/run_audio_wake.sh --wav stm32_drone.wav --verbose
```

```bash
systemctl --user start edth-rbp      # back to the normal setup afterwards
```

STM32 sound **into the web app** without a camera is simply section 2: the
service runs, and the video just stays off.

---

## 4. On the Mac

```bash
cd ~/Documents/Development/Hardware/EDTH/EDTH-TargeTeam

# Web app (needs the Pi running, section 2)
./demo/scripts/run_pc.sh             # http://localhost:8501
```

Test the audio model locally, without the Pi. One-time setup (onnxruntime has no
Python 3.15 build yet, so it gets its own Python 3.13 environment):

```bash
/opt/homebrew/bin/python3.13 -m venv models/audio/.venv
models/audio/.venv/bin/pip install numpy onnxruntime sounddevice pillow
```

```bash
# Model self-test
(cd models/audio/raspberry_pi && ../.venv/bin/python inference_pi.py)

# The reference drone sample: expect drone=0.97 and "WAKE sent"
models/audio/.venv/bin/python -m demo.raspberry.audio_wake.main --wav models/audio/drone_reference.wav --verbose

# Any audio file (wav, flac, mp3, m4a, video...; short clips are looped)
models/audio/.venv/bin/python -m demo.raspberry.audio_wake.main --wav ~/Downloads/sound.mp3 --verbose

# The Mac's own microphone, live (macOS asks for microphone permission once)
models/audio/.venv/bin/python -m demo.raspberry.audio_wake.main --verbose
```

---

## 5. The Pi service (`edth-rbp`)

```bash
systemctl --user start edth-rbp
systemctl --user stop edth-rbp
systemctl --user restart edth-rbp     # after changing demo/.env or the code
systemctl --user status edth-rbp
systemctl --user enable edth-rbp      # start at boot (already on)
systemctl --user disable edth-rbp     # don't start at boot
journalctl --user -u edth-rbp -f      # live log
journalctl --user -u edth-rbp -n 100  # last 100 lines
```

No `sudo`: it is a user service. Run the same program by hand instead:

```bash
systemctl --user stop edth-rbp
cd ~/EDTH-TargeTeam && ./demo/scripts/run_rbp.sh      # Ctrl+C to stop
```

---

## 6. Settings (`demo/.env` on the Pi)

Edit with `nano ~/EDTH-TargeTeam/demo/.env`, then `systemctl --user restart edth-rbp`.

| Setting | Value | Meaning |
|---|---|---|
| `AUDIO_INPUT_FORMAT` | `stm32` | Mic = the STM32 board. `pulse` = the camera's / system mic (`AUDIO_DEVICE`) |
| `PC_HOST` | Mac IP | Where video, sound and results are sent |
| `AUDIO_RMS_WAKE_THRESHOLD` | `0.03` | Loudness (RMS, 0.03 = −30 dBFS) below which the model doesn't run |
| `AUDIO_VEHICLE_PROB_THRESHOLD` | `0.5` | Model confidence that switches the camera on |
| `VIDEO_WAKE_HOLD_SECONDS` | `10.0` | How long the camera stays on after the last detection |
| `STM32_MIC_PORT` / `STM32_MIC_BAUD` | empty / `1000000` | Board serial port (empty = auto-detect) and speed |

All settings: [demo/.env.example](../demo/.env.example).

---

## 7. STM32 firmware (on the Pi)

```bash
cd ~/EDTH-TargeTeam/stm32/mic_stream
make                 # build
make flash           # program the board over its USB cable (edth-rbp can keep running; it reconnects)
make restore         # put back the STM32 team's original firmware (~/stm32_backup/)
```

Details, stream format and checks: [stm32/mic_stream/README.md](../stm32/mic_stream/README.md).

---

## 8. When something doesn't work

| Symptom | Fix |
|---|---|
| `Device or resource busy` on the STM32 port | Another program reads it: `systemctl --user stop edth-rbp`, or stop the other run |
| `STM32 mic: ... retrying every 2 s` in the log | Board unplugged or not found: `lsusb` must show `STMicroelectronics ST-LINK/V2.1`; replug it |
| `Video could not start` in the log | Camera unplugged or wrong `VIDEO_DEVICE` (`v4l2-ctl --list-devices`). Audio keeps working |
| Web app shows nothing | Wrong `PC_HOST` on the Pi, or the Mac firewall blocks Python: System Settings → Network → Firewall → allow it |
| Audio panel always `background` | Sound too quiet at the board: get closer / louder, or lower `AUDIO_RMS_WAKE_THRESHOLD` |
| Always `footsteps` | Known weakness of the current model: most loud room noise looks like footsteps to it |
| `ssh: connect ... timed out` | Pi on another network or new address: `./mac/find-pi.sh` |
