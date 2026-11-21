# mic_stream: STM32 microphone → Raspberry Pi

Firmware for the **B-L475E-IOT01A** (STM32L475VG, "Discovery kit IoT node").
The board's MEMS microphone becomes a 32 kHz audio stream on the Pi's
`/dev/ttyACM0`, over the same USB cable that programs the board (the ST-LINK port).
The board only digitises sound. The Pi computes the mel spectrogram and runs the
model (`demo/raspberry/audio_wake`) with the same code the model was trained with.

```
MP34DT01 mic ──PDM 1.6 MHz──▶ DFSDM sinc5 ÷25 → 64 kHz ──DMA──▶ FIR + ÷2 → 32 kHz int16
             ──USART1 1 Mbaud──▶ ST-LINK ──USB──▶ Pi /dev/ttyACM0 ──▶ stm32_mic.py ──▶ model
```

## Stream format

| | |
|---|---|
| Audio | 32 000 Hz, mono, signed 16-bit, linear PCM. No automatic gain, no compression |
| Serial | 1 000 000 baud, 8N1, on the ST-LINK virtual COM port |
| Packet (one every 8 ms) | `A5 5A` · `seq` u16 · `n` u16 (=256) · `n` × int16 · `checksum` u16. All little-endian |
| Checksum | `seq + n + every sample` (each as u16), mod 65536 |
| `seq` | +1 per packet. A gap means lost packets |
| At boot | One text line: `$MIC,32000,1,S16LE,mic_stream-2` |
| Data rate | 65 000 B/s, about 65 % of the link |

## Signal chain

- **Microphone:** MP34DT01 (M1). PE9 = `DFSDM1_CKOUT` and PE7 = `DFSDM1_DATIN2`, channel 2, falling edge. This is the same wiring and 1.6 MHz clock the original firmware used.
- **DFSDM filter 0:** sinc5, FOSR 25, fast continuous mode, giving 64 kHz. DMA1 channel 4 writes it into RAM.
- **CPU (Cortex-M4F, about 10 % load):**
  - A 64-tap FIR ([src/fir_coeffs.h](src/fir_coeffs.h), from [tools/design_fir.py](tools/design_fir.py)) followed by keeping every 2nd sample:
    - flat within ±0.07 dB from 20 Hz to 14 kHz, including the sinc5 droop correction;
    - −2.4 dB at 15 kHz, −10.5 dB at 16 kHz;
    - at least −70 dB above 18 kHz, and −59 dB first-stage aliasing.
  - A DC-blocking high-pass at about 10 Hz, then a fixed **+12 dB** gain, so int16 clips at about 108 dB SPL.
- **Clock:** 80 MHz from the MSI oscillator plus the PLL, locked to the board's 32.768 kHz crystal, so the sample rate is accurate. Measured: 125.0 packets/s.

## Build, flash, check (on the Pi)

```bash
sudo apt install gcc-arm-none-eabi libnewlib-arm-none-eabi openocd   # once (about 2.5 GB with recommends)
cd ~/EDTH-TargeTeam/stm32/mic_stream
make && make flash                                                   # flash through the ST-LINK USB

cd ~/EDTH-TargeTeam
~/yolo26/bin/python -m demo.raspberry.audio_wake.stm32_mic                        # rate, level, lost packets per second
~/yolo26/bin/python -m demo.raspberry.audio_wake.stm32_mic --seconds 30 --record clip.wav
./demo/scripts/run_audio_wake.sh --device stm32 --verbose                         # live into the model
```

A healthy board shows about 32 000 samples/s and `lost 0 bad 0`. The first second reads a little low, because the reader throws away the stale data in the ST-LINK buffer.

`/etc/udev/rules.d/99-stlink-no-modemmanager.rules` on the Pi stops ModemManager
from probing the port as if it were a modem.

## The original firmware

Before flashing, the board ran the STM32 team's "log-mel streamer → raspberry_pi_audio_v1". A full 1 MB copy is in
`../backup/original_firmware_bank0.bin`, also on the Pi in `~/stm32_backup/`
(SHA-256 `378692335fbf32ea02c50a1e073f972e99614b0f223578f292ce1f63244474cc`). To put it back:

```bash
make restore        # = openocd ... -c "program ~/stm32_backup/original_firmware_bank0.bin 0x08000000 verify reset exit"
```

What it did, read from its output and its registers on 21 Nov 2026:

- **Output:**
  - `$IDLE,<level>,<threshold>,<ms>` every 2 s, 115200 baud;
  - one `$SHIP,<tag>,<ms>,102400` after boot, then `AA 55 AA 55` and a 128 × 200 float32 mel tensor.
- **Why it couldn't feed the model:**
  - The tensor was 99.6 % at the "silence" value. The audio looks scaled about 55 dB too low; the filter's full scale is about 15 600, not 2²³.
  - `level` was always 0 against a threshold of 1000, so live sound was never sent.
  - One tensor takes about 9 s to send at 115200 baud, but the model needs a new window every 0.5 s.
  - The DFSDM filter was not in fast mode. By the reference manual, that gives about 10.4 kHz instead of 32 kHz.
