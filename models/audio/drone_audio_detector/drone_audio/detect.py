"""Score an audio file offline: one line per 5 s window + a summary.

    python -m drone_audio.detect recording.wav                 # 16-bit PCM wav, any rate / channels
    python -m drone_audio.detect recording.wav --hop 1 --csv out.csv

Other formats: ffmpeg -i in.mp3 -ac 1 -ar 16000 -sample_fmt s16 out.wav
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drone_audio.live import SR, read_wav  # noqa: E402
from drone_audio.native import CLASSES, load_native_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("wav", type=Path)
    ap.add_argument("--model", type=Path, default=ROOT / "model")
    ap.add_argument("--hop", type=float, default=1.0, help="seconds between windows")
    ap.add_argument("--threshold", type=float, help="override the model's drone threshold")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    model = load_native_model(args.model, threads=args.threads)
    thr = model.thr if args.threshold is None else args.threshold
    x = read_wav(args.wav)
    win, hop = 5 * SR, max(1, int(args.hop * SR))
    if len(x) < win:  # pad short clips with silence to one window
        x = np.pad(x, (0, win - len(x)))
    rows = []
    for start in range(0, len(x) - win + 1, hop):
        p_drone, p3 = model(x[start:start + win])
        rows.append((start / SR, p_drone, *p3))
        top = CLASSES[int(np.argmax(p3))]
        flag = "DRONE" if p_drone >= thr else "-----"
        print(f"{start / SR:8.1f}s {flag} " + " ".join(f"{c}={p:.2f}" for c, p in zip(CLASSES, p3)) + f"  top={top}")
    P = np.array([r[1] for r in rows])
    print(f"\n{len(rows)} windows, threshold {thr:.3f}: drone in {(P >= thr).sum()} "
          f"({(P >= thr).mean() * 100:.0f} %), max P(drone) {P.max():.3f}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["start_s", "p_drone", *CLASSES])
            w.writerows([[f"{v:.4f}" for v in r] for r in rows])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
