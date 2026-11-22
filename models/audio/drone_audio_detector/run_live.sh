#!/usr/bin/env bash
# Live microphone monitor. Extra args go to drone_audio/live.py (--device N, --plain, --csv out.csv, --threshold 0.4 ...)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=python3; [[ -x .venv/bin/python ]] && PY=.venv/bin/python
exec "$PY" -m drone_audio.live --threads "${THREADS:-2}" "$@"
