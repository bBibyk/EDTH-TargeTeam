#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/demo/.env" ]]; then
    set -a
    source "$ROOT/demo/.env"
    set +a
fi

# Detection on the PC, on the received video: YOLO26n NCNN export @ 320, people only.
# Start the Raspberry side with --no-detection, or boxes are drawn twice.
export PC_DETECTION=1
export YOLO_MODEL_PATH=models/vision/yolo26n_ncnn_model
export YOLO_IMGSZ=320
export DETECTION_CLASSES="${DETECTION_CLASSES:-person}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" -m streamlit run demo/pc/main.py "$@"
