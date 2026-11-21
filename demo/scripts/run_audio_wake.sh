#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/demo/.env" ]]; then
    set -a
    source "$ROOT/demo/.env"
    set +a
fi

# Best Pi model by default; set AUDIO_MODEL_PATH (e.g. in demo/.env) to override.
export AUDIO_MODEL_PATH="${AUDIO_MODEL_PATH:-models/audio/native_knn_mn30}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" -m demo.raspberry.audio_wake.main "$@"
