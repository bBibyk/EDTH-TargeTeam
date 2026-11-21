#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/demo/.env" ]]; then
    set -a
    source "$ROOT/demo/.env"
    set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" -m demo.raspberry.audio_wake.main "$@"
