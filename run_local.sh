#!/usr/bin/env bash
# Local dual-instance run without Docker: two uvicorn processes sharing one
# SQLite file. Requires a Python with the packages from requirements.txt.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
HOST="127.0.0.1"
PORT_A="${PORT_A:-8081}"
PORT_B="${PORT_B:-8082}"
DATA_DIR="${DATA_DIR:-$(pwd)/data}"
DB_PATH="$DATA_DIR/rainfall.db"

mkdir -p "$DATA_DIR"
export RAIN_DB_PATH="$DB_PATH"
export RAIN_BUSY_TIMEOUT_MS="${RAIN_BUSY_TIMEOUT_MS:-10000}"
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH:-}"

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill -INT "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

echo "Starting instance A on http://$HOST:$PORT_A (db=$DB_PATH)"
"$PYTHON" -m uvicorn app.main:app --host "$HOST" --port "$PORT_A" &
pids+=("$!")
echo "Starting instance B on http://$HOST:$PORT_B (same db)"
"$PYTHON" -m uvicorn app.main:app --host "$HOST" --port "$PORT_B" &
pids+=("$!")

wait
