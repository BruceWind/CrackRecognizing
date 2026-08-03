#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/best.pt}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export MODEL_PATH

if [[ -f "$MODEL_PATH" ]]; then
  printf 'Starting with model: %s\n' "$MODEL_PATH"
else
  printf 'No model found. Upload best.pt from the web page.\n'
fi

exec python -m uvicorn recognize_web.app.main:app --host "$HOST" --port "$PORT"
