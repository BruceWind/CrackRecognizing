#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYPI_MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

cd "$PROJECT_ROOT"

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip --index-url "$PYPI_MIRROR"

# WSL on this AMD laptop uses CPU PyTorch rather than CUDA or ROCm.
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  -r requirements/cpu.txt
python -m pip install \
  --index-url "$PYPI_MIRROR" \
  -r requirements/base.txt \
  -r requirements/web.txt

mkdir -p models web_data/uploads web_data/results
printf 'Environment ready. Put best.pt at %s/models/best.pt\n' "$PROJECT_ROOT"
