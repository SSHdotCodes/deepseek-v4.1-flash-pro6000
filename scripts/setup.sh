#!/usr/bin/env bash
# Python environments, both CPython 3.12:
#   .venv         the API server, scripts/download.py and the benchmarks (requirements.txt)
#   .venv-assets  PyTorch 2.14 (CUDA 13.0 wheels) for scripts/prepare_assets.py (requirements-assets.txt, ~6 GB)
# Uses uv to fetch Python 3.12 when python3.12 is not installed.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
make_venv() {
  local dir="$1"
  [[ -x "$dir/bin/python" ]] && return
  if command -v uv > /dev/null; then
    uv venv --seed --python 3.12 "$dir"
  elif command -v python3.12 > /dev/null; then
    python3.12 -m venv "$dir"
  else
    echo "setup.sh: install uv (https://docs.astral.sh/uv/) or Python 3.12" >&2; exit 2
  fi
}
make_venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --quiet -r "$ROOT/requirements.txt"
make_venv "$ROOT/.venv-assets"
# pip, not uv pip: torch comes from the PyTorch CUDA 13.0 index, everything else from PyPI.
"$ROOT/.venv-assets/bin/python" -m pip install --quiet -r "$ROOT/requirements-assets.txt"
"$ROOT/.venv/bin/python" -c 'import fastapi, tokenizers, huggingface_hub; print("API environment OK")'
"$ROOT/.venv-assets/bin/python" -c 'import torch; print("asset environment OK: torch", torch.__version__)'
