#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/guichenxiang/txf_backtest"
VENV="$ROOT/.venv-shioaji"

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$ROOT/requirements_shioaji.txt"

echo "Shioaji environment ready: $VENV"
