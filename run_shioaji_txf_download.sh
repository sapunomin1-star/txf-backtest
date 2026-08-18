#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/guichenxiang/txf_backtest"
VENV="$ROOT/.venv-shioaji"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing venv: $VENV"
  echo "Run first: $ROOT/setup_shioaji_env.sh"
  exit 1
fi

"$VENV/bin/python" "$ROOT/shioaji_txf_kbars_download.py" "$@"
