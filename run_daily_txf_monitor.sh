#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/guichenxiang/txf_backtest"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/txf_monitor_${STAMP}.log"

# 日常監控只讀 Shioaji 近月檔；程式會拒絕過期資料。完整歷史合併檔僅供研究，
# 不可再拿固定 2026-06-26 檔冒充今日訊號。
MINUTE_CSV="${MINUTE_CSV:-$ROOT/shioaji_data/TXFR1_1min.csv}"

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] start"
  "$ROOT/.venv-shioaji/bin/python" "$ROOT/txf_strategy_monitor.py" --minute-csv "$MINUTE_CSV" "$@"
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] done"
} 2>&1 | tee "$LOG_FILE"

echo "Log: $LOG_FILE"
