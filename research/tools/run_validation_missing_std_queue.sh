#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/tim/Projects/LLM"
LOG="$ROOT/logs/validation_missing_std_backfill.out"
PILOT_PATTERN='python -m research.tools.rerun_validation_missing_std --db research/lab_notebook.db --batch-size 3 --limit 3 --poll-seconds 5'
REST_CMD='python -m research.tools.rerun_validation_missing_std --db research/lab_notebook.db --batch-size 5 --poll-seconds 10'

mkdir -p "$ROOT/logs"

{
  echo "[$(date '+%F %T')] queue-runner started"
  while pgrep -f "$PILOT_PATTERN" >/dev/null 2>&1; do
    echo "[$(date '+%F %T')] waiting for pilot batch to finish"
    sleep 15
  done

  echo "[$(date '+%F %T')] pilot batch finished; starting remaining validation reruns"
  cd "$ROOT"
  exec $REST_CMD
} >> "$LOG" 2>&1
