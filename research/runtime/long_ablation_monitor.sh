#!/usr/bin/env bash
set -u

RUN_ID="${1:?run id required}"
HAPPY_TIMES="${2:?happy_times.py path required}"
DEADLINE_SECONDS="${3:-18000}"
API_BASE="${4:-http://127.0.0.1:5000}"
PROJECT_ROOT="${5:-/home/tim/Projects/LLM}"
LOG_FILE="${PROJECT_ROOT}/research/runtime/long_ablation_monitor_${RUN_ID}.log"
START_EPOCH="$(date +%s)"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" >> "$LOG_FILE"
}

api_running() {
  local status_json
  status_json="$(curl -fsS "${API_BASE}/api/aria/cycle-status" 2>/dev/null || true)"
  if [[ -z "$status_json" ]]; then
    printf '1'
    return
  fi
  python -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    print(1)
    raise SystemExit
running = bool(data.get("continuous_active") or data.get("is_running") or data.get("phase") not in ("idle", None))
print(1 if running else 0)
' <<< "$status_json" 2>/dev/null || printf '1'
}

db_running_count() {
  sqlite3 "${PROJECT_ROOT}/research/lab_notebook.db" \
    "SELECT COUNT(*) FROM experiments WHERE status = 'running';" 2>/dev/null || printf '1'
}

log "monitor started run_id=${RUN_ID} deadline_seconds=${DEADLINE_SECONDS}"

while true; do
  now="$(date +%s)"
  elapsed=$((now - START_EPOCH))
  api_state="$(api_running)"
  db_state="$(db_running_count)"
  evidence_count="$(sqlite3 "${PROJECT_ROOT}/research/lab_notebook.db" "SELECT COUNT(*) FROM causal_rule_evidence;" 2>/dev/null || printf '?')"
  observation_count="$(sqlite3 "${PROJECT_ROOT}/research/lab_notebook.db" "SELECT COUNT(*) FROM causal_ablation_child_observations;" 2>/dev/null || printf '?')"
  log "elapsed=${elapsed}s api_running=${api_state} db_running=${db_state} evidence=${evidence_count} observations=${observation_count}"

  if [[ "$api_state" != "1" && "$db_state" == "0" ]]; then
    if (( elapsed < DEADLINE_SECONDS )); then
      log "run completed before deadline; invoking ${HAPPY_TIMES}"
      python "$HAPPY_TIMES" >> "$LOG_FILE" 2>&1
    else
      log "run completed after deadline; leaving host on"
    fi
    exit 0
  fi

  if (( elapsed >= DEADLINE_SECONDS )); then
    log "deadline reached while run still active; leaving host on"
    exit 0
  fi

  sleep 60
done
