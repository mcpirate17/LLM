#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"

echo "Stopping runtime API..."

kill_pid_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    local pid
    pid="$(cat "$file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$file"
  fi
}

kill_pid_file "$RUN_DIR/runtime_api.pid"

runtime_pids="$(lsof -ti tcp:8091 2>/dev/null || true)"
if [[ -n "$runtime_pids" ]]; then
  kill $runtime_pids 2>/dev/null || true
fi

pkill -f "uvicorn app.main:app --host 127.0.0.1 --port 8091" 2>/dev/null || true

echo "Stopped."
