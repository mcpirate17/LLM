#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"

echo "Stopping dev servers..."

kill_pid_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    local pid
    pid="$(cat "$file" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$file"
  fi
}

kill_pid_file "$RUN_DIR/api.pid"
kill_pid_file "$RUN_DIR/ui.pid"

# Fallback cleanup for orphaned listeners from prior runs.
for p in 8091 $(seq 5174 5200); do
  pids="$(lsof -ti tcp:"$p" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    kill $pids 2>/dev/null || true
  fi
done

# Final fallback by command signature.
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "vite --port 5174" 2>/dev/null || true
pkill -f "npx vite --port 5174" 2>/dev/null || true

echo "Stopped."
