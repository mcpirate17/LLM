#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"
mkdir -p "$RUN_DIR"

# ── Check if services are already running ──

api_up=false
ui_up=false

if curl -sf http://127.0.0.1:8091/health >/dev/null 2>&1; then
  api_up=true
  echo "API already running on port 8091"
fi

if curl -sf http://127.0.0.1:5174/ >/dev/null 2>&1; then
  ui_up=true
  echo "UI already running on port 5174"
fi

if $api_up && $ui_up; then
  echo "Both services already running — nothing to do."
  exit 0
fi

# ── Start services that aren't running ──

if ! $api_up; then
  # Clean stale pid file
  rm -f "$RUN_DIR/api.pid"
  echo "Starting API (port 8091)..."
  (
    cd "$ROOT_DIR/api"
    .venv/bin/python -m uvicorn app.main:app --reload --port 8091
  ) &
  API_PID=$!
  echo "$API_PID" > "$RUN_DIR/api.pid"
fi

if ! $ui_up; then
  rm -f "$RUN_DIR/ui.pid"
  echo "Starting UI (port 5174)..."
  (
    cd "$ROOT_DIR/ui"
    npx vite --port 5174 --strictPort
  ) &
  UI_PID=$!
  echo "$UI_PID" > "$RUN_DIR/ui.pid"
fi

echo "API: http://127.0.0.1:8091  UI: http://localhost:5174"
echo "Run 'make dev-stop' to stop both servers."

# Wait for started processes (don't block if none were started)
wait 2>/dev/null || true
