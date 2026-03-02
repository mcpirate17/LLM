#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"
mkdir -p "$RUN_DIR"

# Clean stale pid files.
rm -f "$RUN_DIR/api.pid" "$RUN_DIR/ui.pid"

echo "Starting API (port 8091) and UI (port 5174, strict)..."
(
  cd "$ROOT_DIR/api"
  .venv/bin/python -m uvicorn app.main:app --reload --port 8091
) &
API_PID=$!
echo "$API_PID" > "$RUN_DIR/api.pid"

(
  cd "$ROOT_DIR/ui"
  npx vite --port 5174 --strictPort
) &
UI_PID=$!
echo "$UI_PID" > "$RUN_DIR/ui.pid"

echo "API: http://127.0.0.1:8091  UI: http://localhost:5174"
echo "Run 'make dev-stop' to stop both servers."

wait "$API_PID" "$UI_PID"
