#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"
UI_DIR="$ROOT_DIR/ui"
API_DIR="$ROOT_DIR/api"
mkdir -p "$RUN_DIR"

resolve_python() {
  local candidate
  for candidate in \
    "${VIRTUAL_ENV:-}/bin/python" \
    "$ROOT_DIR/.venv/bin/python" \
    "$(cd "$ROOT_DIR/.." && pwd)/.venv/bin/python"
  do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  echo "No Python interpreter found for aria_designer API startup." >&2
  return 1
}

ensure_ui_deps() {
  if [[ -d "$UI_DIR/node_modules" ]]; then
    return 0
  fi

  echo "Installing UI dependencies in $UI_DIR..."
  (
    cd "$UI_DIR"
    npm ci
  )
}

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
  PYTHON_BIN="$(resolve_python)"
  echo "Starting API (port 8091)..."
  (
    cd "$API_DIR"
    "$PYTHON_BIN" -m uvicorn app.main:app --reload --port 8091
  ) &
  API_PID=$!
  echo "$API_PID" > "$RUN_DIR/api.pid"
fi

if ! $ui_up; then
  rm -f "$RUN_DIR/ui.pid"
  ensure_ui_deps
  echo "Starting UI (port 5174)..."
  (
    cd "$UI_DIR"
    npm exec vite -- --port 5174 --strictPort
  ) &
  UI_PID=$!
  echo "$UI_PID" > "$RUN_DIR/ui.pid"
fi

echo "API: http://127.0.0.1:8091  UI: http://localhost:5174"
echo "Run 'make dev-stop' to stop both servers."

# Wait for started processes (don't block if none were started)
wait 2>/dev/null || true
