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

start_detached() {
  local pid_file="$1"
  local log_file="$2"
  local command="$3"

  rm -f "$pid_file"
  : > "$log_file"

  nohup setsid bash -lc "$command" >>"$log_file" 2>&1 </dev/null &
  echo $! > "$pid_file"
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
  start_detached \
    "$RUN_DIR/api.pid" \
    "$RUN_DIR/api.log" \
    "cd '$API_DIR' && exec '$PYTHON_BIN' -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8091"
fi

if ! $ui_up; then
  rm -f "$RUN_DIR/ui.pid"
  ensure_ui_deps
  echo "Starting UI (port 5174)..."
  start_detached \
    "$RUN_DIR/ui.pid" \
    "$RUN_DIR/ui.log" \
    "cd '$UI_DIR' && exec ./node_modules/.bin/vite --host 127.0.0.1 --port 5174 --strictPort"
fi

echo "API: http://127.0.0.1:8091  UI: http://localhost:5174"
echo "Run 'make dev-stop' to stop both servers."
