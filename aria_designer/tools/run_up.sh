#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"
API_DIR="$ROOT_DIR/api"
UI_DIR="$ROOT_DIR/ui"
mkdir -p "$RUN_DIR"

resolve_python() {
  local candidate
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    printf '%s\n' "${VIRTUAL_ENV}/bin/python"
    return 0
  fi

  for candidate in \
    "$API_DIR/.venv/bin/python" \
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

  echo "No Python interpreter found for aria_designer runtime startup." >&2
  return 1
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

if [[ ! -f "$UI_DIR/dist/index.html" ]]; then
  if [[ -f "$UI_DIR/package.json" ]] && command -v npm >/dev/null 2>&1; then
    echo "Building embedded UI..."
    (cd "$UI_DIR" && npm run build)
  else
    echo "Embedded UI build missing and npm/package.json are unavailable." >&2
  fi
fi

if curl -sf http://127.0.0.1:8091/health >/dev/null 2>&1; then
  echo "API already running on port 8091"
  exit 0
fi

PYTHON_BIN="$(resolve_python)"
echo "Starting runtime API (port 8091)..."
start_detached \
  "$RUN_DIR/runtime_api.pid" \
  "$RUN_DIR/runtime_api.log" \
  "cd '$API_DIR' && exec '$PYTHON_BIN' -m uvicorn app.main:app --host 127.0.0.1 --port 8091"

echo "API: http://127.0.0.1:8091"
echo "UI: served by dashboard /designer-proxy/"
