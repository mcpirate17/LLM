#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/tim/Projects/LLM/aria_designer"
API_DIR="$ROOT/api"
UI_DIR="$ROOT/ui"

# API
if [ ! -d "$API_DIR/.venv" ]; then
  python -m venv "$API_DIR/.venv"
fi
source "$API_DIR/.venv/bin/activate"
python -m pip install -r "$API_DIR/requirements.txt"

# Start API in background
uvicorn app.main:app --reload --port 8091 --app-dir "$API_DIR" &
API_PID=$!

# UI
cd "$UI_DIR"
if [ ! -d node_modules ]; then
  npm install
fi
npm run dev -- --port 5174 &
UI_PID=$!

# Keep running until killed
wait $API_PID $UI_PID
