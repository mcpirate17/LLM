#!/usr/bin/env bash
# Background worker for the once-a-day full-repo bloat audit. Launched detached
# by session-start.sh (which has a 10s timeout the ~1min audit would blow). Writes
# a timestamped report into research/reports/; the NEXT session surfaces it.
# Never blocks the session and never edits code — report generation only.
set -euo pipefail

REPO_ROOT="$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")"
VENV_BIN="/home/tim/venvs/llm/bin"

# Put the project venv first so ruff/vulture/radon resolve; keep node/npx from the
# inherited PATH (jscpd runs via npx).
export PATH="$VENV_BIN:$PATH"
PY="$VENV_BIN/python"
[[ -x "$PY" ]] || PY="python3"

exec "$PY" "$REPO_ROOT/research/tools/full_repo_audit.py" >/dev/null 2>&1
