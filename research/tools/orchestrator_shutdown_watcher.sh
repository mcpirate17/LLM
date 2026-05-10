#!/usr/bin/env bash
# Polls the orchestrator's status file and runs happy_times.py when the
# orchestrator reaches a terminal phase (done / target failed) or a
# safety timeout elapses.

set -u

STATUS_PATH="/home/tim/Projects/LLM/research/reports/orchestrator/orchestrator.status.json"
HAPPY_PATH="/home/tim/Projects/LLM/happy_times.py"
WATCHER_LOG="/home/tim/Projects/LLM/research/reports/orchestrator/shutdown_watcher.log"
ORCH_PROCESS_PATTERN="research.tools.tier_orchestrator"

# Walltime safety cap: shut down even if status never reaches a terminal
# phase, so the box does not stay on indefinitely if a run hangs.
MAX_WAIT_SECONDS=$((20 * 3600))   # 20 hours

# After we observe a terminal phase, give the orchestrator this long to
# finish flushing logs and writing final state before shutting down.
SETTLE_SECONDS=120

# How often to poll.
POLL_SECONDS=60

START=$(date +%s)
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$WATCHER_LOG"
}

log "shutdown watcher start (status=$STATUS_PATH max_wait=${MAX_WAIT_SECONDS}s)"

orch_alive() {
    pgrep -fa "$ORCH_PROCESS_PATTERN" >/dev/null 2>&1
}

read_phase() {
    if [[ ! -f "$STATUS_PATH" ]]; then
        echo ""
        return
    fi
    python3 -c "
import json, sys
try:
    print(json.load(open('$STATUS_PATH')).get('phase', ''))
except Exception:
    print('')
" 2>/dev/null
}

while true; do
    PHASE=$(read_phase)
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))

    if [[ "$PHASE" == "done" || "$PHASE" == "stopped_target_failed" ]]; then
        if orch_alive; then
            log "phase=$PHASE but orchestrator process still alive — waiting"
        else
            log "phase=$PHASE and orchestrator exited; settling ${SETTLE_SECONDS}s before shutdown"
            sleep "$SETTLE_SECONDS"
            log "running happy_times.py"
            python3 "$HAPPY_PATH"
            log "happy_times.py returned $?"
            exit 0
        fi
    fi

    if (( ELAPSED >= MAX_WAIT_SECONDS )); then
        log "MAX_WAIT_SECONDS reached (phase=$PHASE) — shutting down anyway"
        python3 "$HAPPY_PATH"
        exit 1
    fi

    # Heartbeat every ~10 minutes
    if (( ELAPSED % 600 < POLL_SECONDS )); then
        log "heartbeat: phase=${PHASE:-<none>} elapsed=${ELAPSED}s orch_alive=$(orch_alive && echo yes || echo no)"
    fi

    sleep "$POLL_SECONDS"
done
