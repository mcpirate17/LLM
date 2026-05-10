#!/usr/bin/env bash
# Chains two orchestrator batches and shuts down at the end.
#
# Batch 1: already-running orchestrator (PID below) — target 54b0557c-472 + 10 backfills.
# Batch 2: launched here once batch 1 exits — top 20 promising fps with NULL
#          intermediate AUCs (composite >100), to find the real architecture leader
#          beyond loss-biased ranking.
# Final:   happy_times.py to power down.

set -u

CHAIN_LOG="/home/tim/Projects/LLM/research/reports/orchestrator/chain.log"
STATUS_PATH="/home/tim/Projects/LLM/research/reports/orchestrator/orchestrator.status.json"
HAPPY_PATH="/home/tim/Projects/LLM/happy_times.py"
ORCH_PROCESS_PATTERN="research.tools.tier_orchestrator"

# Batch 2 fps (top 20 by composite, missing capability rankers, distinct from batch 1)
BATCH2_TARGET="d18571d1-eff"
BATCH2_BACKFILL=(
    "3504a263-413" "c17fbe7c-992" "3272cd8f-611" "d4a29bc8-037" "7bad912e-674"
    "1a711bd5-215" "d95d84d8-c8b" "4c62426b-010" "8b9b42ed-c58" "01a2b221-7fd"
    "eebf21e5-7ff" "67510f91-245" "3e971c06-1b5" "ef547730-c02" "dcc3859a-af4"
    "e939e46e-b22" "4e546f0c-37f" "80b103c7-de7" "d95022f6-2fc"
)

POLL_SECONDS=60
SETTLE_SECONDS=120
MAX_TOTAL_HOURS=24

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$CHAIN_LOG"
}

orch_alive() {
    pgrep -fa "$ORCH_PROCESS_PATTERN" >/dev/null 2>&1
}

read_phase() {
    [[ -f "$STATUS_PATH" ]] || { echo ""; return; }
    python3 -c "
import json
try:
    print(json.load(open('$STATUS_PATH')).get('phase',''))
except Exception:
    print('')
" 2>/dev/null
}

wait_for_orchestrator_exit() {
    local label="$1"
    local started
    started=$(date +%s)
    local last_heartbeat=0
    while orch_alive; do
        local now
        now=$(date +%s)
        local elapsed=$((now - started))
        if (( now - last_heartbeat >= 600 )); then
            log "  $label heartbeat: still running ${elapsed}s phase=$(read_phase)"
            last_heartbeat=$now
        fi
        if (( elapsed > MAX_TOTAL_HOURS * 3600 )); then
            log "  $label exceeded ${MAX_TOTAL_HOURS}h cap, abandoning wait"
            return 1
        fi
        sleep "$POLL_SECONDS"
    done
    log "  $label exited; final phase=$(read_phase)"
    return 0
}

log "chain start"
log "batch 1: waiting for currently-running orchestrator to finish"
wait_for_orchestrator_exit "batch1"
log "batch 1 settle ${SETTLE_SECONDS}s"
sleep "$SETTLE_SECONDS"

log "batch 2: launching orchestrator with 1 target + 19 backfill (20 fps total)"
cd /home/tim/Projects/LLM || exit
nohup bash -c "source /home/tim/venvs/llm/bin/activate && \
    python -m research.tools.tier_orchestrator \
    --target $BATCH2_TARGET \
    --backfill ${BATCH2_BACKFILL[*]}" \
    >> /home/tim/Projects/LLM/research/reports/orchestrator/run.log 2>&1 &
B2_PID=$!
disown
log "batch 2 launched, PID=$B2_PID"

# Give it a few seconds to start
sleep 5
if ! orch_alive; then
    log "batch 2 died immediately; aborting chain"
    log "running happy_times.py anyway"
    python3 "$HAPPY_PATH"
    exit 1
fi

log "batch 2: waiting to finish"
wait_for_orchestrator_exit "batch2"

log "all batches done; settle ${SETTLE_SECONDS}s before shutdown"
sleep "$SETTLE_SECONDS"
log "running happy_times.py"
python3 "$HAPPY_PATH"
log "happy_times.py returned $?"
