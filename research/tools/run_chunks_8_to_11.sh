#!/bin/bash
set -u
cd /home/tim/Projects/LLM || exit

WAIT_PID=723840
LOG=research/reports/chunk_runner.log

echo "[$(date '+%F %T')] waiting on chunk07 pid=${WAIT_PID}" >> "$LOG"
tail --pid=${WAIT_PID} -f /dev/null
echo "[$(date '+%F %T')] chunk07 pid=${WAIT_PID} exited" >> "$LOG"

for n in 08 09 10 11; do
    OUT="research/reports/controlled_lang_s10_validation_pending_chunk${n}.jsonl"
    CLOG="research/reports/chunk${n}.log"
    echo "[$(date '+%F %T')] starting chunk${n} -> ${OUT}" >> "$LOG"
    python -m research.tools.controlled_lang_backfill \
        --top-n 200 \
        --tiers s10 \
        --target-cohorts validation_pending \
        --missing-before-limit \
        --out "${OUT}" \
        > "${CLOG}" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] finished chunk${n} rc=${rc}" >> "$LOG"
    if [ $rc -ne 0 ]; then
        echo "[$(date '+%F %T')] chunk${n} failed (rc=${rc}); aborting remaining chunks" >> "$LOG"
        exit $rc
    fi
done

echo "[$(date '+%F %T')] all chunks 08-11 complete" >> "$LOG"
