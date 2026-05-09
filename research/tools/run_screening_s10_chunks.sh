#!/bin/bash
# Backfill language_control s10 (sa + nano-bind) over the 'screening' tier,
# restricted to rows that already passed s05 nano-bind. Chunked to keep each
# run resumable and to release the GPU between chunks.
#
# Usage:
#   research/tools/run_screening_s10_chunks.sh                # 10 chunks, default
#   START_CHUNK=03 END_CHUNK=05 research/tools/run_screening_s10_chunks.sh
#   WAIT_PID=12345 research/tools/run_screening_s10_chunks.sh # block until PID exits

set -u
cd /home/tim/Projects/LLM || exit

START_CHUNK="${START_CHUNK:-01}"
END_CHUNK="${END_CHUNK:-10}"
TOP_N="${TOP_N:-200}"
PREFIX="${PREFIX:-language_control_s10_screening_chunk}"
WAIT_PID="${WAIT_PID:-}"

LOG=research/reports/chunk_runner_screening_s10.log

if [[ -n "${WAIT_PID}" ]]; then
    if kill -0 "${WAIT_PID}" 2>/dev/null; then
        echo "[$(date '+%F %T')] waiting on pid=${WAIT_PID}" >> "$LOG"
        tail --pid="${WAIT_PID}" -f /dev/null
        echo "[$(date '+%F %T')] pid=${WAIT_PID} exited" >> "$LOG"
    else
        echo "[$(date '+%F %T')] pid=${WAIT_PID} not running; proceeding" >> "$LOG"
    fi
fi

for n in $(seq -w "${START_CHUNK}" "${END_CHUNK}"); do
    OUT="research/reports/${PREFIX}${n}.jsonl"
    CLOG="research/reports/${PREFIX}${n}.log"
    echo "[$(date '+%F %T')] starting chunk${n} -> ${OUT}" >> "$LOG"
    python -m research.tools.language_control_backfill \
        --top-n "${TOP_N}" \
        --tiers s10 \
        --target-cohorts screening \
        --missing-before-limit \
        --require-s05-nb-pass \
        --out "${OUT}" \
        > "${CLOG}" 2>&1
    rc=$?
    written=$(wc -l < "${OUT}" 2>/dev/null || echo 0)
    echo "[$(date '+%F %T')] finished chunk${n} rc=${rc} rows=${written}" >> "$LOG"
    if [ $rc -ne 0 ]; then
        echo "[$(date '+%F %T')] chunk${n} failed (rc=${rc}); aborting" >> "$LOG"
        exit $rc
    fi
    if [ "${written}" = "0" ]; then
        echo "[$(date '+%F %T')] chunk${n} wrote 0 rows; cohort drained, stopping early" >> "$LOG"
        break
    fi
done

echo "[$(date '+%F %T')] screening s10 backfill done" >> "$LOG"
