#!/usr/bin/env bash
set -euo pipefail

jobs="${JOBS:-5}"
device="${DEVICE:-cuda}"
chunk_dir="${CHUNK_DIR:-/home/tim/Projects/LLM/research/reports/binding_backfill/chunks_20}"
log_dir="${LOG_DIR:-/home/tim/Projects/LLM/research/reports/binding_backfill/logs}"
report_dir="${REPORT_DIR:-/home/tim/Projects/LLM/research/reports/binding_backfill/reports}"

mkdir -p "$log_dir" "$report_dir"

mapfile -t chunks < <(find "$chunk_dir" -maxdepth 1 -type f -name 'chunk_*.txt' | sort)
if [ "${#chunks[@]}" -eq 0 ]; then
  echo "No chunk files found in $chunk_dir" >&2
  exit 1
fi

running=0
for chunk in "${chunks[@]}"; do
  name="$(basename "$chunk" .txt)"
  (
    cd /home/tim/Projects/LLM
    python -m research.tools.backpopulate_screening_metrics \
      --from-report "$chunk" \
      --device "$device" \
      --fallback-device none \
      --batch-commit 1 \
      --post-train-target binding \
      --skip-rapid \
      --selection-slice backfill \
      --report "$report_dir/${name}.tsv"
  ) >"$log_dir/${name}.log" 2>&1 &
  running=$((running + 1))
  if [ "$running" -ge "$jobs" ]; then
    wait -n
    running=$((running - 1))
  fi
done

wait
