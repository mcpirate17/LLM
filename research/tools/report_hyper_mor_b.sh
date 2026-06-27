#!/usr/bin/env bash
# Merge the hyper_mor_b trunk (hyper_mor_b_145m) + its chinchilla extension
# (hyper_mor_b_chin) training logs into ONE continuous data file, then render
# the training-dynamics + depth reports on the merged file. The two runs are a
# single trajectory (the extension resumes the trunk's step-35500 checkpoint),
# so the merged file gives one unbroken loss / val-ppl / depth curve.
#
# Idempotent and newline-safe: call it repeatedly (e.g. from a plot watcher) as
# the extension grows. The plotter keys metrics by step, so the merge is also
# order-robust, but we emit it chronologically (trunk then extension).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
source /home/tim/venvs/llm/bin/activate 2>/dev/null || true

REPORTS=research/reports
TRUNK=$REPORTS/hyper_mor_b_145m.jsonl
CHIN=$REPORTS/hyper_mor_b_chin.jsonl
FULL=$REPORTS/hyper_mor_b_full.jsonl

if [ ! -f "$TRUNK" ]; then
  echo "report_hyper_mor_b: missing trunk log $TRUNK" >&2
  exit 1
fi

# Chronological, newline-safe concatenation; grep -h . drops any blank lines so
# json.loads never sees an empty record.
{ cat "$TRUNK"; printf '\n'; [ -f "$CHIN" ] && cat "$CHIN"; } | grep -h . >"$FULL"

STEP=$(grep -o '"event": "step", "step": [0-9]*' "$FULL" | grep -o '[0-9]*$' | tail -1)
STEP=${STEP:-0}

python3 research/reports/mor_histo.py --mode curves --runs "$FULL" \
  --ppl --depth --log-loss \
  --title "hyper_mor_b FULL trunk+chinchilla (step ${STEP}/125000)" \
  --output "$REPORTS/hyper_mor_b_full_curves.png" >/dev/null 2>&1
python3 research/reports/mor_histo.py --mode mor --input "$FULL" \
  --output "$REPORTS/hyper_mor_b_full_depth_hist.png" >/dev/null 2>&1

echo "merged $FULL @ step ${STEP}; full-trajectory reports refreshed"
