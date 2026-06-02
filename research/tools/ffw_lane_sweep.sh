#!/usr/bin/env bash
# Sequential FineFineWeb mixer_fingerprint sweep over the campaign lanes.
# Each lane trains a TinyLM on FineFineWeb and evals BLiMP + nb05 + nb10 at the
# final checkpoint. Results land in research/reports/mixer_fingerprint/<lane>.jsonl.
#
# Usage: bash research/tools/ffw_lane_sweep.sh <steps> <n_blocks> <lane...>
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

STEPS="${1:?steps}"; NBLOCKS="${2:?n_blocks}"; shift 2
LANES=("$@")
CKPTS="$((STEPS/2)),${STEPS}"
TRAIN=research/corpus/finefineweb_train.npy
VAL=research/corpus/finefineweb_val.npy
OUT=research/reports/mixer_fingerprint
STAMP=$(date +%H%M%S)

for lane in "${LANES[@]}"; do
  echo "=== [$(date +%H:%M:%S)] FFW sweep: $lane (steps=$STEPS blocks=$NBLOCKS) ==="
  python -m research.tools.mixer_fingerprint \
    --mixer "$lane" \
    --corpus-tokens "$TRAIN" --val-corpus-tokens "$VAL" \
    --steps "$STEPS" --checkpoint-steps "$CKPTS" --n-blocks "$NBLOCKS" \
    --device cuda --output "$OUT" \
    > "research/reports/ffw_sweep_${lane}_${STAMP}.log" 2>&1
  echo "    exit=$? -> ${OUT}/${lane}.jsonl"
done
echo "=== FFW sweep complete ==="
