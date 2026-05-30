#!/usr/bin/env bash
# 100M matched-tok/param re-grade of the surprise-memory family with table-stakes
# ON (SwiGLU block FFN), vs a SOTA-grade softmax baseline.
# Lanes: softmax_attention | tropical_surprise_memory | semiring_surprise_memory |
#        semiring_surprise_memory_rope
# dim576 / n_blocks12 / seq512 / FineFineWeb / batch8. One CONTINUOUS cosine
# schedule per lane (NEVER staged-resume — re-warm corrupts matched-budget).
#
# Usage: bash research/tools/regrade_memory_100m.sh <steps> [batch]
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=/home/tim/venvs/llm/bin/python
STEPS="${1:?steps}"
BATCH="${2:-8}"
CKPTS="$((STEPS/2)),${STEPS}"
TRAIN=research/corpus/finefineweb_train.npy
VAL=research/corpus/finefineweb_val.npy
OUT=research/reports/mixer_fingerprint
mkdir -p "$OUT"
LANES="softmax_attention tropical_surprise_memory semiring_surprise_memory semiring_surprise_memory_rope"
rm -f research/reports/regrade_memory_100m.DONE
for lane in $LANES; do
  echo "=== [$(date +%H:%M:%S)] $lane (steps=$STEPS dim576 nb12 seq512 b$BATCH) ==="
  "$PY" -m research.tools.mixer_fingerprint \
    --mixer "$lane" \
    --corpus-tokens "$TRAIN" --val-corpus-tokens "$VAL" \
    --output "$OUT" --run-label "${lane}_100m_ts" \
    --steps "$STEPS" --checkpoint-steps "$CKPTS" \
    --dim 576 --n-blocks 12 --seq-len 512 --batch-size "$BATCH" \
    --device cuda --seed 0 \
    --plateau-patience 99999 --plateau-min-steps 999999 \
    --amp --amp-dtype bf16 --compile \
    > "research/reports/regrade_${lane}_100m.log" 2>&1
  echo "    [$(date +%H:%M:%S)] $lane exit=$? -> ${OUT}/${lane}_100m_ts.jsonl"
done
echo "ALL_DONE $(date +%H:%M:%S)" > research/reports/regrade_memory_100m.DONE
