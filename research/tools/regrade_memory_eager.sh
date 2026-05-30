#!/usr/bin/env bash
# Re-grade ONLY the 3 surprise-memory lanes in EAGER mode (--no-compile).
# softmax_attention already completed WITH compile (ppl 34.35), keep that jsonl.
# Why eager: the memory lanes are O(T) sequential Python scans over 512 timesteps;
# torch.compile tracing that loop at dim576/seq512 HANGS (54min, 0 steps, 100% CPU
# single-core, 58GB RSS, GPU idle). The scan gets no speedup from compile anyway.
#
# Usage: bash research/tools/regrade_memory_eager.sh <steps> [batch]
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
LANES="tropical_surprise_memory semiring_surprise_memory semiring_surprise_memory_rope"
rm -f research/reports/regrade_memory_eager.DONE
for lane in $LANES; do
  echo "=== [$(date +%H:%M:%S)] $lane EAGER (steps=$STEPS dim576 nb12 seq512 b$BATCH) ==="
  "$PY" -m research.tools.mixer_fingerprint \
    --mixer "$lane" \
    --corpus-tokens "$TRAIN" --val-corpus-tokens "$VAL" \
    --output "$OUT" --run-label "${lane}_100m_ts" \
    --steps "$STEPS" --checkpoint-steps "$CKPTS" \
    --dim 576 --n-blocks 12 --seq-len 512 --batch-size "$BATCH" \
    --device cuda --seed 0 \
    --plateau-patience 99999 --plateau-min-steps 999999 \
    --amp --amp-dtype bf16 --no-compile \
    > "research/reports/regrade_${lane}_100m.log" 2>&1
  echo "    [$(date +%H:%M:%S)] $lane exit=$? -> ${OUT}/${lane}_100m_ts.jsonl"
done
echo "ALL_DONE $(date +%H:%M:%S)" > research/reports/regrade_memory_eager.DONE
