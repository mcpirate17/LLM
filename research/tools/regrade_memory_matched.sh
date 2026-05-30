#!/usr/bin/env bash
# FEASIBLE matched-budget 100M re-grade. The naive O(T) surprise-memory scans are
# ~94x slower than softmax (eager ~1.8k tok/s @b8 vs 167k); 122k steps is
# infeasible (~78h/lane). So: all 4 lanes at a MATCHED small budget (same steps,
# batch, optimizer, eager) for a fair RELATIVE comparison. The well-trained 500M
# softmax result (ppl 34.35, run-label *_100m_ts) is kept separately as the
# absolute reference. New run-labels use the *_matched suffix.
#
# batch16 raises memory-lane throughput ~3x vs b8; expandable_segments avoids the
# 100k-vocab logits OOM that killed batch32. softmax ordered FIRST as a fast fit
# check (minutes) before committing the slow memory-lane hours.
#
# Usage: bash research/tools/regrade_memory_matched.sh <steps> <batch>
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/tim/venvs/llm/bin/python
STEPS="${1:-6000}"
BATCH="${2:-16}"
CKPTS="$((STEPS/2)),${STEPS}"
TRAIN=research/corpus/finefineweb_train.npy
VAL=research/corpus/finefineweb_val.npy
OUT=research/reports/mixer_fingerprint
mkdir -p "$OUT"
LANES="softmax_attention tropical_surprise_memory semiring_surprise_memory semiring_surprise_memory_rope"
rm -f research/reports/regrade_matched.DONE
for lane in $LANES; do
  echo "=== [$(date +%H:%M:%S)] $lane MATCHED (steps=$STEPS dim576 nb12 seq512 b$BATCH eager) ==="
  "$PY" -m research.tools.mixer_fingerprint \
    --mixer "$lane" \
    --corpus-tokens "$TRAIN" --val-corpus-tokens "$VAL" \
    --output "$OUT" --run-label "${lane}_matched" \
    --steps "$STEPS" --checkpoint-steps "$CKPTS" \
    --dim 576 --n-blocks 12 --seq-len 512 --batch-size "$BATCH" \
    --device cuda --seed 0 \
    --plateau-patience 99999 --plateau-min-steps 999999 \
    --amp --amp-dtype bf16 --no-compile \
    > "research/reports/regrade_${lane}_matched.log" 2>&1
  echo "    [$(date +%H:%M:%S)] $lane exit=$? -> ${OUT}/${lane}_matched.jsonl"
done
echo "ALL_DONE $(date +%H:%M:%S)" > research/reports/regrade_matched.DONE
