#!/usr/bin/env bash
# Autonomous post-training finish-handler for rdr_padic_46Mactive (the novel
# p-adic gated-mixer scale run). Blocks until the step-100000 final checkpoint
# appears (or the trainer PID exits), then runs the full capability battery,
# ingests everything into runs.db, builds the keeper comparison table, and
# syncs the write-up to Obsidian. Every heavy step is non-fatal: a single probe
# failure logs and the pipeline continues so the morning summary is complete.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
source /home/tim/venvs/llm/bin/activate 2>/dev/null || true
export PYTHONPATH=. PYTHONUNBUFFERED=1

RUN=rdr_padic_46Mactive
LANE=recursive_depth_router_padic
CKDIR=research/checkpoints/$RUN
FINAL_CKPT=$CKDIR/${RUN}_${LANE}_step100000.pt
REPORTS=research/reports
PROBES=$REPORTS/frontier_probes
NOTES=research/notes
TRAIN_PID=926352
LOG=$REPORTS/${RUN}_finish.log
mkdir -p "$PROBES" "$NOTES"

say(){ echo "[finish $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "waiting for final checkpoint $FINAL_CKPT (or trainer pid $TRAIN_PID exit)..."
while true; do
  [ -f "$FINAL_CKPT" ] && break
  kill -0 "$TRAIN_PID" 2>/dev/null || { say "trainer pid gone; scanning for newest ckpt"; break; }
  sleep 60
done
# if the run died before 100k, fall back to the newest checkpoint on disk
if [ ! -f "$FINAL_CKPT" ]; then
  FINAL_CKPT=$(ls -t "$CKDIR"/*.pt 2>/dev/null | head -1)
  say "using newest available checkpoint: $FINAL_CKPT"
fi
sleep 25  # let the final torch.save flush
[ -f "$FINAL_CKPT" ] || { say "FATAL: no checkpoint found, aborting"; exit 1; }
say "final checkpoint: $FINAL_CKPT"

# 1. clean low-variance val PPL (200 batches — the number the 8-batch loop couldn't give)
say "clean 200-batch val PPL (eval-only)..."
python -u research/tools/native_adaptive_hydra_train.py \
  --lane "$LANE" --eval-only --load-checkpoint "$FINAL_CKPT" \
  --dim 640 --n-blocks 8 --batch 16 --seq-len 512 --eval-batches 200 \
  --device cuda --out "$REPORTS/${RUN}_clean_ppl.jsonl" \
  >>"$LOG" 2>&1 || say "WARN: clean-ppl step failed (non-fatal)"

# 2. capability probe battery -> frontier_probes/<run>_seed0_post_eval.json (ingest-named)
say "capability probe battery (induction/ar_validation/binding_v2/gmqar)..."
python -u -m research.tools.eval_trained_checkpoint \
  --mixer "$LANE" --dim 640 --n-blocks 8 \
  --checkpoint "$FINAL_CKPT" \
  --output "$PROBES/${RUN}_seed0_post_eval.json" \
  --device cuda --probe-timeout 1800 \
  >>"$LOG" 2>&1 || say "WARN: probe battery failed (non-fatal)"

# 3. full BLiMP -> frontier_probes/<run>_blimp.json (lane auto-resolved from ckpt payload)
say "full BLiMP..."
python -u research/tools/eval_checkpoints_blimp.py "$FINAL_CKPT" \
  --n-per-subtask 1000 --device cuda \
  --out "$PROBES/${RUN}_blimp.json" \
  >>"$LOG" 2>&1 || say "WARN: BLiMP failed (non-fatal)"

# 4. zero-shot gMQAR with extended grid (associative-recall breaking point)
say "zero-shot gMQAR (extended grid -> 128 pairs)..."
python -u research/tools/calibrated_ar_probe.py \
  --mode zeroshot --device cuda --gpu-frac 0.9 \
  --checkpoint "$FINAL_CKPT" --ckpt-label "${RUN}@100k" \
  --token-pool 2048 --max-pairs 128 \
  --out "$REPORTS/${RUN}_gmqar_final.jsonl" \
  >>"$LOG" 2>&1 || say "WARN: gMQAR failed (non-fatal)"

# 5. ingest into runs.db (scale_run_evals / scale_run_blimp / scale_run_probe_metrics)
say "ingest into runs.db..."
python -u research/tools/ingest_scale_runs.py >>"$LOG" 2>&1 || say "WARN: ingest failed (non-fatal)"

# 6. keeper comparison table (recursion-depth column) -> Obsidian-synced note
say "building keeper comparison table..."
ACTIVE_M=42.7; TOTAL_M=106.9; TOKENS_M=819.2
python -u -m research.tools.compare_scale_runs \
  --new-post-eval "$PROBES/${RUN}_seed0_post_eval.json" \
  --new-blimp "$PROBES/${RUN}_blimp.json" \
  --new-label "$RUN" --new-active-m $ACTIVE_M --new-total-m $TOTAL_M \
  --new-tokens-m $TOKENS_M --new-recursion 1 \
  --out "$NOTES/scale_run_comparison_rdr_padic_2026-06-29.md" \
  >>"$LOG" 2>&1 || say "WARN: comparison table failed (non-fatal)"

# 7. sync notes to Obsidian + rebuild the prose index
say "syncing Obsidian + rebuilding note index..."
python -u .claude/hooks/obsidian_sync.py sync-notes >>"$LOG" 2>&1 || say "WARN: obsidian sync failed"
python -u research/tools/index_notes.py >>"$LOG" 2>&1 || say "WARN: index rebuild failed"

# 8. sentinel for the re-invoked agent to detect completion
say "DONE — artifacts:"
say "  post_eval : $PROBES/${RUN}_seed0_post_eval.json"
say "  blimp     : $PROBES/${RUN}_blimp.json"
say "  gmqar     : $REPORTS/${RUN}_gmqar_final.jsonl"
say "  clean_ppl : $REPORTS/${RUN}_clean_ppl.jsonl"
say "  comparison: $NOTES/scale_run_comparison_rdr_padic_2026-06-29.md"
touch "$REPORTS/${RUN}_finish.DONE"
