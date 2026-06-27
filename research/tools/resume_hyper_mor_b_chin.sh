#!/usr/bin/env bash
# Resume the hyper_mor_b_chin chinchilla run from its step-40000 checkpoint.
# Faithful continuation of the original trajectory: every hyperparameter is taken
# from the run's `start` record; --ponder-weight 0 (LM loss alone decides MoR
# depth) is set explicitly because the lane __init__ default is 1e-2 (which
# collapses depth), and the step-40000 depth plateau (mean ~3.2) is the ponder=0
# signature. ponder_weight is a runtime float attr, not stored in the checkpoint.
#
# --append keeps the existing jsonl (the trainer DELETES it without --append); the
# jsonl + console.log were already truncated to step<=40000 before this resume.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
source /home/tim/venvs/llm/bin/activate 2>/dev/null || true

REPORTS=research/reports
LANE=hyper_mor_surprise_refine_mlp258_native_semiring_adapt_bilane_m32_g0_t1_b1_l0_h2_r7_surprise_memory
CKPT=$REPORTS/hyper_mor_b_chin_ckpts/hyper_mor_b_chin_${LANE}_step040000.pt

export PYTHONPATH=.
export PYTHONUNBUFFERED=1

exec python -u -m research.tools.native_adaptive_hydra_train \
  --lane "$LANE" \
  --dataset codex_ffw60_chat30_pleias10_local \
  --val-dataset codex_ffw60_chat30_pleias10_local \
  --run-label hyper_mor_b_chin \
  --dim 736 --n-blocks 8 --steps 125000 \
  --batch 44 --seq-len 256 \
  --lr 3e-4 --optimizer muon --muon-lr 0.02 \
  --warmup-steps 800 --min-lr-frac 0.1 \
  --vocab-size 100277 \
  --ponder-weight 0 \
  --log-every 50 --eval-every 2000 --save-every 5000 \
  --max-recoveries 3 \
  --device cuda \
  --load-checkpoint "$CKPT" \
  --checkpoint-dir "$REPORTS/hyper_mor_b_chin_ckpts" \
  --out "$REPORTS/hyper_mor_b_chin.jsonl" --append \
  2>&1 | tee -a "$REPORTS/hyper_mor_b_chin.console.log"
