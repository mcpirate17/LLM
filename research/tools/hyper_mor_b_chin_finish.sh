#!/usr/bin/env bash
# Post-training finish-handler for hyper_mor_b_chin. Blocks until the run emits
# its step-125000 'done' (jsonl event OR the final checkpoint appears), then:
#   1. stops the report watcher service,
#   2. runs the post-S1 probe battery (BLiMP + AR + binding + induction) on the
#      ANNEALED final checkpoint, probe-timeout scaled for a 144M model,
#   3. runs zero-shot gMQAR on the final checkpoint with an EXTENDED grid
#      (n_pairs up to 128) to find the true associative-recall breaking point.
# Idempotent-ish: safe to re-run; it just re-evaluates the final checkpoint.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
source /home/tim/venvs/llm/bin/activate 2>/dev/null || true

REPORTS=research/reports
LANE=hyper_mor_surprise_refine_mlp258_native_semiring_adapt_bilane_m32_g0_t1_b1_l0_h2_r7_surprise_memory
JSONL=$REPORTS/hyper_mor_b_chin.jsonl
FINAL_CKPT=$REPORTS/hyper_mor_b_chin_ckpts/hyper_mor_b_chin_${LANE}_step125000.pt
export PYTHONPATH=. PYTHONUNBUFFERED=1

echo "[finish] waiting for step-125000 done / final checkpoint..."
while true; do
  grep -q '"event": "done"' "$JSONL" 2>/dev/null && break
  [ -f "$FINAL_CKPT" ] && break
  sleep 30
done
# give the final torch.save a moment to flush
sleep 20
echo "[finish] training complete; final checkpoint: $FINAL_CKPT"

echo "[finish] stopping report watcher..."
systemctl --user stop hyper_mor_b_chin_report.service 2>/dev/null || true

echo "[finish] running post-S1 probe battery (BLiMP + AR + binding + induction)..."
python -u -m research.tools.eval_trained_checkpoint \
  --mixer "$LANE" --dim 736 --n-blocks 8 \
  --checkpoint "$FINAL_CKPT" \
  --output "$REPORTS/hyper_mor_b_chin_final_post_eval.json" \
  --device cuda --probe-timeout 1800 \
  2>&1 | tee "$REPORTS/hyper_mor_b_chin_final_post_eval.log"

echo "[finish] running zero-shot gMQAR on final checkpoint (extended grid -> 128 pairs)..."
python -u research/tools/calibrated_ar_probe.py \
  --mode zeroshot --device cuda --gpu-frac 0.9 \
  --checkpoint "$FINAL_CKPT" --ckpt-label "hyper_mor_b_CKPT@125k" \
  --token-pool 2048 --max-pairs 128 \
  --out "$REPORTS/calibrated_ar_probe_final.jsonl" \
  2>&1 | tee "$REPORTS/calibrated_ar_probe_final.log"

echo "[finish] BLiMP overall:"
python - <<'PY'
import json
d=json.load(open("research/reports/hyper_mor_b_chin_final_post_eval.json"))
def find(o,key):
    if isinstance(o,dict):
        for k,v in o.items():
            if k==key and isinstance(v,(int,float)): return v
            r=find(v,key)
            if r is not None: return r
    return None
for k in ("blimp_overall","blimp","overall"):
    v=find(d,k)
    if v is not None: print(f"  BLiMP = {v}"); break
PY
echo "[finish] DONE. Artifacts: hyper_mor_b_chin_final_post_eval.json, calibrated_ar_probe_final.jsonl"
