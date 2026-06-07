"""Validate two optimization levers from the cross-axis matrix (2026-06-07).

1. hier_compress timescale widening: does n_levels 4->8 (longest summary period
   8 -> 128) fix long_gap (was 0.21)? Check it doesn't break simple recall /
   state-tracking.
2. ddecay long_gap=1.00 was 1 seed — confirm across 3 seeds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from component_fab.generator.memory_primitives import (
    DataDependentDecayMemoryLane,
    HierarchicalResidualCompressorLane,
)
from component_fab.harness.harder_binding_tasks import (
    default_hard_binding_tasks,
    run_one_task_checkpoints,
)
from component_fab.harness.state_tracking_suite import score_state_tracking

OUT = Path("research/reports/_opt_validate.json")
STEPS = 3000
t0 = time.monotonic()
res: dict[str, object] = {}


def bind_acc(factory, task_names, seeds=(0,)):
    out = {}
    for tn in task_names:
        accs = []
        for s in seeds:
            task = {t.name: t for t in default_hard_binding_tasks(seed=s)}[tn]
            rows = run_one_task_checkpoints(
                factory,
                task,
                eval_at_steps=(STEPS,),
                mixer_label="x",
                dim=64,
                seed=s,
                device="cuda",
            )
            accs.append(rows[STEPS].eval_accuracy)
        out[tn] = sum(accs) / len(accs)
    return out


# 1. hier_compress n_levels = 8 on long_gap + 2 simple tasks (1 seed)
def hc8(d):
    return HierarchicalResidualCompressorLane(d, n_levels=8)


res["hier_compress_n8_bind"] = bind_acc(
    hc8, ["long_gap_recall", "multi_query_kv_recall", "distractor_kv_recall"]
)
res["hier_compress_n8_state"] = score_state_tracking(
    hc8, dim=32, seq_len=32, n_steps=400, seeds=(0, 1, 2), device="cpu"
)["per_axis"]
print(
    f"hier n8 bind={res['hier_compress_n8_bind']} state={res['hier_compress_n8_state']} ({time.monotonic() - t0:.0f}s)",
    flush=True,
)

# 2. ddecay long_gap, 3 seeds
res["ddecay_long_gap_3seed"] = bind_acc(
    lambda d: DataDependentDecayMemoryLane(d), ["long_gap_recall"], seeds=(0, 1, 2)
)
print(
    f"ddecay long_gap 3seed={res['ddecay_long_gap_3seed']} ({time.monotonic() - t0:.0f}s)",
    flush=True,
)

OUT.write_text(json.dumps(res, indent=1))
print(f"saved {OUT} ({time.monotonic() - t0:.0f}s)")
