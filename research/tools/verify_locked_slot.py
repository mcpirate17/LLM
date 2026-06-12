"""Independent verification of codex's locked composer+normalized slot lane.

Reproduces the locked `slot_table_memory` config (exactly as the dispatcher builds it) on a
SEPARATE grade harness, at 10 seeds, on the HARD tasks. Two step budgets:
  * 800  -- reproduce codex's reported 0.628 (rules out a harness artifact).
  * 3200 -- the convergence test. The plain slot lane beat softmax at 800 only because it
            converges faster (softmax caught up by 3200). If the locked lane STILL holds ~0.6 at
            3200 while softmax is pinned ~0.26, it is a real ceiling break, not faster convergence.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from component_fab.generator.memory_primitives import MultiHeadSlotTableMemoryLane
from component_fab.harness.binding_validity import (
    HARD_BINDING_VALIDITY_TASKS,
    run_binding_validity_task,
)
from component_fab.harness.tiny_lm import MultiHeadCausalAttention


def _locked(d: int):
    # Exactly as code_generator.py dispatches "slot_table_memory".
    memory_dim = max(4, ((7 * d) // 32) * 4)
    return MultiHeadSlotTableMemoryLane(
        d, memory_dim=memory_dim, use_delta_update=False, route_from_input=True
    )


MODELS = {
    "locked_slot": _locked,
    "softmax_4h": lambda d: MultiHeadCausalAttention(d, n_heads=4),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-count", type=int, default=10)
    ap.add_argument("--steps", type=int, nargs="+", default=[800, 3200])
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--out", type=Path, default=Path("research/reports/verify_locked_slot.json")
    )
    args = ap.parse_args()

    rows: dict[str, dict] = {}
    for steps in args.steps:
        for name, fac in MODELS.items():
            key = f"{name}@{steps}"
            per_task: dict[str, dict] = {}
            for task in HARD_BINDING_VALIDITY_TASKS:
                accs = [
                    run_binding_validity_task(
                        fac,
                        task,
                        mixer_label=key,
                        dim=64,
                        n_train_steps=steps,
                        seed=seed,
                        device=args.device,
                    ).eval_accuracy
                    for seed in range(args.seed_count)
                ]
                per_task[task.name] = {
                    "mean": statistics.fmean(accs),
                    "stdev": statistics.pstdev(accs),
                }
            mean = statistics.fmean(v["mean"] for v in per_task.values())
            rows[key] = {"steps": steps, "task_mean": mean, "per_task": per_task}
            short = {
                k.replace("hard_", "").split("_")[0]: round(v["mean"], 3)
                for k, v in per_task.items()
            }
            print(f"  {key:18s} mean={mean:.3f}  {short}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"seeds": args.seed_count, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
