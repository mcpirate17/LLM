"""Does the nano slot-memory ceiling lift with more training steps?

Trains slot_table_mh (frozen baseline) and softmax_4h (bar) from scratch for an increasing
number of steps on the HARD tasks and reports per-step task-mean. If accuracy keeps climbing,
800 steps is undertrained; if it plateaus, the ~0.26 nano ceiling is a capacity/scale limit,
not a training-budget artifact — which decides whether "add steps" or "add scale" is the lever.
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

STEPS_LADDER = [400, 800, 1600, 3200, 6400]


def _slot_mh(d: int):
    return MultiHeadSlotTableMemoryLane(
        d,
        memory_dim=56,
        n_slots=8,
        n_heads=4,
        use_null_write=False,
        use_composer=False,
        use_delta_update=False,
        normalize_read=False,
    )


MODELS = {
    "slot_table_mh": _slot_mh,
    "softmax_4h": lambda d: MultiHeadCausalAttention(d, n_heads=4),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-count", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--out", type=Path, default=Path("research/reports/steps_ladder_slot_mh.json")
    )
    args = ap.parse_args()

    rows: dict[str, dict] = {}
    for name, fac in MODELS.items():
        rows[name] = {}
        for steps in STEPS_LADDER:
            task_means = []
            for task in HARD_BINDING_VALIDITY_TASKS:
                accs = [
                    run_binding_validity_task(
                        fac,
                        task,
                        mixer_label=name,
                        dim=64,
                        n_train_steps=steps,
                        seed=seed,
                        device=args.device,
                    ).eval_accuracy
                    for seed in range(args.seed_count)
                ]
                task_means.append(statistics.fmean(accs))
            mean = statistics.fmean(task_means)
            rows[name][steps] = round(mean, 4)
            print(f"  {name:14s} steps={steps:<5d} task_mean={mean:.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"steps_ladder": STEPS_LADDER, "seeds": args.seed_count, "rows": rows},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
