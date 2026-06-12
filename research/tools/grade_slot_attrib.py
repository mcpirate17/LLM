"""Two grading-fix / attribution experiments on the FROZEN baseline slot lane.

Holds MultiHeadSlotTableMemoryLane at its baseline (all write/read improvement flags OFF,
so this is reproducible and independent of codex's in-flight variant screen) and crosses:

  * n_slots in {8, 16}        -- 8 = current HARD grade (collides: HARD has 16 keys),
                                 16 = n_slots==n_keys, the true (uncollided) ceiling.
  * grouped_router in {0, 1}  -- 0 = joint router over all heads' concat keys (current),
                                 1 = per-head router. Answers whether the multi-head win is
                                     head-independence or just a larger joint router.

softmax_4h bar on this hardened benchmark = task-mean 0.244 (10 seeds).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from component_fab.generator.memory_primitives import MultiHeadSlotTableMemoryLane
from component_fab.harness.binding_validity import (
    HARD_BINDING_VALIDITY_TASKS,
    run_binding_validity_task,
)

MEMORY_DIM = 56
N_HEADS = 4
# (n_slots, grouped_router)
CONFIGS: tuple[tuple[int, bool], ...] = (
    (8, False),
    (16, False),
    (8, True),
    (16, True),
)


def _factory(n_slots: int, grouped: bool):
    def make(d: int):
        return MultiHeadSlotTableMemoryLane(
            d,
            memory_dim=MEMORY_DIM,
            n_slots=n_slots,
            n_heads=N_HEADS,
            grouped_router=grouped,
            use_null_write=False,
            use_composer=False,
            use_delta_update=False,
            normalize_read=False,
        )

    return make


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seed-count", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--out", type=Path, default=Path("research/reports/grade_slot_attrib.json")
    )
    args = ap.parse_args()

    started = time.time()
    rows: dict[str, dict] = {}
    for n_slots, grouped in CONFIGS:
        label = f"s{n_slots}_{'grouped' if grouped else 'joint'}"
        factory = _factory(n_slots, grouped)
        params = sum(p.numel() for p in factory(64).parameters())
        per_task: dict[str, dict] = {}
        for task in HARD_BINDING_VALIDITY_TASKS:
            accs = [
                run_binding_validity_task(
                    factory,
                    task,
                    mixer_label=label,
                    dim=64,
                    n_train_steps=args.steps,
                    seed=seed,
                    device=args.device,
                ).eval_accuracy
                for seed in range(args.seed_count)
            ]
            per_task[task.name] = {
                "mean": statistics.fmean(accs),
                "stdev": statistics.pstdev(accs),
            }
        task_mean = statistics.fmean(v["mean"] for v in per_task.values())
        rows[label] = {
            "n_slots": n_slots,
            "grouped_router": grouped,
            "params": params,
            "task_mean": task_mean,
            "per_task": per_task,
        }
        print(
            f"{label:14s} n_slots={n_slots:2d} grouped={int(grouped)} "
            f"params={params:5d} task_mean={task_mean:.3f}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "memory_dim": MEMORY_DIM,
                "n_heads": N_HEADS,
                "steps": args.steps,
                "seeds": args.seed_count,
                "softmax_4h_bar": 0.244,
                "rows": rows,
                "elapsed_s": round(time.time() - started, 1),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
