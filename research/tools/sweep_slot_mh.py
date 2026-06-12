"""Nano head/slot sweep for MultiHeadSlotTableMemoryLane on the hardened binding tasks.

Holds memory_dim fixed (so q/k/v/out params stay ~constant ~16K, matched to softmax_4h)
and sweeps (n_heads, n_slots) to find the best config before committing compute. The
softmax_4h bar on this benchmark is task-mean 0.244 (10 seeds).
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

# (n_heads, n_slots); memory_dim fixed so total params stay ~16K (matched to softmax_4h).
CONFIGS: tuple[tuple[int, int], ...] = (
    (2, 8),
    (4, 8),
    (8, 8),
    (16, 8),
    (8, 4),
    (8, 16),
)
MEMORY_DIM = 56


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seed-count", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--out", type=Path, default=Path("research/reports/sweep_slot_mh.json")
    )
    args = ap.parse_args()

    started = time.time()
    rows: dict[str, dict] = {}
    for n_heads, n_slots in CONFIGS:
        label = f"mh{n_heads}_s{n_slots}"

        def factory(d, h=n_heads, s=n_slots):
            return MultiHeadSlotTableMemoryLane(
                d, memory_dim=MEMORY_DIM, n_slots=s, n_heads=h
            )

        params = sum(p.numel() for p in factory(64).parameters())
        per_task: dict[str, dict] = {}
        for task in HARD_BINDING_VALIDITY_TASKS:
            accs = []
            for seed in range(args.seed_count):
                r = run_binding_validity_task(
                    factory,
                    task,
                    mixer_label=label,
                    dim=64,
                    n_train_steps=args.steps,
                    seed=seed,
                    device=args.device,
                )
                accs.append(r.eval_accuracy)
            per_task[task.name] = {
                "mean": statistics.fmean(accs),
                "stdev": statistics.pstdev(accs),
            }
        task_mean = statistics.fmean(v["mean"] for v in per_task.values())
        rows[label] = {
            "n_heads": n_heads,
            "n_slots": n_slots,
            "params": params,
            "task_mean": task_mean,
            "per_task": per_task,
        }
        print(
            f"{label:10s} heads={n_heads:2d} slots={n_slots:2d} params={params:5d} "
            f"task_mean={task_mean:.3f}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "memory_dim": MEMORY_DIM,
                "steps": args.steps,
                "seeds": args.seed_count,
                "softmax_4h_bar": 0.244,
                "rows": rows,
                "elapsed_s": round(time.time() - started, 1),
            },
            indent=2,
        )
    )
    best = max(rows.items(), key=lambda kv: kv[1]["task_mean"])
    print(f"\nBEST: {best[0]} task_mean={best[1]['task_mean']:.3f} (bar 0.244)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
