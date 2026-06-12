"""Is the locked lane's 0.99@3200 robust, or does it crack at higher difficulty?

binding_validity HARD (16 pairs / 8 queries / seq256) is now SATURATED by the locked lane
(0.99@3200), so it no longer discriminates. This scales difficulty up — more pairs, more queries,
longer sequences — on the two hardest axes (interference, compositional), graded at the 3200-step
convergence budget where the lane solved the standard task. softmax_4h is the bar.

If the locked lane holds ~0.9+ while softmax stays ~0.26 as difficulty climbs, the content-addressing
advantage is robust and worth scaling. If the lane degrades, that exposes the real ceiling and what
to fix before the 40M run.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from component_fab.generator.memory_primitives import MultiHeadSlotTableMemoryLane
from component_fab.harness.binding_validity import (
    BindingValidityTask,
    run_binding_validity_task,
)
from component_fab.harness.tiny_lm import MultiHeadCausalAttention

# n_values fixed at 8 (chance 0.125) for comparability across the ladder.
TASKS = [
    BindingValidityTask(
        "intf_16p_8q_256",
        "distinct_key_interference",
        seq_len=256,
        n_keys=16,
        n_values=8,
        n_pairs=16,
        n_queries=8,
        scatter_writes=True,
    ),
    BindingValidityTask(
        "intf_24p_12q_384",
        "distinct_key_interference",
        seq_len=384,
        n_keys=24,
        n_values=8,
        n_pairs=24,
        n_queries=12,
        scatter_writes=True,
    ),
    BindingValidityTask(
        "intf_32p_16q_512",
        "distinct_key_interference",
        seq_len=512,
        n_keys=32,
        n_values=8,
        n_pairs=32,
        n_queries=16,
        scatter_writes=True,
    ),
    BindingValidityTask(
        "comp_16p_8q_256",
        "episodic_compositional",
        seq_len=256,
        n_entities=8,
        n_attributes=8,
        n_values=8,
        n_pairs=16,
        n_queries=8,
        scatter_writes=True,
    ),
    BindingValidityTask(
        "comp_32p_16q_512",
        "episodic_compositional",
        seq_len=512,
        n_entities=8,
        n_attributes=8,
        n_values=8,
        n_pairs=32,
        n_queries=16,
        scatter_writes=True,
    ),
]


def _locked(d: int):
    memory_dim = max(4, ((7 * d) // 32) * 4)
    return MultiHeadSlotTableMemoryLane(
        d,
        memory_dim=memory_dim,
        use_delta_update=False,
        route_from_input=True,
        normalize_slot_values=True,
    )


MODELS = {
    "locked_slot": _locked,
    "softmax_4h": lambda d: MultiHeadCausalAttention(d, n_heads=4),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-count", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--out", type=Path, default=Path("research/reports/exp_harder_binding.json")
    )
    args = ap.parse_args()

    rows: dict[str, dict] = {}
    for task in TASKS:
        rows[task.name] = {"chance": task.chance_accuracy}
        for name, fac in MODELS.items():
            accs = [
                run_binding_validity_task(
                    fac,
                    task,
                    mixer_label=name,
                    dim=64,
                    n_train_steps=args.steps,
                    seed=seed,
                    device=args.device,
                ).eval_accuracy
                for seed in range(args.seed_count)
            ]
            rows[task.name][name] = {
                "mean": statistics.fmean(accs),
                "stdev": statistics.pstdev(accs),
            }
        print(
            f"  {task.name:18s} locked={rows[task.name]['locked_slot']['mean']:.3f} "
            f"softmax={rows[task.name]['softmax_4h']['mean']:.3f} (chance {task.chance_accuracy:.3f})",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"seeds": args.seed_count, "steps": args.steps, "rows": rows}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
