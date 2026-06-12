"""Cheap experiment: RMSNorm on slot VALUES before the read (codex's recommendation).

The locked read normalizes q and slot_key (for addressing) but reads slot_val raw; value
magnitudes can drift over long sequences, which is where the interference axis (L256) is weak
(~0.30 while other axes are 0.5-0.95). RMSNorm-ing the stored values before the weighted read
may stabilize that. Implemented as a STANDALONE SUBCLASS overriding `_read_slots` only — it does
NOT edit codex's lane, so it cannot collide with codex's pending diff.

Graded vs the locked base on HARD tasks, 10 seeds.
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


def _locked_factory(*, normalize_slot_values: bool):
    def make(d: int):
        memory_dim = max(4, ((7 * d) // 32) * 4)
        return MultiHeadSlotTableMemoryLane(
            d,
            memory_dim=memory_dim,
            use_delta_update=False,
            route_from_input=True,
            normalize_slot_values=normalize_slot_values,
        )

    return make


MODELS = {
    "locked_base": _locked_factory(normalize_slot_values=False),
    "locked_rmsnorm_val": _locked_factory(normalize_slot_values=True),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-count", type=int, default=10)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--variant",
        choices=tuple(MODELS),
        action="append",
        help="Run only the selected current-lane variant; repeat to select multiple.",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("research/reports/exp_slotval_rmsnorm.json")
    )
    args = ap.parse_args()

    rows: dict[str, dict] = {}
    selected = args.variant or list(MODELS)
    for name in selected:
        fac = MODELS[name]
        per_task: dict[str, dict] = {}
        for task in HARD_BINDING_VALIDITY_TASKS:
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
            per_task[task.name] = {
                "mean": statistics.fmean(accs),
                "stdev": statistics.pstdev(accs),
            }
        mean = statistics.fmean(v["mean"] for v in per_task.values())
        rows[name] = {"task_mean": mean, "per_task": per_task}
        short = {
            k.replace("hard_", "").split("_")[0]: round(v["mean"], 3)
            for k, v in per_task.items()
        }
        print(f"  {name:20s} mean={mean:.3f}  {short}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"seeds": args.seed_count, "rows": rows}, indent=2))
    if {"locked_base", "locked_rmsnorm_val"} <= rows.keys():
        task = "hard_interference_16_pairs_8_queries_256"
        base = rows["locked_base"]["per_task"][task]["mean"]
        rmsn = rows["locked_rmsnorm_val"]["per_task"][task]["mean"]
        print(
            f"\ninterference: base={base:.3f} -> "
            f"rmsnorm_val={rmsn:.3f} (Δ={rmsn - base:+.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
