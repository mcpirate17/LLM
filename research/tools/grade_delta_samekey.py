"""Does the delta (last-write-wins) update actually fix same_key_overwrite?

Codex screened `use_delta_update` only on the HARD ladder (which omits same_key_overwrite) and
saw a net drop, so the question delta was meant to answer was never measured. This grades
delta-ON vs delta-OFF on the DEFAULT ladder, which *includes* episodic_same_key_overwrite — the
axis minimax#1 targets (emit the LATEST value of a repeated key, not the cumulative mean).

Frozen otherwise (all other improvement flags off) so the delta effect is isolated.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from component_fab.generator.memory_primitives import MultiHeadSlotTableMemoryLane
from component_fab.harness.binding_validity import (
    DEFAULT_BINDING_VALIDITY_TASKS,
    run_binding_validity_task,
)

MEMORY_DIM = 56
N_HEADS = 4
N_SLOTS = 8


def _factory(delta: bool):
    def make(d: int):
        return MultiHeadSlotTableMemoryLane(
            d,
            memory_dim=MEMORY_DIM,
            n_slots=N_SLOTS,
            n_heads=N_HEADS,
            use_delta_update=delta,
            use_null_write=False,
            use_composer=False,
            normalize_read=False,
        )

    return make


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seed-count", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--out", type=Path, default=Path("research/reports/grade_delta_samekey.json")
    )
    args = ap.parse_args()

    rows: dict[str, dict] = {}
    for delta in (False, True):
        label = "delta_on" if delta else "delta_off"
        factory = _factory(delta)
        per_task: dict[str, dict] = {}
        for task in DEFAULT_BINDING_VALIDITY_TASKS:
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
        rows[label] = {
            "task_mean": statistics.fmean(v["mean"] for v in per_task.values()),
            "per_task": per_task,
        }
        same = per_task["episodic_same_key_overwrite"]["mean"]
        print(
            f"{label:9s} same_key_overwrite={same:.3f}  "
            f"mean={rows[label]['task_mean']:.3f}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"seeds": args.seed_count, "rows": rows}, indent=2))
    off = rows["delta_off"]["per_task"]["episodic_same_key_overwrite"]["mean"]
    on = rows["delta_on"]["per_task"]["episodic_same_key_overwrite"]["mean"]
    print(
        f"\nsame_key_overwrite: delta_off={off:.3f} -> delta_on={on:.3f} (Δ={on - off:+.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
