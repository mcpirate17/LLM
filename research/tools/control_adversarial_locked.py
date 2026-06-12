"""Shortcut control for the locked slot lane's 0.99 binding_validity result.

binding_validity@3200 gave locked_slot 0.991 (vs softmax 0.26) — a "solved" score that, per
the campaign's own lesson, must clear a randomized/length-generalization control before being
believed. The adversarial-retention suite resamples key/value assignments per example (no global
memorization) and includes `extrapolate_one_128_to_256` (train at 128, EVAL at 256): a genuine
content-addressable memory generalizes; a positional/structural shortcut collapses at the longer
eval length. If the locked lane holds on extrapolation + high-load while softmax does not, the
binding result is genuine, not an artifact.

Standalone: reuses the harness, builds the locked config exactly as the dispatcher does, no edits
to shared graders or lanes.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from component_fab.generator.memory_primitives import MultiHeadSlotTableMemoryLane
from component_fab.harness.adversarial_retention import (
    DEFAULT_RETENTION_TASKS,
    run_retention_task,
)
from component_fab.harness.tiny_lm import MultiHeadCausalAttention


def _locked(d: int):
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
    ap.add_argument("--seed-count", type=int, default=5)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("research/reports/control_adversarial_locked.json"),
    )
    args = ap.parse_args()

    rows: dict[str, dict] = {}
    for name, fac in MODELS.items():
        rows[name] = {}
        for task in DEFAULT_RETENTION_TASKS:
            accs = []
            for seed in range(args.seed_count):
                r = run_retention_task(
                    fac,
                    task,
                    mixer_label=name,
                    dim=64,
                    n_train_steps=args.steps,
                    seed=seed,
                    device=args.device,
                )
                accs.append(r.eval_accuracy)
            rows[name][task.name] = {
                "mean": statistics.fmean(accs),
                "stdev": statistics.pstdev(accs),
                "chance": r.chance_accuracy,
            }
            print(
                f"  {name:12s} {task.name:28s} acc={statistics.fmean(accs):.3f} "
                f"(chance {r.chance_accuracy:.3f})",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"seeds": args.seed_count, "rows": rows}, indent=2))
    print("\n--- extrapolation (train128->eval256), the key control ---")
    for name in MODELS:
        e = rows[name]["extrapolate_one_128_to_256"]
        print(f"  {name:12s} extrapolate={e['mean']:.3f} (chance {e['chance']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
