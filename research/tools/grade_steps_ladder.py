"""Steps ladder on the HARDENED (random-query) corrected tasks.

The matched grades showed every model collapses to ~0.5 on hardened tasks at 800
steps — the prior 1.000s were a positional shortcut. Open question: does genuine
random-query content retrieval become solvable with MORE TRAINING, and does
anything separate? This trains the top candidates on the two hard tasks
(interference + compositional, where all sit ~0.3) across a steps ladder.

Orthogonal to codex's pair/load/length ladder (this varies STEPS, not difficulty).
Matched dim64; legendre/slot at ~16K params. 1 seed per rung (directional).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from component_fab.generator.memory_primitives import LegendreSSMLane
from component_fab.harness.binding_validity import (
    BINDING_VALIDITY_VERSION,
    DEFAULT_BINDING_VALIDITY_TASKS,
    run_binding_validity_task,
)
from component_fab.harness.tiny_lm import (
    MultiHeadCausalAttention,
    lane_factory_for_baseline,
)
from research.tools.gemini_slot_snapshot import GeminiSlotMemoryLane

_REPORT = Path(__file__).resolve().parents[2] / "research" / "reports"
DIM = 64
STEPS_LADDER = [800, 2000, 5000]
HARD_TASKS = {"episodic_distinct_key_interference", "episodic_compositional"}

MODELS = {
    "softmax_4h": lambda d: MultiHeadCausalAttention(d, n_heads=4),
    "mamba2": lambda d: lane_factory_for_baseline("mamba2")(d),
    "legendre_ssm": lambda d: LegendreSSMLane(d, state_dim=256),
    "gemini_slot": lambda d: GeminiSlotMemoryLane(d, memory_dim=40),
}


def main() -> int:
    tasks = [t for t in DEFAULT_BINDING_VALIDITY_TASKS if t.name in HARD_TASKS]
    started = time.monotonic()
    rows: dict[str, dict] = {}
    for name, fac in MODELS.items():
        rows[name] = {}
        for steps in STEPS_LADDER:
            per_task = {}
            for task in tasks:
                r = run_binding_validity_task(
                    fac,
                    task,
                    mixer_label=name,
                    dim=DIM,
                    n_train_steps=steps,
                    seed=0,
                    device="cuda",
                )
                per_task[task.name.replace("episodic_", "")] = round(r.eval_accuracy, 3)
            rows[name][steps] = per_task
            print(
                f"  {name:14s} steps={steps:<5d} "
                + " ".join(f"{k[:8]}={v:.2f}" for k, v in per_task.items())
                + f"  ({time.monotonic() - started:.0f}s)",
                flush=True,
            )
    out = _REPORT / "steps_ladder_hardened.json"
    out.write_text(
        json.dumps(
            {
                "task_semantics_version": BINDING_VALIDITY_VERSION,
                "steps_ladder": STEPS_LADDER,
                "rows": rows,
            },
            indent=1,
        )
    )
    print(f"\n[report -> {out}]  ({time.monotonic() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
