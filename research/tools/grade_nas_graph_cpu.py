"""Grade one cached NAS topology against the frontier baselines on CPU.

Standalone tester for a single ``component_fab/catalog/nas_graphs/<fp>.json``
topology that has not been graded into the ledger. Compiles the graph as the
candidate mixer and runs it through the Tier-2 hard-binding suite against
GPT-2 / Mamba / Mamba2 / softmax, all at the SAME dim (fair comparison). CPU
only and thread-limited so it never disturbs a live GPU training run.

Usage::

    CUDA_VISIBLE_DEVICES="" python -m research.tools.grade_nas_graph_cpu \
        --fingerprint 0a4cadb3bdea2f66 --dim 32 --steps 600
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from pathlib import Path

import torch

from component_fab.harness.harder_binding_tasks import (
    default_hard_binding_tasks,
    run_harder_binding_suite,
)
from component_fab.harness.tiny_lm import FRONTIER_BASELINE_NAMES
from research.synthesis.compiler import compile_graph
from research.synthesis.serializer import graph_from_json
from research.tools.run_tier2_binding_cohort import _summarise_per_task

_REPO = Path(__file__).resolve().parents[2]
_CACHE = _REPO / "component_fab" / "catalog" / "nas_graphs"
_REPORTS = _REPO / "research" / "reports"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fingerprint", required=True, type=str)
    p.add_argument("--dim", default=32, type=int)
    p.add_argument("--n-blocks", default=2, type=int)
    p.add_argument("--steps", default=600, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--threads", default=4, type=int)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    torch.set_num_threads(max(1, args.threads))

    js = (_CACHE / f"{args.fingerprint}.json").read_text(encoding="utf-8")
    label = f"nas_grammar_{args.fingerprint[:12]}"

    def candidate_factory(d: int) -> torch.nn.Module:
        return compile_graph(graph_from_json(js, model_dim=d), use_ir=True)

    print(
        f"grading {label} (dim={args.dim}, {args.n_blocks} blocks, {args.steps} "
        f"steps) vs {', '.join(FRONTIER_BASELINE_NAMES)} on CPU"
    )
    t0 = time.monotonic()
    suite = run_harder_binding_suite(
        candidate_factory,
        label,
        tasks=default_hard_binding_tasks(seed=args.seed),
        dim=args.dim,
        n_blocks=args.n_blocks,
        n_train_steps=args.steps,
        seed=args.seed,
        baseline_names=FRONTIER_BASELINE_NAMES,
    )
    per_task = {name: _summarise_per_task(rows) for name, rows in suite.items()}
    n_beats = sum(1 for v in per_task.values() if v.get("beats"))
    elapsed = time.monotonic() - t0

    print(f"\n=== {label} vs frontier  ({elapsed:.0f}s) ===")
    print(f"{'task':28} {'cand':>8} {'best_base':>10} {'delta':>9}  beats")
    for name, v in per_task.items():
        print(
            f"{name:28} {v['candidate_eval_acc']:>8.4f} "
            f"{v['baseline_max']:>10.4f} {v['delta']:>+9.4f}  "
            f"{'YES' if v['beats'] else ''}"
        )
    print(f"\nbeats frontier on {n_beats}/{len(per_task)} tasks")

    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = _REPORTS / f"nas_grade_{args.fingerprint[:12]}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "label": label,
                "fingerprint": args.fingerprint,
                "dim": args.dim,
                "n_blocks": args.n_blocks,
                "steps": args.steps,
                "baselines": list(FRONTIER_BASELINE_NAMES),
                "n_beats": n_beats,
                "n_tasks": len(per_task),
                "per_task": per_task,
                "elapsed_s": round(elapsed, 1),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"[report → {out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
