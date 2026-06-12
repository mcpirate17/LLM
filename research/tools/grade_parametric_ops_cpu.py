"""Grade the P0 parametric op-synthesis mechanisms STANDALONE vs frontier on CPU.

Each StageSpec (Address x Score-norm x Aggregate) is softmax attention at init and
mutates from there during training. This trains each of the 12 mechanisms on its
own (no attention hybrid, per the no-attention-hybrids rule) against the
gpt2/mamba/mamba2/softmax frontier baselines and reports which — if any — beat
frontier. Reuses the baseline-reuse grader so baselines train once.

Usage::

    CUDA_VISIBLE_DEVICES="" python -m research.tools.grade_parametric_ops_cpu \
        --dim 32 --n-blocks 2 --steps 600
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from pathlib import Path

import torch

from component_fab.harness.harder_binding_tasks import default_hard_binding_tasks
from component_fab.harness.tiny_lm import FRONTIER_BASELINE_NAMES
from research.synthesis.parametric_ops import all_stage_specs, build_parametric_mix
from research.tools.grade_ledger_cohort_cpu import (
    _baseline_max_per_task,
    _grade_candidate,
)

_REPORTS = Path(__file__).resolve().parents[2] / "research" / "reports"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", default=32, type=int)
    p.add_argument("--n-blocks", default=2, type=int)
    p.add_argument("--steps", default=600, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--threads", default=6, type=int)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    torch.set_num_threads(max(1, args.threads))

    specs = all_stage_specs()
    tasks = default_hard_binding_tasks(seed=args.seed)
    print(
        f"grading {len(specs)} parametric mechanisms [STANDALONE] (dim={args.dim}, "
        f"{args.steps} steps) vs {', '.join(FRONTIER_BASELINE_NAMES)} on CPU"
    )

    t0 = time.monotonic()
    baseline_max = _baseline_max_per_task(
        tasks, dim=args.dim, n_blocks=args.n_blocks, steps=args.steps, seed=args.seed
    )
    print(f"[baselines trained in {time.monotonic() - t0:.0f}s] {baseline_max}")

    graded: list[dict] = []
    for i, spec in enumerate(specs):
        tc = time.monotonic()
        res = _grade_candidate(
            (lambda _s=spec: lambda d: build_parametric_mix(d, _s)),
            spec.key,
            tasks,
            baseline_max,
            dim=args.dim,
            n_blocks=args.n_blocks,
            steps=args.steps,
            seed=args.seed,
        )
        graded.append({"spec": spec.key, **res})
        print(
            f"[{i + 1}/{len(specs)}] {spec.key:28} beats={res['n_beats']}/6 "
            f"Δ={res['mean_delta']:+.3f} tier2={'PASS' if res['tier2_passed'] else ''} "
            f"({time.monotonic() - tc:.0f}s)"
        )

    graded.sort(
        key=lambda g: (g["tier2_passed"], g["n_beats"], g["mean_delta"]), reverse=True
    )
    n_pass = sum(1 for g in graded if g["tier2_passed"])
    n_any = sum(1 for g in graded if g["n_beats"] > 0)
    elapsed = time.monotonic() - t0

    print(f"\n=== SUMMARY ({len(graded)} mechanisms, {elapsed:.0f}s) ===")
    print(f"tier2 survivors (beat frontier on niche rule): {n_pass}/{len(graded)}")
    print(f"beat frontier on >=1 task:                      {n_any}/{len(graded)}")
    print("\nranked (tier2, n_beats, mean Δ):")
    for g in graded:
        print(
            f"  {'PASS' if g['tier2_passed'] else '    '} beats={g['n_beats']}/6 "
            f"Δ={g['mean_delta']:+.3f}  {g['spec']}"
        )

    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = _REPORTS / f"parametric_ops_grade_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "dim": args.dim,
                "n_blocks": args.n_blocks,
                "steps": args.steps,
                "baselines": list(FRONTIER_BASELINE_NAMES),
                "baseline_max": baseline_max,
                "n_tier2_survivors": n_pass,
                "n_beat_any": n_any,
                "graded": graded,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\n[report → {out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
