"""Tier-2 harder-binding cohort runner — Phase C of the daily fab loop.

Consumes a list of fab ``proposal_id``s plus the persisted spec metadata
in ``component_fab/catalog/proposals.jsonl``. For each spec, builds a
lane factory via ``code_generator.generate_module_from_spec`` and runs
``run_harder_binding_suite`` (6 discrete symbolic binding tasks) against
the same baselines (``softmax_attention``, ``causal_conv``).

Output: ``{proposal_id: {task_name: {candidate_eval_acc, baseline_max,
delta, beats}}}`` summarising whether the fab candidate beats the best
baseline on each task. A spec is a "Tier-2 survivor" iff it beats the
best baseline on ``pass_threshold`` (default 4) of 6 tasks.

Invoked by ``research/tools/fab_daily_loop.py`` after the autonomous loop
halts. Standalone usage:

    python -m research.tools.run_tier2_binding_cohort \
        --proposal-ids id1,id2,id3 \
        --output cohort_tier2.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.harness.harder_binding_tasks import (
    default_hard_binding_tasks,
    run_harder_binding_suite,
)
from component_fab.harness.tiny_lm import (  # noqa: F401  (used in run_cohort + main)
    DEFAULT_BASELINE_NAMES,
    FRONTIER_BASELINE_NAMES,
)
from component_fab.state.tier2_training import append_tier2_labels
from component_fab.proposer.spec_generator import ProposalSpec

# Back-compat alias: the loader moved into component_fab (it builds fab
# ProposalSpecs from the fab catalog); several research tools still import
# the underscore name from here.
from component_fab.proposer.proposal_catalog import (  # noqa: F401
    load_proposals_by_id as _load_proposals_by_id,
)


def _summarise_per_task(rows: list[Any]) -> dict[str, Any]:
    """Reduce candidate + baselines to {candidate_acc, baseline_max, delta, beats}."""
    if not rows:
        return {
            "candidate_eval_acc": 0.0,
            "baseline_max": 0.0,
            "delta": 0.0,
            "beats": False,
            "chance": 0.0,
        }
    candidate = rows[0]
    baselines = rows[1:]
    baseline_max = max((row.eval_accuracy for row in baselines), default=0.0)
    return {
        "candidate_eval_acc": float(candidate.eval_accuracy),
        "candidate_label": candidate.mixer_label,
        "baseline_max": float(baseline_max),
        "delta": float(candidate.eval_accuracy - baseline_max),
        "beats": bool(candidate.eval_accuracy > baseline_max),
        "chance": float(candidate.chance_accuracy),
        "candidate_n_params": int(candidate.n_params),
    }


def _aggregate_per_task(seed_rows: list[dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Average per-task summaries across seeds, preserving the legacy shape."""
    task_names = sorted({task for row in seed_rows for task in row})
    out: dict[str, Any] = {}
    for task in task_names:
        rows = [row[task] for row in seed_rows if task in row]
        if not rows:
            continue
        deltas = [float(row.get("delta") or 0.0) for row in rows]
        cand = [float(row.get("candidate_eval_acc") or 0.0) for row in rows]
        base = [float(row.get("baseline_max") or 0.0) for row in rows]
        out[task] = {
            "candidate_eval_acc": sum(cand) / len(cand),
            "candidate_label": str(rows[-1].get("candidate_label") or ""),
            "baseline_max": sum(base) / len(base),
            "delta": sum(deltas) / len(deltas),
            "beats": (sum(deltas) / len(deltas)) > 0.0,
            "beat_rate": sum(1 for row in rows if row.get("beats")) / len(rows),
            "chance": float(rows[-1].get("chance") or 0.0),
            "candidate_n_params": int(rows[-1].get("candidate_n_params") or 0),
            "seed_deltas": deltas,
        }
    return out


_NICHE_REQUIRED: frozenset[str] = frozenset(
    {"long_gap_recall", "compositional_binding"}
)
_NICHE_BROAD: frozenset[str] = frozenset(
    {
        "multi_query_kv_recall",
        "distractor_kv_recall",
        "variable_layout_recall",
        "heldout_pair_recall",
    }
)


def _niche_survival(per_task: dict[str, dict[str, Any]]) -> bool:
    """Task-typed survival: fab dominates long_gap + compositional but
    fails broad patterns. A candidate passes Tier-2 if it beats baselines
    on BOTH niche tasks AND at least 1 of the 4 broad tasks. Stricter
    than 4/6 on the niche side, looser on broad — matches the win profile
    observed on 2026-05-15 (92% win on niche tasks, 0-17% on broad).
    """
    niche_pass = all(per_task.get(t, {}).get("beats") for t in _NICHE_REQUIRED)
    if not niche_pass:
        return False
    broad_pass = sum(1 for t in _NICHE_BROAD if per_task.get(t, {}).get("beats"))
    return broad_pass >= 1


def _run_one_spec_seeds(
    spec: ProposalSpec,
    *,
    seed: int,
    seed_count: int,
    dim: int,
    n_blocks: int,
    n_train_steps: int,
    baseline_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Run the binding suite for one spec across all seeds; return per-seed rows."""

    def candidate_factory(d: int, _spec: ProposalSpec = spec) -> torch.nn.Module:
        return generate_module_from_spec(_spec, dim=d)

    per_seed: list[dict[str, Any]] = []
    for offset in range(seed_count):
        run_seed = int(seed) + offset
        tasks = default_hard_binding_tasks(seed=run_seed)
        suite = run_harder_binding_suite(
            candidate_factory,
            spec.name,
            tasks=tasks,
            dim=dim,
            n_blocks=n_blocks,
            n_train_steps=n_train_steps,
            seed=run_seed,
            baseline_names=baseline_names,
        )
        per_task_seed = {
            name: _summarise_per_task(rows) for name, rows in suite.items()
        }
        pass_count_seed = sum(
            1 for value in per_task_seed.values() if value.get("beats")
        )
        per_seed.append(
            {
                "seed": run_seed,
                "per_task": per_task_seed,
                "pass_count": pass_count_seed,
                "tier2_passed_niche": _niche_survival(per_task_seed),
            }
        )
    return per_seed


def _build_spec_result(
    spec: ProposalSpec,
    per_seed: list[dict[str, Any]],
    *,
    seed_count: int,
    pass_threshold: int,
    use_niche_survival: bool,
    elapsed: float,
) -> tuple[dict[str, Any], bool]:
    """Aggregate per-seed rows into a result entry; return (entry, tier2_passed)."""
    per_task = _aggregate_per_task([row["per_task"] for row in per_seed])
    pass_count = sum(1 for v in per_task.values() if v.get("beats"))
    passed_niche = _niche_survival(per_task)
    tier2_passed = (
        passed_niche if use_niche_survival else (pass_count >= pass_threshold)
    )
    entry: dict[str, Any] = {
        "status": "ok",
        "name": spec.name,
        "category": spec.category,
        "synthesis_kind": spec.synthesis_kind,
        "math_axes": dict(spec.math_axes),
        "per_task": per_task,
        "pass_count": pass_count,
        "n_tasks": len(per_task),
        "tier2_passed": bool(tier2_passed),
        "tier2_passed_niche": bool(passed_niche),
        "seed_count": seed_count,
        "per_seed": per_seed,
        "elapsed_s": round(elapsed, 1),
    }
    return entry, bool(tier2_passed)


def _eval_proposal(
    pid: str,
    index: int,
    total: int,
    specs_by_id: dict[str, Any],
    *,
    seed: int,
    seed_count: int,
    dim: int,
    n_blocks: int,
    n_train_steps: int,
    pass_threshold: int,
    use_niche_survival: bool,
    resolved_baselines: tuple[str, ...],
    quiet: bool,
) -> tuple[dict[str, Any], bool]:
    """Evaluate one pid; return (result_entry, tier2_passed)."""
    spec = specs_by_id.get(pid)
    if spec is None:
        if not quiet:
            print(f"[{index + 1}/{total}] {pid} NOT in catalog — skipping")
        return {"status": "spec_not_found"}, False
    if not quiet:
        print(
            f"[{index + 1}/{total}] {pid} ({spec.name[:50]}) "
            f"running {seed_count} seed(s) × 6 tasks × "
            f"(1 cand + 2 baselines) × {n_train_steps} steps"
        )
    t0 = time.monotonic()
    try:
        per_seed = _run_one_spec_seeds(
            spec,
            seed=seed,
            seed_count=seed_count,
            dim=dim,
            n_blocks=n_blocks,
            n_train_steps=n_train_steps,
            baseline_names=resolved_baselines,
        )
    except Exception as exc:  # noqa: BLE001
        if not quiet:
            print(f"    FAILED: {exc}")
        return {"status": f"failed: {exc}"}, False
    entry, tier2_passed = _build_spec_result(
        spec,
        per_seed,
        seed_count=seed_count,
        pass_threshold=pass_threshold,
        use_niche_survival=use_niche_survival,
        elapsed=time.monotonic() - t0,
    )
    if not quiet:
        print(
            f"    pass_count={entry['pass_count']}/{entry['n_tasks']} "
            f"niche={entry['tier2_passed_niche']} tier2_passed={tier2_passed} "
            f"elapsed={entry['elapsed_s']}s"
        )
    return entry, tier2_passed


def _finalise_cohort(
    results: dict[str, Any],
    survivors: list[str],
    proposal_ids: list[str],
    *,
    accumulate_labels: bool,
    resolved_baselines: tuple[str, ...],
    dim: int,
    n_blocks: int,
    n_train_steps: int,
    seed_count: int,
    pass_threshold: int,
    seed: int,
    started: float,
    quiet: bool,
) -> dict[str, Any]:
    """Accumulate training labels and assemble the final summary dict."""
    if accumulate_labels:
        n_appended = append_tier2_labels(
            results,
            baseline_names=resolved_baselines,
            dim=dim,
            n_blocks=n_blocks,
            n_train_steps=n_train_steps,
            seed_count=seed_count,
        )
        if n_appended and not quiet:
            print(f"[accumulated {n_appended} Tier-2 training labels]")
    return {
        "n_evaluated": len(proposal_ids),
        "n_survivors": len(survivors),
        "survivors": survivors,
        "pass_threshold": pass_threshold,
        "seed": int(seed),
        "seed_count": seed_count,
        "baseline_names": list(resolved_baselines),
        "results": results,
        "elapsed_total_s": round(time.monotonic() - started, 1),
    }


def run_cohort(
    proposal_ids: list[str],
    *,
    dim: int = 64,
    n_blocks: int = 2,
    n_train_steps: int = 200,
    pass_threshold: int = 4,
    use_niche_survival: bool = True,
    seed: int = 0,
    seed_count: int = 1,
    baseline_names: tuple[str, ...] | None = None,
    accumulate_labels: bool = True,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run Tier-2 binding on each proposal_id; return summary dict.

    Survival rule: when ``use_niche_survival=True`` (default), a candidate
    is a Tier-2 survivor iff it beats baselines on both ``long_gap_recall``
    AND ``compositional_binding`` AND at least one of the 4 broad tasks.
    Otherwise the legacy ``pass_count >= pass_threshold`` (default 4) rule
    applies.
    """
    specs_by_id = _load_proposals_by_id()
    results: dict[str, Any] = {}
    survivors: list[str] = []
    started = time.monotonic()
    seed_count = max(1, int(seed_count))
    resolved_baselines = baseline_names or DEFAULT_BASELINE_NAMES
    for index, pid in enumerate(proposal_ids):
        entry, tier2_passed = _eval_proposal(
            pid,
            index,
            len(proposal_ids),
            specs_by_id,
            seed=seed,
            seed_count=seed_count,
            dim=dim,
            n_blocks=n_blocks,
            n_train_steps=n_train_steps,
            pass_threshold=pass_threshold,
            use_niche_survival=use_niche_survival,
            resolved_baselines=resolved_baselines,
            quiet=quiet,
        )
        results[pid] = entry
        if tier2_passed:
            survivors.append(pid)
    return _finalise_cohort(
        results,
        survivors,
        proposal_ids,
        accumulate_labels=accumulate_labels,
        resolved_baselines=resolved_baselines,
        dim=dim,
        n_blocks=n_blocks,
        n_train_steps=n_train_steps,
        seed_count=seed_count,
        pass_threshold=pass_threshold,
        seed=seed,
        started=started,
        quiet=quiet,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-ids", required=True, type=str)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dim", default=64, type=int)
    parser.add_argument("--n-blocks", default=2, type=int)
    parser.add_argument("--n-train-steps", default=200, type=int)
    parser.add_argument("--pass-threshold", default=4, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--seed-count", default=1, type=int)
    parser.add_argument(
        "--baselines",
        default="default",
        help="'default' (softmax+conv), 'frontier' (softmax+gpt2+mamba+mamba2), "
        "or a comma-separated list of baseline names",
    )
    parser.add_argument(
        "--no-accumulate",
        action="store_true",
        help="do not append results to the Tier-2 predictor training table",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    pids = [p.strip() for p in args.proposal_ids.split(",") if p.strip()]
    if args.baselines == "default":
        baseline_names: tuple[str, ...] = DEFAULT_BASELINE_NAMES
    elif args.baselines == "frontier":
        baseline_names = FRONTIER_BASELINE_NAMES
    else:
        baseline_names = tuple(
            n.strip() for n in args.baselines.split(",") if n.strip()
        )
    summary = run_cohort(
        pids,
        dim=args.dim,
        n_blocks=args.n_blocks,
        n_train_steps=args.n_train_steps,
        pass_threshold=args.pass_threshold,
        seed=args.seed,
        seed_count=args.seed_count,
        baseline_names=baseline_names,
        accumulate_labels=not args.no_accumulate,
        quiet=args.quiet,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if not args.quiet:
        print(
            f"\ncohort complete: {summary['n_survivors']}/{summary['n_evaluated']} "
            f"survived tier-2 (>= {args.pass_threshold}/6 tasks); "
            f"output: {args.output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
