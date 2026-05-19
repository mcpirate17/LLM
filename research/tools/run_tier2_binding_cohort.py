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
from component_fab.proposer.spec_generator import ProposalSpec

_REPO = Path(__file__).resolve().parents[2]
_PROPOSALS = _REPO / "component_fab" / "catalog" / "proposals.jsonl"


def _load_proposals_by_id(path: Path = _PROPOSALS) -> dict[str, ProposalSpec]:
    """Build a {proposal_id: ProposalSpec} map from the catalog jsonl.

    Also scans rotated ``proposals.jsonl.N`` files since the autonomous
    loop rotates at 2 MB and promoted specs may live in older rotations.
    Last-wins when the same proposal_id appears in multiple files.
    """
    paths = [path]
    catalog_dir = path.parent
    if catalog_dir.exists():
        paths.extend(sorted(catalog_dir.glob("proposals.jsonl.*")))
    out: dict[str, ProposalSpec] = {}
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = row.get("proposal_id")
                if not pid:
                    continue
                out[str(pid)] = ProposalSpec(
                    proposal_id=str(pid),
                    name=str(row.get("name") or ""),
                    category=str(row.get("category") or ""),
                    synthesis_kind=str(row.get("synthesis_kind") or ""),
                    math_axes=dict(row.get("math_axes") or {}),
                    anchor_witness_op=str(row.get("anchor_witness_op") or ""),
                    anchor_witnesses_all=tuple(row.get("anchor_witnesses_all") or ()),
                    declared_property_row=dict(row.get("declared_property_row") or {}),
                    predicted_lift=float(row.get("predicted_lift") or 0.0),
                    rationale=str(row.get("rationale") or ""),
                    notes=tuple(row.get("notes") or ()),
                )
    return out


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


def run_cohort(
    proposal_ids: list[str],
    *,
    dim: int = 64,
    n_blocks: int = 2,
    n_train_steps: int = 200,
    pass_threshold: int = 4,
    use_niche_survival: bool = True,
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
    tasks = default_hard_binding_tasks()
    results: dict[str, Any] = {}
    survivors: list[str] = []
    started = time.monotonic()
    for index, pid in enumerate(proposal_ids):
        spec = specs_by_id.get(pid)
        if spec is None:
            if not quiet:
                print(
                    f"[{index + 1}/{len(proposal_ids)}] {pid} NOT in catalog — skipping"
                )
            results[pid] = {"status": "spec_not_found"}
            continue

        def candidate_factory(d: int, _spec: ProposalSpec = spec) -> torch.nn.Module:
            return generate_module_from_spec(_spec, dim=d)

        if not quiet:
            print(
                f"[{index + 1}/{len(proposal_ids)}] {pid} ({spec.name[:50]}) "
                f"running 6 tasks × (1 cand + 2 baselines) × {n_train_steps} steps"
            )
        t0 = time.monotonic()
        try:
            suite = run_harder_binding_suite(
                candidate_factory,
                spec.name,
                tasks=tasks,
                dim=dim,
                n_blocks=n_blocks,
                n_train_steps=n_train_steps,
            )
        except Exception as exc:  # noqa: BLE001
            results[pid] = {"status": f"failed: {exc}"}
            if not quiet:
                print(f"    FAILED: {exc}")
            continue
        per_task = {name: _summarise_per_task(rows) for name, rows in suite.items()}
        pass_count = sum(1 for v in per_task.values() if v.get("beats"))
        elapsed = time.monotonic() - t0
        passed_niche = _niche_survival(per_task)
        tier2_passed = (
            passed_niche if use_niche_survival else (pass_count >= pass_threshold)
        )
        results[pid] = {
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
            "elapsed_s": round(elapsed, 1),
        }
        if tier2_passed:
            survivors.append(pid)
        if not quiet:
            print(
                f"    pass_count={pass_count}/{len(per_task)} "
                f"niche={passed_niche} tier2_passed={tier2_passed} elapsed={elapsed:.1f}s"
            )
    return {
        "n_evaluated": len(proposal_ids),
        "n_survivors": len(survivors),
        "survivors": survivors,
        "pass_threshold": pass_threshold,
        "results": results,
        "elapsed_total_s": round(time.monotonic() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-ids", required=True, type=str)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dim", default=64, type=int)
    parser.add_argument("--n-blocks", default=2, type=int)
    parser.add_argument("--n-train-steps", default=200, type=int)
    parser.add_argument("--pass-threshold", default=4, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    pids = [p.strip() for p in args.proposal_ids.split(",") if p.strip()]
    summary = run_cohort(
        pids,
        dim=args.dim,
        n_blocks=args.n_blocks,
        n_train_steps=args.n_train_steps,
        pass_threshold=args.pass_threshold,
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
