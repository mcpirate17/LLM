"""CLI: invention-track fab loop.

This loop is intentionally separate from ``run_autonomous`` rehab. It starts
from mechanism blueprints, applies a hard invention gate, reuses the existing
fab validators, and can optionally run the TinyLM hard-binding suite against
standard mixers before recording results in an invention-specific ledger.

Usage:
    python -m component_fab.tools.run_invention --dry-run
    python -m component_fab.tools.run_invention --max-specs 4 --probe-steps 40
    python -m component_fab.tools.run_invention --run-lm-binding --binding-task-limit 2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.harness.harder_binding_tasks import (
    default_hard_binding_tasks,
    run_harder_binding_suite,
)
from component_fab.harness.tiny_lm import DEFAULT_BASELINE_NAMES
from component_fab.inventor.mechanism_catalog import (
    enumerate_invention_specs,
    invention_gate_reasons,
)
from component_fab.policies.promotion import (
    PromotionRules,
    apply_decisions,
    decide_promotions_for_ledger,
)
from component_fab.proposer.spec_generator import ProposalSpec, spec_to_json
from component_fab.state.ledger import (
    PROMOTION_REJECTED,
    DEFAULT_LEDGER_PATH,
    Ledger,
)
from component_fab.validator.capability import (
    capability_scorecard_to_dict,
    validate_capabilities,
)
from component_fab.validator.in_context import validate_in_context
from component_fab.validator.solo import validate_solo

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_INVENTION_LEDGER = DEFAULT_LEDGER_PATH.with_name("invention_ledger.jsonl")
DEFAULT_REPORT = _REPO / "component_fab" / "catalog" / "invention_run_latest.json"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="component_fab invention loop")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-specs", type=int, default=4)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--probe-steps", type=int, default=60)
    parser.add_argument("--skip-in-context", action="store_true")
    parser.add_argument("--run-lm-binding", action="store_true")
    parser.add_argument("--binding-task-limit", type=int, default=2)
    parser.add_argument("--binding-steps", type=int, default=150)
    parser.add_argument("--binding-batch-size", type=int, default=16)
    parser.add_argument("--binding-dim", type=int, default=32)
    parser.add_argument("--binding-blocks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-range-probe",
        dest="run_range_probe",
        action="store_false",
        help="Skip the distance-resolved sparse/long-range binding probe.",
    )
    parser.add_argument(
        "--range-train-steps",
        type=int,
        default=300,
        help="Train steps for the range probe (recurrent lanes need more; "
        "600 reaches full-range binding for memory lanes).",
    )
    parser.add_argument(
        "--veto-range-blind",
        action="store_true",
        help="Block promotion of specs whose MEASURED range_effective_distance is "
        "below --min-range-distance. Use with adequate --range-train-steps (>=600 "
        "for recurrent/scan lanes) or it will falsely veto undertrained binders.",
    )
    parser.add_argument("--min-range-distance", type=int, default=1)
    parser.add_argument("--ledger", default=str(DEFAULT_INVENTION_LEDGER))
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    return parser.parse_args(argv)


def _factory_from_spec(spec: ProposalSpec):
    axes = dict(spec.math_axes)

    def factory(dim: int):
        from component_fab.generator.code_generator import generate_module

        return generate_module(axes, dim=dim)

    return factory


def _lm_binding_summary(
    spec: ProposalSpec,
    *,
    task_limit: int,
    steps: int,
    batch_size: int,
    dim: int,
    n_blocks: int,
    seed: int,
) -> dict[str, Any]:
    tasks = default_hard_binding_tasks(seed=seed)[: max(0, task_limit)]
    results = run_harder_binding_suite(
        _factory_from_spec(spec),
        candidate_label=spec.name,
        tasks=tasks,
        baseline_names=tuple(DEFAULT_BASELINE_NAMES),
        dim=dim,
        n_blocks=n_blocks,
        n_train_steps=steps,
        batch_size=batch_size,
        seed=seed,
    )
    candidate_wins = 0
    margins: list[float] = []
    payload: dict[str, list[dict[str, Any]]] = {}
    for task_name, rows in results.items():
        payload[task_name] = [asdict(row) for row in rows]
        candidate = rows[0]
        baselines = rows[1:]
        best_baseline = max((row.eval_accuracy for row in baselines), default=0.0)
        margin = candidate.eval_accuracy - best_baseline
        margins.append(margin)
        if margin > 0.02:
            candidate_wins += 1
    return {
        "candidate_wins": candidate_wins,
        "mean_margin": sum(margins) / len(margins) if margins else 0.0,
        "results": payload,
    }


def _grade_invention(
    spec: ProposalSpec,
    *,
    dim: int,
    seq_len: int,
    probe_steps: int,
    skip_in_context: bool,
    run_lm_binding: bool,
    binding_task_limit: int,
    binding_steps: int,
    binding_batch_size: int,
    binding_dim: int,
    binding_blocks: int,
    seed: int,
    run_range_probe: bool = True,
    range_train_steps: int = 300,
) -> dict[str, Any]:
    module = generate_module_from_spec(spec, dim=dim)
    capability = validate_capabilities(
        spec,
        module,
        dim=dim,
        seq_len=seq_len,
        run_range_probe=run_range_probe,
        range_train_steps=range_train_steps,
    )
    capability_dict = capability_scorecard_to_dict(capability)
    if capability.eliminated_by is not None:
        return {
            "spec": spec_to_json(spec),
            "status": "eliminated",
            "eliminated_by": capability.eliminated_by,
            "capability": capability_dict,
            "score": 0.0,
        }

    solo = validate_solo(spec, module, dim=dim, seq_len=seq_len)
    in_context = None
    score = 0.45 if solo.promoted else 0.15
    if not skip_in_context and solo.promoted:
        in_context = validate_in_context(
            spec,
            module,
            dim=dim,
            seq_len=seq_len,
            n_steps=probe_steps,
        )
        if in_context.learned_signal:
            score += 0.25
        score += min(0.15, max(0.0, in_context.aggregate_loss_ratio - 1.0) * 0.05)
    if capability.can_bind:
        score += 0.15

    binding = None
    if run_lm_binding:
        binding = _lm_binding_summary(
            spec,
            task_limit=binding_task_limit,
            steps=binding_steps,
            batch_size=binding_batch_size,
            dim=binding_dim,
            n_blocks=binding_blocks,
            seed=seed,
        )
        score += min(0.2, 0.08 * binding["candidate_wins"])
        score += max(-0.1, min(0.1, float(binding["mean_margin"])))

    return {
        "spec": spec_to_json(spec),
        "status": "graded",
        "capability": capability_dict,
        "solo": asdict(solo),
        "in_context": asdict(in_context) if in_context is not None else None,
        "lm_binding": binding,
        "score": round(max(0.0, min(1.0, score)), 4),
    }


def _record_result(ledger: Ledger, result: dict[str, Any], cycle: int) -> None:
    spec = result["spec"]
    capability = result.get("capability") or {}
    metadata = {
        "track": "invention",
        "mechanism": spec["math_axes"].get("op_invention_mechanism"),
        # Persist the full build recipe so promoted specs stay re-gradeable from
        # the ledger (generate_module is a pure function of math_axes; the ledger
        # otherwise drops it and block-template winners become unrebuildable).
        "math_axes": dict(spec["math_axes"]),
        "eliminated_by": result.get("eliminated_by"),
        "can_bind": bool(capability.get("can_bind")),
        "erf_density": float(capability.get("erf_density") or 0.0),
        "nb_max_accuracy": float(capability.get("nb_max_accuracy") or 0.0),
        "range_effective_distance": int(
            capability.get("range_effective_distance") or 0
        ),
        "range_aggregate_acc": float(capability.get("range_aggregate_acc") or 0.0),
        "lm_binding_candidate_wins": (
            (result.get("lm_binding") or {}).get("candidate_wins")
        ),
        "lm_binding_mean_margin": ((result.get("lm_binding") or {}).get("mean_margin")),
    }
    ledger.record_grade(
        proposal_id=spec["proposal_id"],
        name=spec["name"],
        category=spec["category"],
        synthesis_kind=spec["synthesis_kind"],
        cycle=cycle,
        composite_score=float(result.get("score") or 0.0),
        smoke_pass=bool((result.get("solo") or {}).get("promoted")),
        learned_signal=bool((result.get("in_context") or {}).get("learned_signal")),
        metadata=metadata,
    )
    if result.get("status") == "eliminated":
        ledger.record_promotion(spec["proposal_id"], PROMOTION_REJECTED)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    specs = enumerate_invention_specs()[: max(0, args.max_specs)]
    gated = [(spec, invention_gate_reasons(spec)) for spec in specs]
    blocked = [(spec, reasons) for spec, reasons in gated if reasons]
    active = [spec for spec, reasons in gated if not reasons]
    if args.dry_run:
        payload = {
            "active": [spec_to_json(spec) for spec in active],
            "blocked": [
                {"spec": spec_to_json(spec), "reasons": list(reasons)}
                for spec, reasons in blocked
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    ledger = Ledger(args.ledger)
    results = []
    for index, spec in enumerate(active, start=1):
        result = _grade_invention(
            spec,
            dim=args.dim,
            seq_len=args.seq_len,
            probe_steps=args.probe_steps,
            skip_in_context=args.skip_in_context,
            run_lm_binding=args.run_lm_binding,
            binding_task_limit=args.binding_task_limit,
            binding_steps=args.binding_steps,
            binding_batch_size=args.binding_batch_size,
            binding_dim=args.binding_dim,
            binding_blocks=args.binding_blocks,
            seed=args.seed,
            run_range_probe=args.run_range_probe,
            range_train_steps=args.range_train_steps,
        )
        _record_result(ledger, result, cycle=index)
        results.append(result)

    rules = PromotionRules(
        promote_min_streak_cycles=1,
        promote_min_composite=0.7,
        promote_require_learned_signal=False,
        reject_after_n_cycles=2,
        reject_max_composite=0.25,
        veto_range_blind=args.veto_range_blind,
        min_range_effective_distance=args.min_range_distance,
    )
    promotion_counts = apply_decisions(
        ledger, decide_promotions_for_ledger(ledger, rules)
    )
    payload = {
        "track": "invention",
        "n_active": len(active),
        "n_blocked": len(blocked),
        "promotion_counts": promotion_counts,
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote: {out}")
    for result in results:
        spec = result["spec"]
        print(
            f"{spec['name']:<42} {result['status']:<10} "
            f"score={float(result.get('score') or 0.0):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
