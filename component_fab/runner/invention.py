"""Invention-track grading and ledger helpers."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from component_fab.harness.harder_binding_tasks import (
    default_hard_binding_tasks,
    run_harder_binding_suite,
)
from component_fab.harness.tiny_lm import DEFAULT_BASELINE_NAMES
from component_fab.policies.promotion import (
    PromotionRules,
    apply_decisions,
    decide_promotions_for_ledger,
)
from component_fab.proposer.spec_generator import ProposalSpec, spec_to_json
from component_fab.state.ledger import Ledger, PROMOTION_REJECTED
from component_fab.validator.grade import factory_from_spec, grade_candidate


def lm_binding_summary(
    spec: ProposalSpec,
    *,
    task_limit: int,
    steps: int,
    batch_size: int,
    dim: int,
    n_blocks: int,
    seed: int,
) -> dict[str, Any]:
    """Run the optional hard-binding TinyLM comparison for one invention spec."""

    tasks = default_hard_binding_tasks(seed=seed)[: max(0, task_limit)]
    results = run_harder_binding_suite(
        factory_from_spec(spec),
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


def grade_invention(
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
    """Grade one invention candidate and return a JSON-ready result row."""

    bundle = grade_candidate(
        spec,
        dim=dim,
        seq_len=seq_len,
        n_steps=probe_steps,
        run_range_probe=run_range_probe,
        range_train_steps=range_train_steps,
        run_in_context=not skip_in_context,
    )
    if bundle.eliminated_by is not None:
        return {
            "spec": spec_to_json(spec),
            "status": "eliminated",
            "eliminated_by": bundle.eliminated_by,
            "capability": bundle.capability,
            "score": 0.0,
        }

    solo = bundle.solo
    assert solo is not None
    in_context = bundle.in_context
    score = 0.45 if solo.promoted else 0.15
    if in_context is not None:
        if in_context.learned_signal:
            score += 0.25
        score += min(0.15, max(0.0, in_context.aggregate_loss_ratio - 1.0) * 0.05)
    if bundle.capability.get("can_bind"):
        score += 0.15

    binding = None
    if run_lm_binding:
        binding = lm_binding_summary(
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
        "capability": bundle.capability,
        "solo": asdict(solo),
        "in_context": asdict(in_context) if in_context is not None else None,
        "lm_binding": binding,
        "score": round(max(0.0, min(1.0, score)), 4),
    }


def metadata_for_invention_result(result: dict[str, Any]) -> dict[str, Any]:
    """Ledger metadata for one invention result."""

    spec = result["spec"]
    capability = result.get("capability") or {}
    binding = result.get("lm_binding") or {}
    return {
        "track": "invention",
        "mechanism": spec["math_axes"].get("op_invention_mechanism"),
        "math_axes": dict(spec["math_axes"]),
        "eliminated_by": result.get("eliminated_by"),
        "can_bind": bool(capability.get("can_bind")),
        "erf_density": float(capability.get("erf_density") or 0.0),
        "nb_max_accuracy": float(capability.get("nb_max_accuracy") or 0.0),
        "range_effective_distance": int(
            capability.get("range_effective_distance") or 0
        ),
        "range_aggregate_acc": float(capability.get("range_aggregate_acc") or 0.0),
        "lm_binding_candidate_wins": binding.get("candidate_wins"),
        "lm_binding_mean_margin": binding.get("mean_margin"),
    }


def record_invention_result(ledger: Ledger, result: dict[str, Any], cycle: int) -> None:
    """Record one invention result into the invention ledger."""

    spec = result["spec"]
    ledger.record_grade(
        proposal_id=spec["proposal_id"],
        name=spec["name"],
        category=spec["category"],
        synthesis_kind=spec["synthesis_kind"],
        cycle=cycle,
        composite_score=float(result.get("score") or 0.0),
        smoke_pass=bool((result.get("solo") or {}).get("promoted")),
        learned_signal=bool((result.get("in_context") or {}).get("learned_signal")),
        metadata=metadata_for_invention_result(result),
    )
    if result.get("status") == "eliminated":
        ledger.record_promotion(spec["proposal_id"], PROMOTION_REJECTED)


def apply_invention_promotions(
    ledger: Ledger,
    *,
    veto_range_blind: bool,
    min_range_distance: int,
) -> dict[str, int]:
    """Apply the invention-track promotion policy and return status counts."""

    rules = PromotionRules(
        promote_min_streak_cycles=1,
        promote_min_composite=0.7,
        promote_require_learned_signal=False,
        reject_after_n_cycles=2,
        reject_max_composite=0.25,
        veto_range_blind=veto_range_blind,
        min_range_effective_distance=min_range_distance,
    )
    return apply_decisions(ledger, decide_promotions_for_ledger(ledger, rules))
