"""Grading and ledger-record assembly for autonomous fab cycles."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from component_fab.improver.ranking import composite_score
from component_fab.proposer.nas_screen import NasScreenResult, nas_score_multiplier
from component_fab.proposer.spec_generator import ProposalSpec
from component_fab.proposer.tier2_feedback import Tier2Feedback, tier2_score_multiplier
from component_fab.runner.niche import annotate_niche_metadata
from component_fab.state.ledger import Ledger, PROMOTION_REJECTED
from component_fab.validator.grade import eliminated_solo_scorecard, grade_candidate
from component_fab.validator.paired import paired_metadata_for_spec
from component_fab.validator.solo import SoloScorecard


def grade_spec(
    spec: ProposalSpec,
    *,
    dim: int,
    seq_len: int,
    probe_steps: int,
    skip_probe: bool,
    run_range_probe: bool = False,
    range_train_steps: int = 300,
) -> tuple[SoloScorecard, dict | None, dict, str | None, Any | None, dict | None]:
    """Return ``(solo, probe, capability, eliminated_by, mechanism, compression)``."""

    bundle = grade_candidate(
        spec,
        dim=dim,
        seq_len=seq_len,
        n_steps=probe_steps,
        run_range_probe=run_range_probe,
        range_train_steps=range_train_steps,
        persist_solo_scorecard=True,
        run_in_context=not skip_probe,
    )
    if bundle.eliminated_by is not None:
        solo = eliminated_solo_scorecard(spec, bundle.eliminated_by)
        return solo, None, bundle.capability, bundle.eliminated_by, None, None
    assert bundle.solo is not None
    probe_dict = asdict(bundle.in_context) if bundle.in_context is not None else None
    return (
        bundle.solo,
        probe_dict,
        bundle.capability,
        None,
        bundle.observables,
        bundle.compression,
    )


def _physics_probe_metadata(spec: ProposalSpec, probe: dict | None) -> dict[str, Any]:
    if spec.math_axes.get("op_search_track") != "physics_atom" or not probe:
        return {}
    per_task = probe.get("per_task") or {}
    ratios = {
        name: round(float(result.get("loss_ratio_initial_over_final") or 0.0), 4)
        for name, result in per_task.items()
        if result.get("trained_successfully")
    }
    return {
        "physics_probe_aggregate_loss_ratio": round(
            float(probe.get("aggregate_loss_ratio") or 0.0), 4
        ),
        "physics_probe_mean_loss_ratio": round(
            float(probe.get("mean_loss_ratio") or 0.0), 4
        ),
        "physics_probe_task_ratios": ratios,
        "physics_probe_notes": list(probe.get("notes") or ()),
    }


def metadata_for_grade(
    spec: ProposalSpec,
    capability: dict | None,
    eliminated_by: str | None,
    probe: dict | None = None,
    mechanism: Any | None = None,
    compression: dict | None = None,
) -> dict[str, Any]:
    """Ledger metadata for one graded spec (pure assembly, no side effects)."""

    math_knobs = str(spec.math_axes.get("op_math_knobs") or "")
    capability_eliminated_by = (capability or {}).get("eliminated_by")
    meta = {
        "math_knobs": [part for part in math_knobs.split("+") if part],
        "eliminated_by": eliminated_by,
        "capability_eliminated_by": capability_eliminated_by,
        "soft_gate_escape": bool(capability_eliminated_by and eliminated_by is None),
        "can_bind": bool(capability and capability.get("can_bind")),
        "erf_density": float(capability.get("erf_density") or 0.0)
        if capability
        else 0.0,
        "nb_max_accuracy": float(capability.get("nb_max_accuracy") or 0.0)
        if capability
        else 0.0,
        "math_axes": dict(spec.math_axes),
        "range_effective_distance": (
            int(capability.get("range_effective_distance") or 0) if capability else 0
        ),
        "range_ran": bool(capability and capability.get("range_ran")),
        **_physics_probe_metadata(spec, probe),
    }
    if mechanism:
        meta.update(
            {
                "routing_entropy_mean": round(float(mechanism.routing_entropy_mean), 4),
                "load_balance_cv": round(float(mechanism.load_balance_cv), 4),
                "state_degeneracy": round(float(mechanism.state_degeneracy), 4),
                "active_lane_fraction": round(float(mechanism.active_lane_fraction), 4),
                "relaxation_slope": round(float(mechanism.relaxation_slope), 4),
                "address_entropy": round(float(mechanism.address_entropy), 4),
                "mechanism_passed": bool(mechanism.passed),
            }
        )
    if compression:
        meta.update(compression)
    return meta


def record_eliminated(
    ledger: Ledger,
    spec: ProposalSpec,
    solo: SoloScorecard,
    metadata: dict[str, Any],
    *,
    cycle: int,
) -> None:
    """Record a gate-eliminated spec as a zero-score grade plus rejection."""

    ledger.record_grade(
        proposal_id=spec.proposal_id,
        name=solo.name,
        category=solo.category,
        synthesis_kind=solo.synthesis_kind,
        cycle=cycle,
        composite_score=0.0,
        smoke_pass=False,
        learned_signal=False,
        metadata=metadata,
    )
    ledger.record_promotion(spec.proposal_id, PROMOTION_REJECTED)


def finalize_survivors(
    survivors: list[dict[str, Any]],
    ledger: Ledger,
    *,
    cycle: int,
    niche_promotion: bool,
) -> None:
    """Record deferred survivor grades, adding niche metadata when enabled."""

    if niche_promotion and survivors:
        annotate_niche_metadata(survivors, ledger)
    for surv in survivors:
        ledger.record_grade(
            proposal_id=surv["proposal_id"],
            name=surv["name"],
            category=surv["category"],
            synthesis_kind=surv["synthesis_kind"],
            cycle=cycle,
            composite_score=surv["composite_score"],
            smoke_pass=surv["smoke_pass"],
            learned_signal=surv["learned_signal"],
            metadata=surv["metadata"],
        )


def grade_active_specs(
    active_specs: list[ProposalSpec],
    ledger: Ledger,
    *,
    cycle: int,
    dim: int,
    seq_len: int,
    probe_steps: int,
    skip_probe: bool,
    run_range_probe: bool = False,
    range_train_steps: int = 300,
    tier2_feedback_by_id: dict[str, Tier2Feedback] | None = None,
    nas_screen_by_id: dict[str, NasScreenResult] | None = None,
    paired_seeds: int = 0,
    niche_promotion: bool = False,
) -> tuple[list[dict], dict[str, dict], dict[str, dict], dict[str, int]]:
    """Grade active specs and persist grade events to the ledger."""

    cycle_scorecards: list[dict] = []
    cycle_probes: dict[str, dict] = {}
    cycle_capabilities: dict[str, dict] = {}
    eliminated_by_gate: dict[str, int] = {}
    survivors: list[dict[str, Any]] = []
    for spec in active_specs:
        solo, probe, capability, eliminated_by, mechanism, compression = grade_spec(
            spec,
            dim=dim,
            seq_len=seq_len,
            probe_steps=probe_steps,
            skip_probe=skip_probe,
            run_range_probe=run_range_probe,
            range_train_steps=range_train_steps,
        )
        cycle_scorecards.append(asdict(solo))
        if probe is not None:
            cycle_probes[spec.proposal_id] = probe
        if capability is not None:
            cycle_capabilities[spec.proposal_id] = capability
        metadata = metadata_for_grade(
            spec, capability, eliminated_by, probe, mechanism, compression
        )
        if eliminated_by is not None:
            record_eliminated(ledger, spec, solo, metadata, cycle=cycle)
            eliminated_by_gate[eliminated_by] = (
                eliminated_by_gate.get(eliminated_by, 0) + 1
            )
            continue
        if paired_seeds > 0:
            metadata.update(
                paired_metadata_for_spec(
                    spec,
                    seeds=tuple(range(paired_seeds)),
                    dim=dim,
                    seq_len=seq_len,
                    n_steps=probe_steps,
                )
            )
        score, _ = composite_score(asdict(solo), probe, capability)
        score *= tier2_score_multiplier(
            (tier2_feedback_by_id or {}).get(spec.proposal_id)
        )
        score *= nas_score_multiplier((nas_screen_by_id or {}).get(spec.proposal_id))
        survivors.append(
            {
                "proposal_id": spec.proposal_id,
                "name": solo.name,
                "category": solo.category,
                "synthesis_kind": solo.synthesis_kind,
                "composite_score": score,
                "smoke_pass": bool(
                    solo.smoke.get("forward_passed")
                    and solo.smoke.get("backward_passed")
                ),
                "learned_signal": bool(probe and probe.get("learned_signal")),
                "probe": probe,
                "capability": capability,
                "metadata": metadata,
            }
        )
    finalize_survivors(survivors, ledger, cycle=cycle, niche_promotion=niche_promotion)
    return cycle_scorecards, cycle_probes, cycle_capabilities, eliminated_by_gate
