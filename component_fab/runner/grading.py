"""Grading and ledger-record assembly for autonomous fab cycles."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from component_fab.improver.ranking import (
    composite_score,
    objective_vector,
    pareto_front_indices,
)
from component_fab.metrics.behavior_fingerprint import (
    Normalizer,
    behavior_fingerprint,
    fingerprint_from_metadata,
    is_clone,
    novelty_distance,
)
from component_fab.proposer.nas_screen import NasScreenResult, nas_score_multiplier
from component_fab.proposer.spec_generator import ProposalSpec
from component_fab.proposer.tier2_feedback import Tier2Feedback, tier2_score_multiplier
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
) -> tuple[SoloScorecard, dict | None, dict, str | None]:
    """Return ``(solo, probe, capability, eliminated_by)`` for one spec."""

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
        return solo, None, bundle.capability, bundle.eliminated_by
    assert bundle.solo is not None
    probe_dict = asdict(bundle.in_context) if bundle.in_context is not None else None
    return bundle.solo, probe_dict, bundle.capability, None


def metadata_for_grade(
    spec: ProposalSpec, capability: dict | None, eliminated_by: str | None
) -> dict[str, Any]:
    """Ledger metadata for one graded spec."""

    math_knobs = str(spec.math_axes.get("op_math_knobs") or "")
    return {
        "math_knobs": [part for part in math_knobs.split("+") if part],
        "eliminated_by": eliminated_by,
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
    }


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


def annotate_niche_metadata(survivors: list[dict[str, Any]], ledger: Ledger) -> None:
    """Attach behavior novelty and Pareto-front membership to survivor rows."""

    catalog = [
        fingerprint_from_metadata(entry.metadata_history[-1])
        for entry in ledger.all_entries()
        if entry.metadata_history
    ]
    fingerprints = [
        behavior_fingerprint(s["probe"], s["capability"]) for s in survivors
    ]
    normalizer = Normalizer.fit(catalog + fingerprints)
    for surv, fp in zip(survivors, fingerprints):
        dist = novelty_distance(fp, catalog, normalizer=normalizer)
        finite = dist != float("inf")
        surv["metadata"]["behavior_fingerprint"] = fp
        surv["metadata"]["novelty_distance"] = dist if finite else -1.0
        surv["metadata"]["behavior_clone"] = bool(is_clone(dist)) if finite else False
    vectors = [
        objective_vector(
            s["probe"],
            s["capability"],
            novelty=max(0.0, s["metadata"]["novelty_distance"]),
        )
        for s in survivors
    ]
    front = set(pareto_front_indices(vectors))
    for index, (surv, vector) in enumerate(zip(survivors, vectors)):
        surv["metadata"]["pareto_objective_vector"] = dict(vector)
        surv["metadata"]["on_pareto_front"] = index in front


def finalize_survivors(
    survivors: list[dict[str, Any]], ledger: Ledger, *, cycle: int, niche_promotion: bool
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
        solo, probe, capability, eliminated_by = grade_spec(
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
        metadata = metadata_for_grade(spec, capability, eliminated_by)
        if eliminated_by is not None:
            record_eliminated(ledger, spec, solo, metadata, cycle=cycle)
            eliminated_by_gate[eliminated_by] = eliminated_by_gate.get(eliminated_by, 0) + 1
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
        score *= tier2_score_multiplier((tier2_feedback_by_id or {}).get(spec.proposal_id))
        score *= nas_score_multiplier((nas_screen_by_id or {}).get(spec.proposal_id))
        survivors.append(
            {
                "proposal_id": spec.proposal_id,
                "name": solo.name,
                "category": solo.category,
                "synthesis_kind": solo.synthesis_kind,
                "composite_score": score,
                "smoke_pass": bool(
                    solo.smoke.get("forward_passed") and solo.smoke.get("backward_passed")
                ),
                "learned_signal": bool(probe and probe.get("learned_signal")),
                "probe": probe,
                "capability": capability,
                "metadata": metadata,
            }
        )
    finalize_survivors(survivors, ledger, cycle=cycle, niche_promotion=niche_promotion)
    return cycle_scorecards, cycle_probes, cycle_capabilities, eliminated_by_gate
