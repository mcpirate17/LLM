"""Per-spec grading + selection pipeline for the autonomous fab loop.

Split out of ``run_autonomous`` (god-file split, behavior-preserving). Covers
the screen → order/budget → tiered-gate grade → niche-annotate → record path
for one cycle's candidate specs. The outer loop and promotion bookkeeping stay
in ``run_autonomous``.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from component_fab.metrics.behavior_fingerprint import (
    FRONTIER_SPECTRA,
    Normalizer,
    is_degenerate,
    operational_spectrum,
    orthogonality_radius,
    spectrum_from_metadata,
)
from component_fab.improver.ranking import (
    composite_score,
    objective_vector,
    pareto_front_indices,
)
from component_fab.proposer.acquisition import select_by_acquisition
from component_fab.proposer.spec_generator import ProposalSpec
from component_fab.proposer.nas_screen import (
    NasScreenResult,
    nas_score_multiplier,
    score_specs_with_nas,
)
from component_fab.proposer.quality import (
    allocate_budget_buckets,
    bucket_counts,
    physics_s05_failure_count_for_spec,
    score_specs_quality,
)
from component_fab.proposer.tier2_feedback import (
    Tier2Feedback,
    tier2_score_multiplier,
)
from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.harness.capability_probes import causality_stability_gate
from component_fab.harness.standard_block import LaneTestBlock
from component_fab.validator.trust import axes_counts_for_specs
from component_fab.state.ledger import (
    Ledger,
    PROMOTION_PROMOTED,
    PROMOTION_REJECTED,
)
from component_fab.state.gates import GATE_S05_CAUSALITY_STABILITY
from component_fab.state.surrogate import MeanFieldApproximant
from component_fab.validator.grade import eliminated_solo_scorecard, grade_candidate
from component_fab.validator.paired import paired_metadata_for_spec
from component_fab.validator.solo import SoloScorecard


def _grade_spec(
    spec: ProposalSpec,
    *,
    dim: int,
    seq_len: int,
    probe_steps: int,
    skip_probe: bool,
    run_range_probe: bool = False,
    range_train_steps: int = 300,
) -> tuple[SoloScorecard, dict | None, dict, str | None, Any | None]:
    """Return ``(solo, probe, capability, eliminated_by, mechanism)``.

    Tiered capability gates (S0.5 → ERF → NB → AR) run first as the
    cheapest filter. If any gate eliminates the proposal, solo + probe
    skip and the caller marks it rejected immediately with the gate
    name recorded. The chain itself lives in ``validator.grade``.
    """
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
        return solo, None, bundle.capability, bundle.eliminated_by, None
    assert bundle.solo is not None  # run_solo=True and not eliminated
    probe_dict = asdict(bundle.in_context) if bundle.in_context is not None else None
    return bundle.solo, probe_dict, bundle.capability, None, bundle.observables


def _annotate_niche_metadata(survivors: list[dict], ledger: Ledger) -> None:
    """Attach WS-5 behavior fingerprint/novelty + WS-4 Pareto-front membership.

    Computed over the whole survivor set (Pareto + novelty are relative). Novelty
    is measured against the existing ledger catalog; the front is computed within
    this cycle's survivors. param_count is not threaded yet, so the efficiency
    objective is currently constant (neutral) — a documented follow-up.
    """
    # Build the catalog from the in-memory rollup instead of re-parsing the
    # ledger file from disk every cycle (the rollup also includes rotated
    # history the active file lacks).
    # Anchor the catalog against BOTH prior ledger spectra (clone detection) and
    # the frontier baselines (softmax/gpt2/mamba2) — a candidate must be far from
    # both to earn orthogonality credit, so a softmax-twin is penalized, not just
    # an intra-population clone.
    catalog = [
        spectrum_from_metadata(entry.metadata_history[-1])
        for entry in ledger.all_entries()
        if entry.metadata_history
    ]
    catalog.extend(FRONTIER_SPECTRA.values())
    spectra = [operational_spectrum(s["probe"], s["capability"]) for s in survivors]
    normalizer = Normalizer.fit(catalog + spectra)
    for surv, fp in zip(survivors, spectra):
        radius = orthogonality_radius(fp, catalog, normalizer=normalizer)
        finite = radius != float("inf")
        surv["metadata"]["operational_spectrum"] = fp
        surv["metadata"]["orthogonality_radius"] = radius if finite else -1.0
        surv["metadata"]["state_degenerate"] = (
            bool(is_degenerate(radius)) if finite else False
        )
    vectors = [
        objective_vector(
            s["probe"],
            s["capability"],
            orthogonality=max(0.0, s["metadata"]["orthogonality_radius"]),
        )
        for s in survivors
    ]
    front = set(pareto_front_indices(vectors))
    for index, (surv, vector) in enumerate(zip(survivors, vectors)):
        surv["metadata"]["pareto_objective_vector"] = dict(vector)
        surv["metadata"]["on_pareto_front"] = index in front


def _finalize_survivors(
    survivors: list[dict], ledger: Ledger, *, cycle: int, niche_promotion: bool
) -> None:
    """Record deferred survivor grades, attaching niche metadata when enabled."""
    if niche_promotion and survivors:
        _annotate_niche_metadata(survivors, ledger)
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


def _physics_probe_metadata(spec: ProposalSpec, probe: dict | None) -> dict:
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


def _metadata_for_grade(
    spec: ProposalSpec,
    capability: dict | None,
    eliminated_by: str | None,
    probe: dict | None = None,
    mechanism: Any | None = None,
) -> dict:
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
        # Persist the full build recipe so promoted specs stay re-gradeable
        # from the ledger (generate_module is a pure function of math_axes).
        "math_axes": dict(spec.math_axes),
        # Range signal (only populated when --range-probe is on); feeds the
        # optional veto_range_blind promotion rule.
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
    return meta


def _record_eliminated(
    ledger: Ledger,
    spec: ProposalSpec,
    solo: SoloScorecard,
    metadata: dict,
    *,
    cycle: int,
) -> None:
    """Record a gate-eliminated spec: zero-score grade + immediate rejection."""
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


def _physics_s05_prescreen_specs(
    specs: list[ProposalSpec],
    ledger: Ledger,
    *,
    cycle: int,
    dim: int,
    seq_len: int,
) -> tuple[list[ProposalSpec], int]:
    safe: list[ProposalSpec] = []
    failed = 0
    for spec in specs:
        if spec.math_axes.get("op_search_track") != "physics_atom":
            safe.append(spec)
            continue
        try:
            module = generate_module_from_spec(spec, dim=dim)
            s05 = causality_stability_gate(
                LaneTestBlock(module, dim).eval(),
                seq_len=seq_len,
                dim=dim,
            )
        except Exception:  # noqa: BLE001 - let full grading handle non-screenable specs
            safe.append(spec)
            continue
        if s05.passed:
            safe.append(spec)
            continue
        failed += 1
        ledger.record_grade(
            proposal_id=spec.proposal_id,
            name=spec.name,
            category=spec.category,
            synthesis_kind=spec.synthesis_kind,
            cycle=cycle,
            composite_score=0.0,
            smoke_pass=False,
            learned_signal=False,
            metadata={
                "math_axes": dict(spec.math_axes),
                "eliminated_by": GATE_S05_CAUSALITY_STABILITY,
                "capability_eliminated_by": GATE_S05_CAUSALITY_STABILITY,
                "physics_s05_prescreen_failed": True,
                "s05_max_first_half_drift": float(s05.max_first_half_drift),
                "notes": list(s05.notes),
            },
        )
        ledger.record_promotion(spec.proposal_id, PROMOTION_REJECTED)
    return safe, failed


def _grade_active_specs(
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
    cycle_scorecards: list[dict] = []
    cycle_probes: dict[str, dict] = {}
    cycle_capabilities: dict[str, dict] = {}
    eliminated_by_gate: dict[str, int] = {}
    survivors: list[dict] = []
    for spec in active_specs:
        solo, probe, capability, eliminated_by, mechanism = _grade_spec(
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
        metadata = _metadata_for_grade(
            spec, capability, eliminated_by, probe, mechanism
        )
        if eliminated_by is not None:
            _record_eliminated(ledger, spec, solo, metadata, cycle=cycle)
            eliminated_by_gate[eliminated_by] = (
                eliminated_by_gate.get(eliminated_by, 0) + 1
            )
            continue
        # WS-2: paired delta vs anchor for survivors only (the promotable set).
        # Eliminated specs are already rejected, so skip the extra training cost.
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
        # Defer recording: WS-4/WS-5 niche metadata (Pareto front + novelty) needs
        # the whole survivor set together, so finalize after the grade loop.
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
    _finalize_survivors(survivors, ledger, cycle=cycle, niche_promotion=niche_promotion)
    return cycle_scorecards, cycle_probes, cycle_capabilities, eliminated_by_gate


def _order_active_specs_by_quality(
    active_specs: list[ProposalSpec],
    ledger: Ledger,
    *,
    tier2_feedback_by_id: dict[str, Tier2Feedback],
    nas_screen_by_id: dict[str, NasScreenResult],
    max_graded_per_cycle: int = 0,
) -> tuple[list[ProposalSpec], dict[str, int]]:
    """Rank active specs by fused quality, applying the budget split.

    Additive ordering layer: candidates are graded in descending quality so the
    best are reached first under a wall-clock budget. Coverage is only capped
    when ``max_graded_per_cycle`` > 0 (then the 60/25/15 exploit/repair/explore
    split decides which specs are graded this cycle). Returns the ordered specs
    and the bucket histogram for reporting.
    """

    if not active_specs:
        return active_specs, bucket_counts(())
    quality_by_id = score_specs_quality(
        active_specs,
        tier2_by_id=tier2_feedback_by_id,
        nas_by_id=nas_screen_by_id,
        entries_by_id=ledger.entries,
        axes_counts=axes_counts_for_specs(active_specs),
    )
    scores = list(quality_by_id.values())
    if max_graded_per_cycle > 0:
        chosen = allocate_budget_buckets(scores, total=max_graded_per_cycle)
    else:
        chosen = sorted(scores, key=lambda s: s.quality_score, reverse=True)
    chosen_ids = [s.proposal_id for s in chosen]
    spec_by_id = {s.proposal_id: s for s in active_specs}
    ordered = [spec_by_id[pid] for pid in chosen_ids if pid in spec_by_id]
    return ordered, bucket_counts(chosen)


def _top_orthogonality_pending(
    pool: list[ProposalSpec], ledger: Ledger, k: int
) -> list[ProposalSpec]:
    """Top-``k`` pending Pareto-front specs by PEAK orthogonality across history.

    The orthogonality radius decays to ~0 once a spec enters the ledger catalog
    (it is min-distance to a catalog that now contains itself), so first-sighting
    novelty lives in the MAX over a spec's grade history, not its latest value.
    Composite ordering starves these genuinely-novel candidates (mid composite vs
    high-composite recombinations), so they never re-grade and never accumulate the
    paired-CI a niche promotion needs — the gate-abandons-novel pathology. Forcing
    them back into the grading budget is the fix.
    """
    scored: list[tuple[float, ProposalSpec]] = []
    for spec in pool:
        entry = ledger.entries.get(spec.proposal_id)
        if entry is None or not entry.metadata_history:
            continue
        if not entry.metadata_history[-1].get("on_pareto_front"):
            continue
        peak = max(
            (
                float(m.get("orthogonality_radius") or 0.0)
                for m in entry.metadata_history
            ),
            default=0.0,
        )
        if peak > 0.0:
            scored.append((peak, spec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [spec for _, spec in scored[:k]]


def _inject_novelty_regrades(
    active_specs: list[ProposalSpec],
    pool: list[ProposalSpec],
    ledger: Ledger,
    *,
    k: int,
    budget: int,
) -> list[ProposalSpec]:
    """Prepend the top-``k`` orthogonality front-members to the graded set (kept
    within ``budget``). Opt-in: ``k <= 0`` returns ``active_specs`` unchanged."""
    if k <= 0:
        return active_specs
    selected = {s.proposal_id for s in active_specs}
    extra = [
        s
        for s in _top_orthogonality_pending(pool, ledger, k)
        if s.proposal_id not in selected
    ]
    if not extra:
        return active_specs
    merged = extra + active_specs
    return merged[:budget] if budget > 0 else merged


def _order_grading_queue(
    active_specs: list[ProposalSpec],
    ledger: Ledger,
    *,
    selection: str,
    acquisition_beta: float,
    use_quality_order: bool,
    max_graded_per_cycle: int,
    tier2_feedback_by_id: dict[str, Tier2Feedback],
    nas_screen_by_id: dict[str, NasScreenResult],
) -> tuple[list[ProposalSpec], dict[str, int]]:
    """Order/budget the screened pool — surrogate-UCB or legacy quality split.

    ``surrogate`` fills the budget with the MeanFieldApproximant's highest-UCB
    candidates (falls back to identity order if it can't fit); otherwise the
    legacy 60/25/15 quality order applies. Returns ``(ordered, bucket_summary)``.
    """
    if selection == "surrogate":
        ordered = select_by_acquisition(
            active_specs,
            MeanFieldApproximant.fit(),
            budget=max_graded_per_cycle,
            beta=acquisition_beta,
        )
        return ordered, bucket_counts(())
    if use_quality_order:
        return _order_active_specs_by_quality(
            active_specs,
            ledger,
            tier2_feedback_by_id=tier2_feedback_by_id,
            nas_screen_by_id=nas_screen_by_id,
            max_graded_per_cycle=max_graded_per_cycle,
        )
    return active_specs, bucket_counts(())


def _select_active_specs(
    specs: list[ProposalSpec],
    ledger: Ledger,
    *,
    cycle: int,
    dim: int,
    seq_len: int,
    selection: str,
    acquisition_beta: float,
    use_nas_screen: bool,
    use_quality_order: bool,
    max_graded_per_cycle: int,
    tier2_feedback_by_id: dict[str, Tier2Feedback],
    regrade_top_orthogonality: int = 0,
) -> tuple[
    list[ProposalSpec],
    dict[str, NasScreenResult],
    dict[str, int],
    int,
    int,
    int,
    int,
    int,
]:
    """Filter terminal specs, screen, and order/budget the grading queue.

    Returns ``(active_specs, nas_screen_by_id, bucket_summary,
    n_new_selected, n_new_available, n_terminal_skipped,
    n_physics_s05_skipped, n_physics_s05_prescreen_failed)``.
    """
    # Re-grade every spec each cycle so the ledger accumulates score history;
    # promotion requires a streak across cycles to fire. Skip only proposals
    # that have already reached a terminal status (promoted or rejected).
    # Filter BEFORE the NAS/measured screen: the screen builds the real
    # module per spec (2 seeds + Jacobian probes), so screening terminal
    # specs that are then discarded was the dominant wasted per-cycle cost.
    skippable = {
        pid
        for pid, entry in ledger.entries.items()
        if entry.promotion_status in (PROMOTION_PROMOTED, PROMOTION_REJECTED)
    }
    active_specs = [s for s in specs if s.proposal_id not in skippable]
    n_terminal_skipped = len(specs) - len(active_specs)
    physics_safe_specs: list[ProposalSpec] = []
    n_physics_s05_skipped = 0
    for spec in active_specs:
        if physics_s05_failure_count_for_spec(spec, ledger.entries) > 0:
            n_physics_s05_skipped += 1
            continue
        physics_safe_specs.append(spec)
    active_specs = physics_safe_specs
    active_specs, n_physics_s05_prescreen_failed = _physics_s05_prescreen_specs(
        active_specs,
        ledger,
        cycle=cycle,
        dim=dim,
        seq_len=seq_len,
    )
    n_new_available = sum(1 for s in active_specs if not ledger.has_seen(s.proposal_id))
    nas_screen_by_id = score_specs_with_nas(active_specs, enabled=use_nas_screen)
    pool = list(active_specs)  # full screened set, before ordering/budget reduces it

    active_specs, bucket_summary = _order_grading_queue(
        active_specs,
        ledger,
        selection=selection,
        acquisition_beta=acquisition_beta,
        use_quality_order=use_quality_order,
        max_graded_per_cycle=max_graded_per_cycle,
        tier2_feedback_by_id=tier2_feedback_by_id,
        nas_screen_by_id=nas_screen_by_id,
    )
    # Force the top-orthogonality pending front-members into the graded budget so
    # genuinely-novel candidates re-grade and can accumulate paired-CI (fixes the
    # composite-selection-starves-novel pathology). Opt-in: 0 => unchanged.
    active_specs = _inject_novelty_regrades(
        active_specs,
        pool,
        ledger,
        k=regrade_top_orthogonality,
        budget=max_graded_per_cycle,
    )
    return (
        active_specs,
        nas_screen_by_id,
        bucket_summary,
        sum(1 for s in active_specs if not ledger.has_seen(s.proposal_id)),
        n_new_available,
        n_terminal_skipped,
        n_physics_s05_skipped,
        n_physics_s05_prescreen_failed,
    )
