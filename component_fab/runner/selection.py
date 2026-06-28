"""Candidate filtering, screening, and ordering for autonomous fab cycles."""

from __future__ import annotations

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.harness.capability_probes import causality_stability_gate
from component_fab.harness.standard_block import LaneTestBlock
from component_fab.proposer.acquisition import select_by_acquisition
from component_fab.proposer.nas_screen import NasScreenResult, score_specs_with_nas
from component_fab.proposer.quality import (
    allocate_budget_buckets,
    bucket_counts,
    physics_s05_failure_count_for_spec,
    score_specs_quality,
)
from component_fab.proposer.spec_generator import ProposalSpec
from component_fab.proposer.tier2_feedback import Tier2Feedback
from component_fab.state.gates import GATE_S05_CAUSALITY_STABILITY
from component_fab.state.ledger import Ledger, PROMOTION_PROMOTED, PROMOTION_REJECTED
from component_fab.state.surrogate import MeanFieldApproximant
from component_fab.validator.trust import axes_counts_for_specs


def order_active_specs_by_quality(
    active_specs: list[ProposalSpec],
    ledger: Ledger,
    *,
    tier2_feedback_by_id: dict[str, Tier2Feedback],
    nas_screen_by_id: dict[str, NasScreenResult],
    max_graded_per_cycle: int = 0,
) -> tuple[list[ProposalSpec], dict[str, int]]:
    """Rank active specs by fused quality, applying the budget split if capped."""

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


def physics_s05_prescreen_specs(
    specs: list[ProposalSpec],
    ledger: Ledger,
    *,
    cycle: int,
    dim: int,
    seq_len: int,
) -> tuple[list[ProposalSpec], int]:
    """Reject physics-atom specs that fail the S0.5 causality-stability gate."""

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


def top_orthogonality_pending(
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


def inject_novelty_regrades(
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
        for s in top_orthogonality_pending(pool, ledger, k)
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
    """Order/budget the screened pool — surrogate-UCB or legacy quality split."""

    if selection == "surrogate":
        ordered = select_by_acquisition(
            active_specs,
            MeanFieldApproximant.fit(),
            budget=max_graded_per_cycle,
            beta=acquisition_beta,
        )
        return ordered, bucket_counts(())
    if use_quality_order:
        return order_active_specs_by_quality(
            active_specs,
            ledger,
            tier2_feedback_by_id=tier2_feedback_by_id,
            nas_screen_by_id=nas_screen_by_id,
            max_graded_per_cycle=max_graded_per_cycle,
        )
    return active_specs, bucket_counts(())


def select_active_specs(
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
    active_specs, n_physics_s05_prescreen_failed = physics_s05_prescreen_specs(
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
    active_specs = inject_novelty_regrades(
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
