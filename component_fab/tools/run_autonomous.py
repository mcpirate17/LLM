"""CLI: fully autonomous fab loop — runs N cycles end-to-end with no operator.

One cycle:
  1. scope existing components (intake)
  2. enumerate goal-(b) axis-variants + cross-anchor variants
  3. dispatch each spec to a runnable nn.Module (code_generator)
  4. solo-grade (smoke + cross-check)
  5. in-context probe-suite grade (multi-task training)
  6. composite-rank
  7. update persistent ledger
  8. consult promotion policy
  9. print human-readable cycle summary

Halts when N cycles complete OR when M consecutive cycles produce no
new promotions and no new candidates.

Usage:
    python -m component_fab.tools.run_autonomous --cycles 5
    python -m component_fab.tools.run_autonomous --cycles 10 --probe-steps 60 --halt-quiescent 3
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path


from component_fab.metrics.behavior_fingerprint import (
    Normalizer,
    behavior_fingerprint,
    fingerprint_from_metadata,
    is_clone,
    novelty_distance,
)
from component_fab.improver.ranking import (
    composite_score,
    objective_vector,
    pareto_front_indices,
    leaderboard_to_json,
    rank_proposals,
)
from component_fab.policies.promotion import (
    DEFAULT_PROMOTION_RULES,
    PromotionRules,
    apply_decisions,
    decide_promotions_for_ledger,
)
from component_fab.intake.scope_existing import top_underperforming_names
from component_fab.proposer.acquisition import select_by_acquisition
from component_fab.proposer.enumeration import enumerate_cycle_specs
from component_fab.proposer.spec_generator import ProposalSpec
from component_fab.proposer.nas_screen import (
    NasScreenResult,
    nas_score_multiplier,
    score_specs_with_nas,
)
from component_fab.proposer.quality import (
    allocate_budget_buckets,
    bucket_counts,
    score_specs_quality,
)
from component_fab.proposer.tier2_feedback import (
    Tier2Feedback,
    load_tier2_feedback,
    tier2_score_multiplier,
)
from component_fab.validator.trust import axes_counts_for_specs
from component_fab.state.ledger import (
    Ledger,
    PROMOTION_PROMOTED,
    PROMOTION_REJECTED,
    _prune_rotations,
)
from component_fab.proposer.dynamic import spec_from_ledger_entry
from component_fab.state.aria_registration import register_promotion
from component_fab.state.surrogate import Surrogate
from component_fab.tools._cli import open_ledger, write_report
from component_fab.validator.grade import eliminated_solo_scorecard, grade_candidate
from component_fab.validator.paired import paired_metadata_for_spec
from component_fab.validator.solo import (
    SoloScorecard,
    close_scorecard_writers,
)

_REPO = Path(__file__).resolve().parents[2]
_CATALOG_DIR = _REPO / "component_fab" / "catalog"
_DEFAULT_TOP_N_ANCHORS = 5


def _add_loop_args(parser: argparse.ArgumentParser) -> None:
    """Cycle/budget/enumeration knobs for the outer loop."""
    parser.add_argument("--cycles", default=5, type=int)
    parser.add_argument("--dim", default=32, type=int)
    parser.add_argument("--seq-len", default=32, type=int)
    parser.add_argument("--probe-steps", default=60, type=int)
    parser.add_argument("--top-anchors", default=_DEFAULT_TOP_N_ANCHORS, type=int)
    parser.add_argument(
        "--halt-quiescent",
        default=2,
        type=int,
        help="halt after this many consecutive cycles with no new candidates",
    )
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--reset-ledger", action="store_true")
    parser.add_argument(
        "--use-promoted-as-anchors",
        action="store_true",
        help="feed promoted fab components back as anchors for compounding",
    )
    parser.add_argument("--max-cross-pairs", default=30, type=int)
    parser.add_argument("--max-knob-specs", default=48, type=int)
    parser.add_argument(
        "--max-nas-specs",
        default=6,
        type=int,
        help="fresh NAS-synthesized graph topologies to grade per cycle (0 disables)",
    )
    parser.add_argument(
        "--nas-archive-guided",
        action="store_true",
        help="bias NAS grammar sampling toward empty behaviour niches in the "
        "cached NAS population (anti-collapse) instead of random seeds",
    )
    parser.add_argument(
        "--max-dynamic-specs",
        default=32,
        type=int,
        help="max ledger-feedback proposals synthesized per cycle",
    )
    parser.add_argument(
        "--tier2-feedback",
        nargs="*",
        default=None,
        help="optional Tier-2 cohort JSON artifacts to feed proposal repair and scoring",
    )
    parser.add_argument(
        "--time-budget-minutes",
        default=None,
        type=float,
        help="continuous mode — run until this wall-clock budget elapses (overrides --cycles)",
    )
    parser.add_argument(
        "--rotate-at-mb",
        default=2,
        type=float,
        help="rotate ledger.jsonl + proposals.jsonl when they exceed this size",
    )
    parser.add_argument(
        "--emit-run-summary",
        action="store_true",
        help="write component_fab/catalog/autonomous_run_<timestamp>.json",
    )
    parser.add_argument("--quiet", action="store_true")


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    """Screening/ordering/budgeting of which candidates get graded."""
    parser.add_argument(
        "--disable-nas-screen",
        action="store_true",
        help="disable cheap NAS/oracle screening multiplier for fab candidates",
    )
    parser.add_argument(
        "--disable-quality-order",
        action="store_true",
        help="disable fused-quality ordering of candidates before grading "
        "(ordering is additive; it does not change which specs are graded unless "
        "--max-graded-per-cycle is set)",
    )
    parser.add_argument(
        "--max-graded-per-cycle",
        default=0,
        type=int,
        help="if >0, grade only this many specs per cycle, filled by the "
        "60/25/15 exploit/repair/exploration quality-budget split",
    )
    parser.add_argument(
        "--selection",
        default="legacy",
        choices=("legacy", "surrogate"),
        help=(
            "WS-3 candidate selection. 'legacy' = quality-order + static caps. "
            "'surrogate' = fill the per-cycle grading budget (--max-graded-per-cycle) "
            "with the ledger surrogate's highest-UCB candidates. Default legacy "
            "until run_surrogate reports acceptance_passed=True."
        ),
    )
    parser.add_argument(
        "--acquisition-beta",
        default=1.0,
        type=float,
        help="UCB exploration weight for --selection surrogate (median + beta*(upper-median)).",
    )


def _add_promotion_args(parser: argparse.ArgumentParser) -> None:
    """Evidence gathering + promotion-rule knobs."""
    parser.add_argument(
        "--range-probe",
        action="store_true",
        help="Run the sparse/long-range binding probe during grading (adds cost; "
        "scan lanes are slow). Populates range_effective_distance metadata.",
    )
    parser.add_argument("--range-train-steps", default=300, type=int)
    parser.add_argument(
        "--veto-range-blind",
        action="store_true",
        help="Block promotion of candidates whose MEASURED range_effective_distance "
        "is below --min-range-distance (no effect without --range-probe).",
    )
    parser.add_argument("--min-range-distance", default=1, type=int)
    parser.add_argument(
        "--niche-promotion",
        action="store_true",
        help=(
            "WS-4/WS-5: tag each survivor with a behavioral-novelty distance and "
            "first-Pareto-front membership, and let a front member promote in its "
            "niche (PromotionRules.promote_by_pareto) even below the scalar bar. "
            "Off by default — the legacy scalar composite stays the sole gate."
        ),
    )
    parser.add_argument(
        "--paired-seeds",
        default=0,
        type=int,
        help=(
            "WS-2: if > 0, grade each surviving spec against its anchor baseline "
            "on this many shared seeds and record the paired-delta 95%% CI. "
            "The default promotion policy is fail-closed: 0 preserves loop cost "
            "but leaves new streak-eligible candidates pending because CI evidence "
            "is absent. 3-5 recommended for promotion-capable runs."
        ),
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="component_fab autonomous loop")
    _add_loop_args(parser)
    _add_selection_args(parser)
    _add_promotion_args(parser)
    return parser.parse_args(argv)


_INTERRUPTED = False


def _install_signal_handler() -> None:
    def _handler(_signum, _frame) -> None:
        global _INTERRUPTED
        _INTERRUPTED = True
        print("\n[interrupt received — halting after this cycle]", flush=True)

    signal.signal(signal.SIGINT, _handler)


def _grade_spec(
    spec: ProposalSpec,
    *,
    dim: int,
    seq_len: int,
    probe_steps: int,
    skip_probe: bool,
    run_range_probe: bool = False,
    range_train_steps: int = 300,
) -> tuple[SoloScorecard, dict | None, dict, str | None]:
    """Return ``(solo, probe, capability, eliminated_by)``.

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
        return solo, None, bundle.capability, bundle.eliminated_by
    assert bundle.solo is not None  # run_solo=True and not eliminated
    probe_dict = asdict(bundle.in_context) if bundle.in_context is not None else None
    return bundle.solo, probe_dict, bundle.capability, None


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


def _metadata_for_grade(
    spec: ProposalSpec, capability: dict | None, eliminated_by: str | None
) -> dict:
    """Ledger metadata for one graded spec (pure assembly, no side effects)."""
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
        # Persist the full build recipe so promoted specs stay re-gradeable
        # from the ledger (generate_module is a pure function of math_axes).
        "math_axes": dict(spec.math_axes),
        # Range signal (only populated when --range-probe is on); feeds the
        # optional veto_range_blind promotion rule.
        "range_effective_distance": (
            int(capability.get("range_effective_distance") or 0) if capability else 0
        ),
        "range_ran": bool(capability and capability.get("range_ran")),
    }


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
        solo, probe, capability, eliminated_by = _grade_spec(
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
        metadata = _metadata_for_grade(spec, capability, eliminated_by)
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


def _register_promoted(ledger: Ledger, decisions: list) -> None:
    """WS-7 loop closure: emit an ARIA handoff row for each FRESH promotion."""
    for decision in decisions:
        if (
            decision.decision != PROMOTION_PROMOTED
            or decision.reason == "already promoted"
        ):
            continue
        entry = ledger.entries.get(decision.proposal_id)
        if entry is None:
            continue
        spec = spec_from_ledger_entry(entry)
        if spec is None:
            continue
        meta = entry.metadata_history[-1] if entry.metadata_history else {}
        evidence = {
            "composite": entry.composite_history[-1]
            if entry.composite_history
            else 0.0,
            "transplant_portability": meta.get("transplant_portability"),
            "on_pareto_front": meta.get("on_pareto_front"),
        }
        register_promotion(spec, evidence=evidence)


def _select_active_specs(
    specs: list[ProposalSpec],
    ledger: Ledger,
    *,
    selection: str,
    acquisition_beta: float,
    use_nas_screen: bool,
    use_quality_order: bool,
    max_graded_per_cycle: int,
    tier2_feedback_by_id: dict[str, Tier2Feedback],
) -> tuple[list[ProposalSpec], dict[str, NasScreenResult], dict[str, int], int, int]:
    """Filter terminal specs, screen, and order/budget the grading queue.

    Returns ``(active_specs, nas_screen_by_id, bucket_summary,
    n_new_proposals, n_terminal_skipped)``.
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
    nas_screen_by_id = score_specs_with_nas(active_specs, enabled=use_nas_screen)
    n_new_proposals = sum(1 for s in active_specs if not ledger.has_seen(s.proposal_id))

    bucket_summary = bucket_counts(())
    if selection == "surrogate":
        # WS-3: fill the grading budget with the surrogate's highest-UCB
        # candidates instead of the legacy quality order. max_graded_per_cycle
        # is the budget (0 => no cap). Falls back to identity order if the
        # surrogate can't fit (too little ledger history).
        active_specs = select_by_acquisition(
            active_specs,
            Surrogate.fit(),
            budget=max_graded_per_cycle,
            beta=acquisition_beta,
        )
    elif use_quality_order:
        active_specs, bucket_summary = _order_active_specs_by_quality(
            active_specs,
            ledger,
            tier2_feedback_by_id=tier2_feedback_by_id,
            nas_screen_by_id=nas_screen_by_id,
            max_graded_per_cycle=max_graded_per_cycle,
        )
    return (
        active_specs,
        nas_screen_by_id,
        bucket_summary,
        n_new_proposals,
        len(skippable),
    )


def _run_cycle(
    cycle: int,
    *,
    ledger: Ledger,
    dim: int,
    seq_len: int,
    probe_steps: int,
    top_anchors: int,
    skip_probe: bool,
    use_promoted_as_anchors: bool = False,
    max_cross_pairs: int = 30,
    max_knob_specs: int = 48,
    max_dynamic_specs: int = 32,
    max_nas_specs: int = 6,
    nas_archive_guided: bool = False,
    run_range_probe: bool = False,
    range_train_steps: int = 300,
    tier2_feedback_paths: list[str] | None = None,
    use_nas_screen: bool = True,
    use_quality_order: bool = True,
    max_graded_per_cycle: int = 0,
    promotion_rules: PromotionRules = DEFAULT_PROMOTION_RULES,
    paired_seeds: int = 0,
    selection: str = "legacy",
    acquisition_beta: float = 1.0,
    niche_promotion: bool = False,
) -> dict:
    anchors = top_underperforming_names(top_anchors)
    tier2_feedback_by_id = load_tier2_feedback(tier2_feedback_paths)
    specs = enumerate_cycle_specs(
        ledger,
        anchors,
        cycle=cycle,
        dim=dim,
        use_promoted_as_anchors=use_promoted_as_anchors,
        max_cross_pairs=max_cross_pairs,
        max_knob_specs=max_knob_specs,
        max_dynamic_specs=max_dynamic_specs,
        max_nas_specs=max_nas_specs,
        nas_archive_guided=nas_archive_guided,
        tier2_feedback_by_id=tier2_feedback_by_id,
    )
    active_specs, nas_screen_by_id, bucket_summary, n_new_proposals, n_skipped = (
        _select_active_specs(
            specs,
            ledger,
            selection=selection,
            acquisition_beta=acquisition_beta,
            use_nas_screen=use_nas_screen,
            use_quality_order=use_quality_order,
            max_graded_per_cycle=max_graded_per_cycle,
            tier2_feedback_by_id=tier2_feedback_by_id,
        )
    )

    cycle_scorecards, cycle_probes, cycle_capabilities, eliminated_by_gate = (
        _grade_active_specs(
            active_specs,
            ledger,
            cycle=cycle,
            dim=dim,
            seq_len=seq_len,
            probe_steps=probe_steps,
            skip_probe=skip_probe,
            run_range_probe=run_range_probe,
            range_train_steps=range_train_steps,
            tier2_feedback_by_id=tier2_feedback_by_id,
            nas_screen_by_id=nas_screen_by_id,
            paired_seeds=paired_seeds,
            niche_promotion=niche_promotion,
        )
    )

    decisions = decide_promotions_for_ledger(ledger, promotion_rules)
    counts = apply_decisions(ledger, decisions)
    _register_promoted(ledger, decisions)
    ranked = rank_proposals(
        cycle_scorecards,
        cycle_probes,
        cycle_capabilities,
        tier2_feedback_by_id=tier2_feedback_by_id,
        nas_screen_by_id=nas_screen_by_id,
    )
    n_can_bind = sum(1 for c in cycle_capabilities.values() if c.get("can_bind"))
    return {
        "cycle": cycle,
        "anchors": anchors,
        "n_specs_considered": len(specs),
        "n_active_regraded": len(active_specs),
        "n_new_proposals": n_new_proposals,
        "n_terminal_skipped": n_skipped,
        "n_eliminated": sum(eliminated_by_gate.values()),
        "eliminated_by_gate": dict(eliminated_by_gate),
        "n_can_bind": n_can_bind,
        "quality_buckets": bucket_summary,
        "promotion_counts": counts,
        "top_5": leaderboard_to_json(ranked)[:5],
    }


def _print_cycle(summary: dict) -> None:
    print(f"\n=== cycle {summary['cycle']} ===")
    print(f"anchors:          {', '.join(summary['anchors'])}")
    print(f"specs considered: {summary['n_specs_considered']}")
    print(
        f"active regraded:  {summary['n_active_regraded']} "
        f"(new: {summary['n_new_proposals']}, "
        f"terminal-skipped: {summary['n_terminal_skipped']})"
    )
    eliminated = summary.get("eliminated_by_gate", {})
    print(
        f"gate eliminations: total {summary.get('n_eliminated', 0)} "
        f"(s05={eliminated.get('s05_causality_stability', 0)}, "
        f"erf={eliminated.get('erf_density', 0)}, "
        f"nb={eliminated.get('nano_bind', 0)})"
    )
    print(f"AR binders:       {summary.get('n_can_bind', 0)} passed the binding probe")
    buckets = summary.get("quality_buckets", {})
    if buckets:
        print(
            f"quality buckets:  exploit={buckets.get('exploit', 0)}, "
            f"repair={buckets.get('repair', 0)}, "
            f"exploration={buckets.get('exploration', 0)}"
        )
    counts = summary["promotion_counts"]
    print(
        f"promotions:       {counts.get(PROMOTION_PROMOTED, 0)} promoted, "
        f"{counts.get(PROMOTION_REJECTED, 0)} rejected, "
        f"{counts.get('pending', 0)} pending"
    )
    print("top 5 this cycle:")
    for row in summary["top_5"]:
        c = row["components"]
        print(
            f"  {row['rank']}. {row['name']:<50} "
            f"score={row['composite_score']:.3f}  "
            f"(smoke={c['smoke']:.2f} cross={c['cross_check']:.2f} learn={c['learning']:.2f})"
        )


def _rotate_proposals(proposals_path: Path, rotate_bytes: int, quiet: bool) -> None:
    if not proposals_path.exists() or proposals_path.stat().st_size < rotate_bytes:
        return
    # One past the highest existing index — monotonic even after pruning
    # frees lower indices, so suffix order matches chronology (the ledger's
    # rotated-replay sorts by mtime, but keep numbering sane regardless).
    prefix = proposals_path.name + "."
    existing = [
        int(child.name[len(prefix) :])
        for child in proposals_path.parent.glob(f"{proposals_path.name}.*")
        if child.name[len(prefix) :].isdigit()
    ]
    rotated = proposals_path.with_suffix(
        proposals_path.suffix + f".{max(existing, default=0) + 1}"
    )
    proposals_path.rename(rotated)
    proposals_path.touch()
    deleted = _prune_rotations(proposals_path)
    if not quiet:
        print(f"[rotated proposals.jsonl → {rotated.name}]")
        if deleted:
            print(f"[pruned {deleted} old proposals.jsonl rotations]")


def _prune_autonomous_run_summaries(catalog_dir: Path, keep: int = 3) -> int:
    summaries = sorted(
        catalog_dir.glob("autonomous_run_*.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    deleted = 0
    for stale in summaries[keep:]:
        stale.unlink(missing_ok=True)
        deleted += 1
    return deleted


def _should_halt(
    summary: dict,
    quiescent_streak: int,
    halt_quiescent: int,
    quiet: bool,
) -> tuple[bool, int]:
    moved = (
        summary["n_new_proposals"] > 0
        or summary["promotion_counts"].get(PROMOTION_PROMOTED, 0) > 0
        or summary["promotion_counts"].get(PROMOTION_REJECTED, 0) > 0
    )
    quiescent_streak = 0 if moved else quiescent_streak + 1
    if quiescent_streak >= halt_quiescent:
        if not quiet:
            print(f"\nhalting: {quiescent_streak} consecutive cycles with no movement")
        return True, quiescent_streak
    if summary["n_active_regraded"] == 0:
        if not quiet:
            print("\nhalting: every proposal has reached a terminal status")
        return True, quiescent_streak
    return False, quiescent_streak


def _drive_loop(args, ledger: Ledger, proposals_path: Path) -> list[dict]:
    rotate_bytes = int(args.rotate_at_mb * 1_048_576)
    started = time.monotonic()

    def _budget_exhausted() -> bool:
        if args.time_budget_minutes is None:
            return False
        return (time.monotonic() - started) / 60.0 >= args.time_budget_minutes

    cycle_summaries: list[dict] = []
    quiescent_streak = 0
    cycle = 0
    while True:
        cycle += 1
        if args.time_budget_minutes is None and cycle > args.cycles:
            break
        if _budget_exhausted():
            if not args.quiet:
                print(
                    f"\nhalting: wall-clock budget {args.time_budget_minutes}m exhausted"
                )
            break
        if _INTERRUPTED:
            break
        summary = _run_cycle(
            cycle,
            ledger=ledger,
            dim=args.dim,
            seq_len=args.seq_len,
            probe_steps=args.probe_steps,
            top_anchors=args.top_anchors,
            skip_probe=args.skip_probe,
            use_promoted_as_anchors=args.use_promoted_as_anchors,
            max_cross_pairs=args.max_cross_pairs,
            max_knob_specs=args.max_knob_specs,
            max_dynamic_specs=args.max_dynamic_specs,
            max_nas_specs=args.max_nas_specs,
            nas_archive_guided=args.nas_archive_guided,
            run_range_probe=args.range_probe,
            range_train_steps=args.range_train_steps,
            tier2_feedback_paths=args.tier2_feedback,
            use_nas_screen=not args.disable_nas_screen,
            use_quality_order=not args.disable_quality_order,
            max_graded_per_cycle=args.max_graded_per_cycle,
            promotion_rules=PromotionRules(
                veto_range_blind=args.veto_range_blind,
                min_range_effective_distance=args.min_range_distance,
                promote_by_pareto=args.niche_promotion,
            ),
            paired_seeds=args.paired_seeds,
            selection=args.selection,
            acquisition_beta=args.acquisition_beta,
            niche_promotion=args.niche_promotion,
        )
        cycle_summaries.append(summary)
        if not args.quiet:
            _print_cycle(summary)

        rotated_ledger = ledger.rotate_if_oversized(rotate_bytes)
        if rotated_ledger and not args.quiet:
            print(f"[rotated ledger.jsonl → {rotated_ledger.name}]")
        _rotate_proposals(proposals_path, rotate_bytes, args.quiet)

        halted, quiescent_streak = _should_halt(
            summary,
            quiescent_streak,
            args.halt_quiescent,
            args.quiet,
        )
        if halted:
            break
    return cycle_summaries


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    _install_signal_handler()
    _CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path = _CATALOG_DIR / "ledger.jsonl"
    proposals_path = _CATALOG_DIR / "proposals.jsonl"
    if args.reset_ledger and ledger_path.exists():
        ledger_path.unlink()
    # Replay rotated audit trail so promoted-fab anchors carried forward
    # from prior days remain visible to adaptive_axis_variants. Without this
    # the anchor pool collapses to whatever the most-recent rotation kept.
    ledger = open_ledger(ledger_path)

    try:
        cycle_summaries = _drive_loop(args, ledger, proposals_path)
    finally:
        # Explicit flush+release of every cached JSONL handle (ledger +
        # per-path scorecard writers) so a crash mid-run can't strand a
        # buffered tail line.
        ledger.close()
        close_scorecard_writers()

    out_path = None
    pruned_run_summaries = 0
    if args.emit_run_summary:
        out_path = write_report(
            {
                "cycles_run": len(cycle_summaries),
                "summaries": cycle_summaries,
                "ledger_size": len(ledger.entries),
                "promoted_components": [
                    asdict(entry)
                    for entry in ledger.all_entries()
                    if entry.promotion_status == PROMOTION_PROMOTED
                ],
            },
            default_dir=_CATALOG_DIR,
            prefix="autonomous_run",
            quiet=True,  # the summary block below reports the path
        )
        pruned_run_summaries = _prune_autonomous_run_summaries(_CATALOG_DIR)
    if not args.quiet:
        promoted = sum(
            1
            for entry in ledger.all_entries()
            if entry.promotion_status == PROMOTION_PROMOTED
        )
        print(
            f"\nautonomous run complete: {len(cycle_summaries)} cycles, "
            f"{len(ledger.entries)} total proposals tracked, {promoted} promoted"
        )
        if out_path is not None:
            print(f"wrote: {out_path}")
            if pruned_run_summaries:
                print(f"pruned {pruned_run_summaries} old autonomous run summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
