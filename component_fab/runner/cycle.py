"""One-cycle orchestration for the component_fab autonomous runner."""

from __future__ import annotations

from component_fab.improver.ranking import leaderboard_to_json, rank_proposals
from component_fab.intake.scope_existing import top_underperforming_names
from component_fab.policies.promotion import (
    DEFAULT_PROMOTION_RULES,
    PromotionRules,
    apply_decisions,
    decide_promotions_for_ledger,
)
from component_fab.proposer.enumeration import enumerate_cycle_specs
from component_fab.proposer.tier2_feedback import load_tier2_feedback
from component_fab.runner.grading import grade_active_specs
from component_fab.runner.promotion import register_promoted
from component_fab.runner.selection import select_active_specs
from component_fab.state.ledger import Ledger, PROMOTION_PROMOTED, PROMOTION_REJECTED


def run_cycle(
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
    """Run one autonomous propose/select/grade/promote cycle."""

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
        select_active_specs(
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
        grade_active_specs(
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
    register_promoted(ledger, decisions)
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


def print_cycle(summary: dict) -> None:
    """Print the human-readable one-cycle summary."""

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
