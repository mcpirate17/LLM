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

The CLI surface lives in ``_autonomous_cli`` and the per-spec grading
pipeline in ``_autonomous_grading``; this module owns the cycle/loop
orchestration and promotion bookkeeping.

Usage:
    python -m component_fab.tools.run_autonomous --cycles 5
    python -m component_fab.tools.run_autonomous --cycles 10 --probe-steps 60 --halt-quiescent 3
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path

from component_fab.improver.ranking import (
    leaderboard_to_json,
    rank_proposals,
)
from component_fab.policies.promotion import (
    DEFAULT_PROMOTION_RULES,
    PromotionDecision,
    PromotionRules,
    apply_decisions,
    decide_promotions_for_ledger,
)
from component_fab.intake.scope_existing import top_underperforming_names
from component_fab.proposer.enumeration import enumerate_cycle_specs
from component_fab.proposer.tier2_feedback import load_tier2_feedback
from component_fab.state.ledger import (
    Ledger,
    PROMOTION_PROMOTED,
    PROMOTION_REJECTED,
    _prune_rotations,
)
from component_fab.proposer.dynamic import spec_from_ledger_entry
from component_fab.state.aria_registration import register_promotion
from component_fab.tools._cli import open_ledger, write_report
from component_fab.tools._autonomous_cli import (
    _install_signal_handler,
    _parse_args,
    _print_cycle,
    is_interrupted,
)
from component_fab.tools._autonomous_grading import (
    _annotate_niche_metadata,  # re-exported for tests
    _finalize_survivors,  # re-exported for tests
    _grade_active_specs,
    _select_active_specs,
)
from component_fab.validator.solo import close_scorecard_writers

__all__ = [
    "main",
    "_parse_args",
    "_rotate_proposals",
    "_prune_autonomous_run_summaries",
    "_annotate_niche_metadata",
    "_finalize_survivors",
]

_REPO = Path(__file__).resolve().parents[2]
_CATALOG_DIR = _REPO / "component_fab" / "catalog"


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


def _scale_gate_promotions(
    ledger: Ledger,
    decisions: list,
    *,
    dim: int,
    steps: int,
    seeds: int,
    seq_len: int,
) -> list:
    """Final gate before promotion: re-verify each FRESH promotion beats its
    anchor at SCALE.

    The nano paired-CI is not scale-predictive — at dim32/100 steps inventions
    showed tiny positive margins that INVERT to large losses at dim96/1500
    (validated 2026-06-16). So before a candidate promotes, re-run the paired
    probe (vs its catalog anchor, or the softmax-frontier fallback) at a larger
    width + many more steps. A candidate that does not beat its anchor at scale
    is REJECTED — terminal, so it is not re-tested (and re-promoted) every cycle —
    rather than minted as a scale-losing artifact.
    """
    from component_fab.validator.paired import paired_metadata_for_spec

    gated: list = []
    for decision in decisions:
        entry = ledger.entries.get(decision.proposal_id)
        # Only re-verify FRESH promotions (not already-promoted, not pending/reject).
        if (
            decision.decision != PROMOTION_PROMOTED
            or entry is None
            or entry.promotion_status == PROMOTION_PROMOTED
        ):
            gated.append(decision)
            continue
        spec = spec_from_ledger_entry(entry)
        if spec is None:
            gated.append(decision)
            continue
        md = paired_metadata_for_spec(
            spec, seeds=tuple(range(seeds)), dim=dim, seq_len=seq_len, n_steps=steps
        )
        beats = bool(md.get("paired_delta_ci_excludes_zero"))
        anchor = md.get("paired_anchor_op", "?")
        ci_low = md.get("paired_delta_ci_low")
        print(
            f"  scale-gate {decision.proposal_id[:24]} vs {anchor} "
            f"@dim{dim}/{steps}st: {'PASS' if beats else 'FAIL'} ci_low={ci_low}"
        )
        if beats:
            gated.append(decision)
        else:
            gated.append(
                PromotionDecision(
                    proposal_id=decision.proposal_id,
                    decision=PROMOTION_REJECTED,
                    reason=(
                        f"scale-gate: loses to {anchor} at dim{dim}/{steps}st "
                        f"(ci_low={ci_low})"
                    ),
                    composite_history=decision.composite_history,
                )
            )
    return gated


def _resolve_promotions(
    ledger: Ledger,
    promotion_rules: PromotionRules,
    *,
    scale_gate: bool,
    scale_gate_dim: int,
    scale_gate_steps: int,
    scale_gate_seeds: int,
    seq_len: int,
) -> dict[str, int]:
    """Decide promotions, optionally scale-gate the fresh ones, apply + register."""
    decisions = decide_promotions_for_ledger(ledger, promotion_rules)
    if scale_gate:
        decisions = _scale_gate_promotions(
            ledger,
            decisions,
            dim=scale_gate_dim,
            steps=scale_gate_steps,
            seeds=scale_gate_seeds,
            seq_len=seq_len,
        )
    counts = apply_decisions(ledger, decisions)
    _register_promoted(ledger, decisions)
    return counts


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
    regrade_top_orthogonality: int = 0,
    scale_gate: bool = False,
    scale_gate_dim: int = 96,
    scale_gate_steps: int = 1500,
    scale_gate_seeds: int = 5,
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
    (
        active_specs,
        nas_screen_by_id,
        bucket_summary,
        n_new_selected,
        n_new_available,
        n_skipped,
        n_physics_s05_skipped,
        n_physics_s05_prescreen_failed,
    ) = _select_active_specs(
        specs,
        ledger,
        cycle=cycle,
        dim=dim,
        seq_len=seq_len,
        selection=selection,
        acquisition_beta=acquisition_beta,
        use_nas_screen=use_nas_screen,
        use_quality_order=use_quality_order,
        max_graded_per_cycle=max_graded_per_cycle,
        tier2_feedback_by_id=tier2_feedback_by_id,
        regrade_top_orthogonality=regrade_top_orthogonality,
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

    counts = _resolve_promotions(
        ledger,
        promotion_rules,
        scale_gate=scale_gate,
        scale_gate_dim=scale_gate_dim,
        scale_gate_steps=scale_gate_steps,
        scale_gate_seeds=scale_gate_seeds,
        seq_len=seq_len,
    )
    ranked = rank_proposals(
        cycle_scorecards,
        cycle_probes,
        cycle_capabilities,
        tier2_feedback_by_id=tier2_feedback_by_id,
        nas_screen_by_id=nas_screen_by_id,
    )
    n_can_bind = sum(1 for c in cycle_capabilities.values() if c.get("can_bind"))
    physics_probe_task_ratios = {
        proposal_id: {
            name: round(float(result.get("loss_ratio_initial_over_final") or 0.0), 4)
            for name, result in (probe.get("per_task") or {}).items()
            if result.get("trained_successfully")
        }
        for proposal_id, probe in cycle_probes.items()
        if any(
            note.startswith("physics_probe_tasks=") for note in probe.get("notes", ())
        )
    }
    return {
        "cycle": cycle,
        "anchors": anchors,
        "n_specs_considered": len(specs),
        "n_active_regraded": len(active_specs),
        "n_new_proposals": n_new_selected,
        "n_new_available": n_new_available,
        "n_terminal_skipped": n_skipped,
        "n_physics_s05_skipped": n_physics_s05_skipped,
        "n_physics_s05_prescreen_failed": n_physics_s05_prescreen_failed,
        "n_eliminated": sum(eliminated_by_gate.values()),
        "eliminated_by_gate": dict(eliminated_by_gate),
        "n_can_bind": n_can_bind,
        "quality_buckets": bucket_summary,
        "physics_probe_task_ratios": physics_probe_task_ratios,
        "promotion_counts": counts,
        "top_5": leaderboard_to_json(ranked)[:5],
    }


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
        if is_interrupted():
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
            regrade_top_orthogonality=args.regrade_top_orthogonality,
            scale_gate=args.scale_gate,
            scale_gate_dim=args.scale_gate_dim,
            scale_gate_steps=args.scale_gate_steps,
            scale_gate_seeds=args.scale_gate_seeds,
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
