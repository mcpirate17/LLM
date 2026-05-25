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
import datetime as _dt
import json
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path


from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.improver.adaptive import (
    adaptive_axis_variants,
    adaptive_cross_anchor_variants,
    build_anchor_pool,
)
from component_fab.improver.axis_variants import enumerate_axis_variants
from component_fab.improver.cross_anchor import enumerate_cross_anchor_variants
from component_fab.improver.math_knob_catalog import (
    enumerate_adaptive_math_knob_compositions,
)
from component_fab.improver.ranking import (
    composite_score,
    leaderboard_to_json,
    rank_proposals,
)
from component_fab.intake.scope_existing import scope_all
from component_fab.policies.promotion import (
    DEFAULT_PROMOTION_RULES,
    PromotionRules,
    apply_decisions,
    decide_promotions_for_ledger,
)
from component_fab.proposer.spec_generator import (
    ProposalSpec,
    dedupe_specs_by_axes,
)
from component_fab.state.ledger import (
    Ledger,
    PROMOTION_PROMOTED,
    PROMOTION_REJECTED,
    _prune_rotations,
)
from component_fab.validator.capability import (
    capability_scorecard_to_dict,
    validate_capabilities,
)
from component_fab.validator.in_context import validate_in_context
from component_fab.validator.solo import SoloScorecard, append_scorecard, validate_solo

_REPO = Path(__file__).resolve().parents[2]
_CATALOG_DIR = _REPO / "component_fab" / "catalog"
_DEFAULT_TOP_N_ANCHORS = 5


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="component_fab autonomous loop")
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
    return parser.parse_args(argv)


_INTERRUPTED = False


def _install_signal_handler() -> None:
    def _handler(signum, frame) -> None:  # noqa: ARG001
        global _INTERRUPTED
        _INTERRUPTED = True
        print("\n[interrupt received — halting after this cycle]", flush=True)

    signal.signal(signal.SIGINT, _handler)


def _gather_anchors(top_n: int) -> list[str]:
    report = scope_all()
    return [t["name"] for t in report["underperforming_novel_ops"][:top_n]]


def _all_specs_for_cycle(
    anchors: list[str],
    ledger: Ledger,
    *,
    use_promoted_as_anchors: bool,
    max_cross_pairs: int,
    max_knob_specs: int,
    cycle: int,
) -> list[ProposalSpec]:
    knob_specs = enumerate_adaptive_math_knob_compositions(
        anchors,
        ledger,
        max_specs=max_knob_specs,
    )
    if not use_promoted_as_anchors:
        return dedupe_specs_by_axes(
            enumerate_axis_variants(anchors)
            + enumerate_cross_anchor_variants(anchors)
            + knob_specs
        )
    anchor_pool = build_anchor_pool(
        anchors,
        ledger,
        use_promoted_as_anchors=True,
    )
    axis_specs = adaptive_axis_variants(anchor_pool, ledger)
    cross_specs = adaptive_cross_anchor_variants(
        anchor_pool,
        ledger,
        max_pairs=max_cross_pairs,
        seed=cycle,
    )
    return dedupe_specs_by_axes(axis_specs + cross_specs + knob_specs)


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
    name recorded.
    """
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
        fake_smoke = {
            "forward_passed": True,
            "backward_passed": True,
            "output_finite": True,
            "param_grad_finite": True,
            "eliminated_by": capability.eliminated_by,
        }
        solo = SoloScorecard(
            proposal_id=spec.proposal_id,
            name=spec.name,
            category=spec.category,
            synthesis_kind=spec.synthesis_kind,
            math_axes=dict(spec.math_axes),
            smoke=fake_smoke,
            metrics={"skipped": f"eliminated_by_{capability.eliminated_by}"},
            property_cross_check={},
            promoted=False,
        )
        return solo, None, capability_dict, capability.eliminated_by

    solo = validate_solo(spec, module, dim=dim, seq_len=seq_len)
    append_scorecard(solo)
    probe_dict: dict | None = None
    if not skip_probe and solo.promoted:
        probe = validate_in_context(
            spec,
            module,
            dim=dim,
            seq_len=seq_len,
            n_steps=probe_steps,
        )
        probe_dict = asdict(probe)
    return solo, probe_dict, capability_dict, None


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
) -> tuple[list[dict], dict[str, dict], dict[str, dict], dict[str, int]]:
    cycle_scorecards: list[dict] = []
    cycle_probes: dict[str, dict] = {}
    cycle_capabilities: dict[str, dict] = {}
    eliminated_by_gate: dict[str, int] = {}
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
        math_knobs = str(spec.math_axes.get("op_math_knobs") or "")
        metadata = {
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
                int(capability.get("range_effective_distance") or 0)
                if capability
                else 0
            ),
            "range_ran": bool(capability and capability.get("range_ran")),
        }
        if eliminated_by is not None:
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
            eliminated_by_gate[eliminated_by] = (
                eliminated_by_gate.get(eliminated_by, 0) + 1
            )
            continue
        score, _ = composite_score(asdict(solo), probe, capability)
        ledger.record_grade(
            proposal_id=spec.proposal_id,
            name=solo.name,
            category=solo.category,
            synthesis_kind=solo.synthesis_kind,
            cycle=cycle,
            composite_score=score,
            smoke_pass=bool(
                solo.smoke.get("forward_passed") and solo.smoke.get("backward_passed")
            ),
            learned_signal=bool(probe and probe.get("learned_signal")),
            metadata=metadata,
        )
    return cycle_scorecards, cycle_probes, cycle_capabilities, eliminated_by_gate


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
    run_range_probe: bool = False,
    range_train_steps: int = 300,
    promotion_rules: PromotionRules = DEFAULT_PROMOTION_RULES,
) -> dict:
    anchors = _gather_anchors(top_anchors)
    specs = _all_specs_for_cycle(
        anchors,
        ledger,
        use_promoted_as_anchors=use_promoted_as_anchors,
        max_cross_pairs=max_cross_pairs,
        max_knob_specs=max_knob_specs,
        cycle=cycle,
    )
    # Re-grade every spec each cycle so the ledger accumulates score history;
    # promotion requires a streak across cycles to fire. Skip only proposals
    # that have already reached a terminal status (promoted or rejected).
    skippable = {
        pid
        for pid, entry in ledger.entries.items()
        if entry.promotion_status in (PROMOTION_PROMOTED, PROMOTION_REJECTED)
    }
    active_specs = [s for s in specs if s.proposal_id not in skippable]
    n_new_proposals = sum(1 for s in active_specs if not ledger.has_seen(s.proposal_id))

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
        )
    )

    decisions = decide_promotions_for_ledger(ledger, promotion_rules)
    counts = apply_decisions(ledger, decisions)
    ranked = rank_proposals(cycle_scorecards, cycle_probes, cycle_capabilities)
    n_can_bind = sum(1 for c in cycle_capabilities.values() if c.get("can_bind"))
    return {
        "cycle": cycle,
        "anchors": anchors,
        "n_specs_considered": len(specs),
        "n_active_regraded": len(active_specs),
        "n_new_proposals": n_new_proposals,
        "n_terminal_skipped": len(skippable),
        "n_eliminated": sum(eliminated_by_gate.values()),
        "eliminated_by_gate": dict(eliminated_by_gate),
        "n_can_bind": n_can_bind,
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
    index = 1
    while (
        rotated := proposals_path.with_suffix(proposals_path.suffix + f".{index}")
    ).exists():
        index += 1
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
            run_range_probe=args.range_probe,
            range_train_steps=args.range_train_steps,
            promotion_rules=PromotionRules(
                veto_range_blind=args.veto_range_blind,
                min_range_effective_distance=args.min_range_distance,
            ),
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
    ledger = Ledger(ledger_path, include_rotated=True)

    cycle_summaries = _drive_loop(args, ledger, proposals_path)

    out_path = None
    pruned_run_summaries = 0
    if args.emit_run_summary:
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = _CATALOG_DIR / f"autonomous_run_{timestamp}.json"
        out_path.write_text(
            json.dumps(
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
                indent=2,
                default=str,
            ),
            encoding="utf-8",
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
