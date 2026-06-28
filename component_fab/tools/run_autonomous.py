"""CLI: fully autonomous fab loop.

The CLI owns argument parsing, signal handling, rotation, and report emission.
Cycle selection/grading/promotion orchestration lives under
``component_fab.runner`` so it can be tested without invoking the command line.

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

from component_fab.policies.promotion import PromotionRules
from component_fab.runner.cycle import print_cycle, run_cycle
from component_fab.state.ledger import Ledger, PROMOTION_PROMOTED, _prune_rotations
from component_fab.tools._cli import open_ledger, write_report
from component_fab.validator.solo import close_scorecard_writers

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
        help="bias NAS grammar sampling toward empty behaviour niches",
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
        help="optional Tier-2 cohort JSON artifacts to feed proposal repair/scoring",
    )
    parser.add_argument(
        "--time-budget-minutes",
        default=None,
        type=float,
        help="continuous mode; run until this wall-clock budget elapses",
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
    """Screening/ordering/budgeting knobs."""

    parser.add_argument("--disable-nas-screen", action="store_true")
    parser.add_argument("--disable-quality-order", action="store_true")
    parser.add_argument(
        "--max-graded-per-cycle",
        default=0,
        type=int,
        help="if >0, grade only this many specs per cycle",
    )
    parser.add_argument(
        "--selection",
        default="legacy",
        choices=("legacy", "surrogate"),
        help="candidate selection policy",
    )
    parser.add_argument("--acquisition-beta", default=1.0, type=float)
    parser.add_argument(
        "--regrade-top-orthogonality",
        default=0,
        type=int,
        help="force the top-K pending Pareto-front specs by peak orthogonality "
        "back into the grading budget so genuinely-novel candidates accumulate "
        "paired-CI (fixes the composite-selection-starves-novel pathology)",
    )


def _add_promotion_args(parser: argparse.ArgumentParser) -> None:
    """Evidence gathering and promotion-rule knobs."""

    parser.add_argument("--range-probe", action="store_true")
    parser.add_argument("--range-train-steps", default=300, type=int)
    parser.add_argument("--veto-range-blind", action="store_true")
    parser.add_argument("--min-range-distance", default=1, type=int)
    parser.add_argument("--niche-promotion", action="store_true")
    parser.add_argument(
        "--paired-seeds",
        default=0,
        type=int,
        help="if >0, grade each survivor against its anchor on this many seeds",
    )
    parser.add_argument(
        "--scale-gate",
        action="store_true",
        help="re-verify each fresh promotion beats its anchor at SCALE before "
        "promoting; a candidate that loses at scale is rejected, not minted",
    )
    parser.add_argument("--scale-gate-dim", default=96, type=int)
    parser.add_argument("--scale-gate-steps", default=1500, type=int)
    parser.add_argument("--scale-gate-seeds", default=5, type=int)


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


def _rotate_proposals(proposals_path: Path, rotate_bytes: int, quiet: bool) -> None:
    if not proposals_path.exists() or proposals_path.stat().st_size < rotate_bytes:
        return
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
        or summary["promotion_counts"].get("rejected", 0) > 0
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


def _drive_loop(
    args: argparse.Namespace, ledger: Ledger, proposals_path: Path
) -> list[dict]:
    rotate_bytes = int(args.rotate_at_mb * 1_048_576)
    started = time.monotonic()

    def budget_exhausted() -> bool:
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
        if budget_exhausted():
            if not args.quiet:
                print(
                    f"\nhalting: wall-clock budget {args.time_budget_minutes}m exhausted"
                )
            break
        if _INTERRUPTED:
            break
        summary = run_cycle(
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
            print_cycle(summary)
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
    ledger = open_ledger(ledger_path)

    try:
        cycle_summaries = _drive_loop(args, ledger, proposals_path)
    finally:
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
            quiet=True,
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
