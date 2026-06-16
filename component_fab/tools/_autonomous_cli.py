"""CLI surface for the autonomous fab loop: argparse, signal handling, printing.

Split out of ``run_autonomous`` (god-file split, behavior-preserving). The
loop orchestration lives in ``run_autonomous``; the per-spec grading pipeline
in ``_autonomous_grading``.
"""

from __future__ import annotations

import argparse
import signal

from component_fab.state.ledger import PROMOTION_PROMOTED, PROMOTION_REJECTED

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
        "--regrade-top-orthogonality",
        default=0,
        type=int,
        help=(
            "If > 0, force the top-K pending Pareto-front candidates by PEAK "
            "orthogonality (distance from softmax/frontier+ledger) into each "
            "cycle's grading budget. Counters the composite-selection pathology "
            "that starves genuinely-novel candidates so they never re-grade / "
            "accumulate the paired-CI a niche promotion needs. 0 = off (default)."
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


def is_interrupted() -> bool:
    """True once a SIGINT has been received (drains the loop after the cycle)."""
    return _INTERRUPTED


def _install_signal_handler() -> None:
    def _handler(_signum, _frame) -> None:
        global _INTERRUPTED
        _INTERRUPTED = True
        print("\n[interrupt received — halting after this cycle]", flush=True)

    signal.signal(signal.SIGINT, _handler)


def _print_cycle(summary: dict) -> None:
    print(f"\n=== cycle {summary['cycle']} ===")
    print(f"anchors:          {', '.join(summary['anchors'])}")
    print(f"specs considered: {summary['n_specs_considered']}")
    print(
        f"active regraded:  {summary['n_active_regraded']} "
        f"(new selected: {summary['n_new_proposals']}, "
        f"new available: {summary.get('n_new_available', summary['n_new_proposals'])}, "
        f"terminal-skipped: {summary['n_terminal_skipped']})"
    )
    eliminated = summary.get("eliminated_by_gate", {})
    print(
        f"gate eliminations: total {summary.get('n_eliminated', 0)} "
        f"(s05={eliminated.get('s05_causality_stability', 0)}, "
        f"erf={eliminated.get('erf_density', 0)}, "
        f"nb={eliminated.get('nano_bind', 0)})"
    )
    skipped_physics = int(summary.get("n_physics_s05_skipped", 0) or 0)
    if skipped_physics:
        print(f"physics S0.5 pre-skip: {skipped_physics} known-unstable coordinates")
    prescreen_failed = int(summary.get("n_physics_s05_prescreen_failed", 0) or 0)
    if prescreen_failed:
        print(f"physics S0.5 pre-fail: {prescreen_failed} newly unstable coordinates")
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
