"""Offline proposal-quality report for component_fab (read-only).

Fuses ledger evidence, Tier-2 cohort feedback, the NAS/oracle screen, novelty,
and curated external research priors into a quality-ranked candidate report. It
does not train models and does not mutate any DB; it writes a versioned JSON
artifact under ``research/reports/``.

Usage::

    python -m component_fab.tools.run_proposal_quality --top 40
    python -m component_fab.tools.run_proposal_quality --tier2 tasks/audit/fab_tier2_*.json --dry-run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from component_fab.proposer.dynamic import (
    enumerate_dynamic_proposals,
    specs_from_ledger_entries,
)
from component_fab.proposer.nas_screen import score_specs_with_nas
from component_fab.proposer.quality import (
    QualityScore,
    allocate_budget_buckets,
    bucket_counts,
    passing_only,
    score_specs_quality,
    verdict_counts,
)
from component_fab.proposer.research_priors import to_catalog_rows
from component_fab.proposer.spec_generator import ProposalSpec, dedupe_specs_by_axes
from component_fab.proposer.tier2_feedback import load_tier2_feedback
from component_fab.state.ledger import DEFAULT_LEDGER_PATH, Ledger
from component_fab.validator.trust import axes_counts_for_specs

_REPO = Path(__file__).resolve().parents[2]
_REPORTS_DIR = _REPO / "research" / "reports"
_REPORT_VERSION = "proposal_quality_v1"


def _gather_anchor_names(top_n: int) -> list[str]:
    from component_fab.intake.scope_existing import scope_all

    report = scope_all()
    return [t["name"] for t in report["underperforming_novel_ops"][:top_n]]


def _candidate_specs(
    ledger: Ledger, *, top_anchors: int, max_dynamic: int
) -> list[ProposalSpec]:
    specs = list(specs_from_ledger_entries(ledger))
    try:
        anchors = _gather_anchor_names(top_anchors)
        specs.extend(
            enumerate_dynamic_proposals(anchors, ledger, max_specs=max_dynamic)
        )
    except Exception as exc:  # noqa: BLE001 - intake is best-effort for an offline report
        print(f"[anchor intake unavailable: {exc}]", file=sys.stderr)
    return dedupe_specs_by_axes(specs)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="component_fab proposal-quality report"
    )
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument(
        "--tier2", nargs="*", default=None, help="Tier-2 cohort JSON artifacts"
    )
    parser.add_argument("--top-anchors", type=int, default=5)
    parser.add_argument("--max-dynamic", type=int, default=32)
    parser.add_argument(
        "--top", type=int, default=40, help="rows to keep in the ranked report"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=0,
        help="if >0, also emit the budget-bucketed grading queue of this size",
    )
    parser.add_argument(
        "--disable-nas-screen", action="store_true", help="skip the NAS screen"
    )
    parser.add_argument(
        "--capability-screen",
        action="store_true",
        help="attach induction/nano capability-screener predictions (DIAGNOSTIC only "
        "— validated cross-family but does NOT predict 'beats baseline' on fab; "
        "adds module-build cost per candidate)",
    )
    parser.add_argument(
        "--comparative-probe",
        type=int,
        default=0,
        metavar="N",
        help="run the label-free comparative binding probe on the top-N "
        "quality-ranked candidates (MEASURES beats-baseline margin; validated "
        "Spearman 0.54 vs full Tier-2; trains 2 models per candidate)",
    )
    parser.add_argument("--probe-steps", type=int, default=60)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _capability_diagnostics(specs: list[ProposalSpec]) -> dict[str, object]:
    """Induction/nano capability-screener predictions — DIAGNOSTIC, not a ranker.

    Validated cross-family but does NOT predict 'beats baseline' on the fab
    population (induction r~0, nano r~-0.28, audit 2026-06-03), so it is reported
    separately and never feeds the quality score.
    """

    from component_fab.proposer.capability_screen import (
        capability_screen_for_spec,
        load_capability_screeners,
    )

    try:
        models = load_capability_screeners()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)}
    preds = {}
    for spec in specs:
        cs = capability_screen_for_spec(spec, models=models)
        if cs.available:
            preds[spec.proposal_id] = {
                "induction_pred": round(cs.induction_pred, 4),
                "nano_pred": round(cs.nano_pred, 4),
            }
    return {
        "available": True,
        "note": "induction/nano CAPABILITY estimates — validated cross-family but "
        "NOT predictive of Tier-2 'beats baseline' on fab; diagnostic only",
        "predictions": preds,
    }


def _comparative_probe_top_n(
    ranked: list[QualityScore],
    specs_by_id: dict[str, ProposalSpec],
    *,
    top_n: int,
    probe_steps: int,
) -> dict[str, object]:
    """MEASURE (not predict) beats-baseline margin on the top-N quality shortlist.

    Label-free: runs the candidate + baseline through a short synthetic binding
    probe and compares. Validated Spearman 0.54 vs full Tier-2 (n=16).
    """

    from component_fab.proposer.comparative_probe import comparative_binding_screen

    rows = {}
    for score in ranked[:top_n]:
        spec = specs_by_id.get(score.proposal_id)
        if spec is None:
            continue
        cp = comparative_binding_screen(spec, n_train_steps=probe_steps)
        if cp.available:
            rows[score.proposal_id] = {
                "margin_mean": round(cp.margin_mean, 4),
                "beats_baseline": cp.beats_baseline,
                "baseline": cp.baseline_name,
                "per_task": cp.per_task,
            }
    return {
        "note": "label-free MEASURED beats-baseline margin (Spearman 0.54 vs full "
        "Tier-2); the only fab signal that tracks Tier-2 ranking",
        "probe_steps": probe_steps,
        "measurements": rows,
    }


def _build_report(args: argparse.Namespace) -> dict[str, object]:
    ledger = Ledger(args.ledger, include_rotated=True)
    specs = _candidate_specs(
        ledger, top_anchors=args.top_anchors, max_dynamic=args.max_dynamic
    )
    tier2_by_id = load_tier2_feedback(args.tier2)
    nas_by_id = score_specs_with_nas(specs, enabled=not args.disable_nas_screen)
    axes_counts = axes_counts_for_specs(specs)
    scores = score_specs_quality(
        specs,
        tier2_by_id=tier2_by_id,
        nas_by_id=nas_by_id,
        entries_by_id=ledger.entries,
        axes_counts=axes_counts,
    )
    ranked: list[QualityScore] = sorted(
        scores.values(), key=lambda s: s.quality_score, reverse=True
    )
    capability_diag = _capability_diagnostics(specs) if args.capability_screen else None
    passing = passing_only(ranked)
    report: dict[str, object] = {
        "version": _REPORT_VERSION,
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_specs": len(specs),
        "n_with_tier2_evidence": sum(1 for s in ranked if s.has_tier2_evidence),
        # The strict filter: how many are MEASURED to beat the baseline.
        "n_passing_hard_filter": len(passing),
        "verdict_counts": verdict_counts(ranked),
        "bucket_counts": bucket_counts(ranked),
        "research_priors": to_catalog_rows(),
        "passing_hard_filter": [s.to_json() for s in passing[: args.top]],
        "ranked": [s.to_json() for s in ranked[: args.top]],
    }
    if capability_diag is not None:
        report["capability_diagnostics"] = capability_diag
    if args.comparative_probe > 0:
        report["comparative_probe"] = _comparative_probe_top_n(
            ranked,
            {s.proposal_id: s for s in specs},
            top_n=args.comparative_probe,
            probe_steps=args.probe_steps,
        )
    if args.budget > 0:
        queue = allocate_budget_buckets(ranked, total=args.budget)
        report["budget_queue"] = {
            "total": args.budget,
            "bucket_counts": bucket_counts(queue),
            "proposal_ids": [s.proposal_id for s in queue],
        }
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    report = _build_report(args)
    payload = json.dumps(report, indent=2, default=str)
    if args.dry_run:
        print(payload)
        return 0
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output or (_REPORTS_DIR / f"proposal_quality_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")
    print(f"wrote: {out} ({report['n_specs']} specs ranked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
