"""Colab-safe component_fab seed runner that does not require research/meta_analysis.db.

The normal autonomous loop scopes anchors from ``research/meta_analysis.db``. That
DB is local-machine state and is usually absent in Colab. This runner uses a
small static anchor set, enumerates real buildable ProposalSpecs, grades a bounded
sample, and writes them to the ledger so follow-on surrogate/fidelity tooling has
real proposal ids instead of synthetic placeholders.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from component_fab.improver.ranking import composite_score
from component_fab.policies.promotion import (
    PromotionRules,
    apply_decisions,
    decide_promotions_for_ledger,
)
from component_fab.proposer.enumeration import enumerate_cycle_specs
from component_fab.state.ledger import DEFAULT_LEDGER_PATH, Ledger, write_json_report
from component_fab.validator.grade import eliminated_solo_scorecard, grade_candidate


_STATIC_ANCHORS: tuple[str, ...] = (
    "tropical_attention",
    "tropical_router",
    "tropical_gate",
    "clifford_attention",
    "padic_gate",
    "spike_rate_code",
    "grade_mix",
    "ultrametric_attention",
)


def _metadata_for_spec(spec, capability: dict[str, Any] | None, eliminated_by: str | None) -> dict[str, Any]:
    return {
        "track": "colab_static_seed",
        "synthetic": False,
        "colab_static_anchor_seed": True,
        "eliminated_by": eliminated_by,
        "math_axes": dict(spec.math_axes),
        "can_bind": bool(capability and capability.get("can_bind")),
        "erf_density": float((capability or {}).get("erf_density") or 0.0),
        "nb_max_accuracy": float((capability or {}).get("nb_max_accuracy") or 0.0),
        "range_effective_distance": int((capability or {}).get("range_effective_distance") or 0),
    }


def _grade_and_record(ledger: Ledger, spec, *, cycle: int, dim: int, seq_len: int, steps: int, skip_probe: bool) -> dict[str, Any]:
    bundle = grade_candidate(
        spec,
        dim=dim,
        seq_len=seq_len,
        n_steps=steps,
        run_range_probe=False,
        run_in_context=not skip_probe,
        persist_solo_scorecard=True,
    )
    if bundle.eliminated_by is not None:
        solo = eliminated_solo_scorecard(spec, bundle.eliminated_by)
        metadata = _metadata_for_spec(spec, bundle.capability, bundle.eliminated_by)
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
        return {
            "proposal_id": spec.proposal_id,
            "name": spec.name,
            "status": "eliminated",
            "eliminated_by": bundle.eliminated_by,
            "composite_score": 0.0,
        }

    assert bundle.solo is not None
    solo_dict = asdict(bundle.solo)
    probe_dict = asdict(bundle.in_context) if bundle.in_context is not None else None
    score, components = composite_score(solo_dict, probe_dict, bundle.capability)
    smoke = solo_dict.get("smoke") or {}
    metadata = _metadata_for_spec(spec, bundle.capability, None)
    ledger.record_grade(
        proposal_id=spec.proposal_id,
        name=spec.name,
        category=spec.category,
        synthesis_kind=spec.synthesis_kind,
        cycle=cycle,
        composite_score=float(score),
        smoke_pass=bool(smoke.get("forward_passed") and smoke.get("backward_passed")),
        learned_signal=bool(probe_dict and probe_dict.get("learned_signal")),
        metadata=metadata,
    )
    return {
        "proposal_id": spec.proposal_id,
        "name": spec.name,
        "status": "graded",
        "composite_score": float(score),
        "components": components,
        "can_bind": bool((bundle.capability or {}).get("can_bind")),
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colab-safe static-anchor component_fab seed run")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cycle", type=int, default=1)
    parser.add_argument("--max-specs", type=int, default=48)
    parser.add_argument("--max-graded", type=int, default=12)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--anchor", action="append", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    anchors = tuple(args.anchor) if args.anchor else _STATIC_ANCHORS
    ledger = Ledger(args.ledger, include_rotated=True)
    try:
        specs = enumerate_cycle_specs(
            ledger,
            list(anchors),
            cycle=args.cycle,
            dim=args.dim,
            use_promoted_as_anchors=False,
            include_static_variants=True,
            include_frontier=False,
            include_nas=False,
            include_ledger_specs=False,
            max_cross_pairs=max(0, args.max_specs // 2),
            max_knob_specs=max(0, args.max_specs),
            max_dynamic_specs=0,
            max_nas_specs=0,
        )[: max(0, args.max_specs)]
        active = [spec for spec in specs if not ledger.has_seen(spec.proposal_id)]
        selected = active[: max(0, args.max_graded)]
        results = [
            _grade_and_record(
                ledger,
                spec,
                cycle=args.cycle,
                dim=args.dim,
                seq_len=args.seq_len,
                steps=args.steps,
                skip_probe=args.skip_probe,
            )
            for spec in selected
        ]
        rules = PromotionRules(
            promote_min_streak_cycles=1,
            promote_min_composite=0.6,
            require_complete_promotion_evidence=False,
            require_ci_excludes_zero=False,
            reject_after_n_cycles=2,
            reject_max_composite=0.20,
        )
        promotion_counts = apply_decisions(ledger, decide_promotions_for_ledger(ledger, rules))
    finally:
        ledger.close()

    report = {
        "track": "colab_static_seed",
        "anchors": list(anchors),
        "n_specs_enumerated": len(specs),
        "n_new_specs": len(active),
        "n_graded": len(selected),
        "promotion_counts": promotion_counts,
        "results": results,
    }
    if args.output:
        write_json_report(report, args.output)
    if not args.quiet:
        print(
            f"static seed: enumerated={len(specs)} new={len(active)} graded={len(selected)} "
            f"promotions={promotion_counts}"
        )
        for row in results[:10]:
            print(f"  {row['status']:<10} score={row['composite_score']:.3f} {row['name'][:72]}")
        if args.output:
            print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
