"""Tests for axis_lift + failure_attribution analyzers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from component_fab.state.axis_lift import (
    compute_axis_lift,
    load_axis_lift,
    write_axis_lift,
)
from component_fab.state.failure_attribution import (
    CANONICAL_GATE_ORDER,
    compute_failure_attribution,
    write_failure_attribution,
)
from component_fab.tests.conftest import (
    grade_record,
    promote_record,
    write_ledger_jsonl,
)


# -------------------- axis_lift --------------------


def test_axis_lift_shrinks_toward_global_with_small_n(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    records = [
        grade_record("a", knobs=("good",), composite=0.6),
        promote_record("a", "promoted"),
        # five rejected proposals so global pass rate = 1/6 ≈ 0.167
        *(
            rec
            for i in range(5)
            for rec in (
                grade_record(f"r{i}", knobs=("other",), eliminated_by="nano_bind"),
                promote_record(f"r{i}", "rejected"),
            )
        ),
    ]
    write_ledger_jsonl(ledger, records)

    report = compute_axis_lift(ledger, prior_strength=5.0, min_n=1)
    assert report.global_total == 6
    assert report.global_promoted == 1
    knob_rows = {r.value: r for r in report.by_axis["math_knob"]}
    good = knob_rows["good"]
    other = knob_rows["other"]
    # raw = 1/1, but shrunk should be pulled toward 1/6 by the prior strength
    assert good.pass_rate_raw == pytest.approx(1.0)
    assert good.pass_rate_shrunk < 0.8
    assert good.pass_rate_shrunk > report.global_pass_rate
    # never-promoted knob shrinks above zero
    assert other.pass_rate_shrunk > 0


def test_axis_lift_pair_axis_present_only_for_multi_knob(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    records = [
        grade_record("a", knobs=("x", "y"), composite=0.5),
        promote_record("a", "promoted"),
        grade_record("b", knobs=("x",), eliminated_by="nano_bind"),
        promote_record("b", "rejected"),
    ]
    write_ledger_jsonl(ledger, records)
    report = compute_axis_lift(ledger, min_n=1)
    pair_values = {r.value for r in report.by_axis.get("math_knob_pair", [])}
    assert "x+y" in pair_values
    # singletons never reach the pair axis
    assert pair_values == {"x+y"}


def test_axis_lift_emits_math_sweep_variant_axes(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    sweep_grade = grade_record("sweep", composite=0.7, learned=True)
    sweep_grade["metadata"].update(
        {
            "math_sweep_passed": True,
            "math_variant_family": "algebraic+tropical",
            "math_variant_transform": "reciprocal_cauchy_read",
            "math_variant_target": "binding",
            "math_variant_stability_band": "marginal",
            "math_variant_score": 0.21,
            "math_variant_delta_long_range_reach": 0.12,
            "math_variant_delta_content_match_gating": 0.0,
        }
    )
    write_ledger_jsonl(ledger, [sweep_grade, promote_record("sweep", "promoted")])

    report = compute_axis_lift(ledger, min_n=1)

    assert {
        row.value for row in report.by_axis["math_variant_family"]
    } == {"algebraic+tropical"}
    assert {
        row.value for row in report.by_axis["math_variant_family_pair"]
    } == {"algebraic+tropical"}
    assert {
        row.value for row in report.by_axis["math_variant_transform"]
    } == {"reciprocal_cauchy_read"}
    assert {row.value for row in report.by_axis["math_variant_target"]} == {"binding"}
    assert {
        row.value for row in report.by_axis["math_variant_delta_long_range_reach"]
    } == {"up"}
    assert {
        row.value
        for row in report.by_axis["math_variant_delta_content_match_gating"]
    } == {"flat"}
    assert {
        row.value for row in report.by_axis["math_variant_target_improved"]
    } == {"true"}


def test_axis_lift_round_trip_json(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    out = tmp_path / "axis_lift.json"
    write_ledger_jsonl(
        ledger, [grade_record("a", knobs=("k",)), promote_record("a", "promoted")]
    )
    report = compute_axis_lift(ledger, min_n=1)
    write_axis_lift(report, output_path=out)
    loaded = load_axis_lift(out)
    assert loaded is not None
    assert loaded.global_promoted == 1
    assert loaded.by_axis["math_knob"][0].value == "k"


def test_axis_lift_empty_ledger_safe(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    report = compute_axis_lift(ledger)
    assert report.global_total == 0
    assert report.global_pass_rate == 0.0
    assert report.by_axis == {}


# -------------------- failure_attribution --------------------


def test_failure_attribution_flags_over_eager_gate(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    records: list[dict] = []
    # 30 ERF kills, 5 nano_bind kills, 5 survivors -> ERF rate = 30/40 = 75%
    # bump above 0.7 by using a custom threshold below.
    for i in range(30):
        records.append(grade_record(f"e{i}", eliminated_by="erf_density"))
        records.append(promote_record(f"e{i}", "rejected"))
    for i in range(5):
        records.append(grade_record(f"nb{i}", eliminated_by="nano_bind"))
        records.append(promote_record(f"nb{i}", "rejected"))
    for i in range(5):
        records.append(grade_record(f"s{i}", composite=0.7, learned=True))
        records.append(promote_record(f"s{i}", "promoted"))
    write_ledger_jsonl(ledger, records)

    report = compute_failure_attribution(
        ledger, over_eager_threshold=0.7, min_n_for_over_eager=10
    )
    gates = {g.gate: g for g in report.gate_stats}
    assert gates["erf_density"].killed == 30
    # ERF reached count = 40 (all proposals — smoke killed nobody)
    assert gates["erf_density"].reached == 40
    assert gates["erf_density"].kill_rate == pytest.approx(0.75)
    assert gates["erf_density"].over_eager
    assert "erf_density" in report.over_eager_gates


def test_failure_attribution_anchor_pool_filters_and_ranks(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    records = [
        # high composite, late-gate kill -> anchor candidate
        grade_record("good_late", eliminated_by="nano_bind", composite=0.6, erf=0.15),
        promote_record("good_late", "rejected"),
        # high ERF, late-gate kill -> anchor candidate
        grade_record("erf_late", eliminated_by="nano_bind", composite=0.1, erf=0.25),
        promote_record("erf_late", "rejected"),
        # low composite, low ERF -> excluded
        grade_record("noise", eliminated_by="nano_bind", composite=0.0, erf=0.01),
        promote_record("noise", "rejected"),
        # promoted -> excluded
        grade_record("winner", composite=0.9, erf=0.4),
        promote_record("winner", "promoted"),
    ]
    write_ledger_jsonl(ledger, records)
    report = compute_failure_attribution(
        ledger, anchor_min_composite=0.5, anchor_min_erf=0.2
    )
    pool_ids = [c.proposal_id for c in report.anchor_pool]
    assert pool_ids == ["good_late", "erf_late"]
    # composite-first ordering
    assert (
        report.anchor_pool[0].composite_score >= report.anchor_pool[1].composite_score
    )


def test_failure_attribution_canonical_gate_order_reached_counts(
    tmp_path: Path,
) -> None:
    """Reach counts must respect canonical gate order: later gates see fewer."""
    ledger = tmp_path / "ledger.jsonl"
    records: list[dict] = []
    for i in range(10):
        records.append(grade_record(f"e{i}", eliminated_by="erf_density"))
        records.append(promote_record(f"e{i}", "rejected"))
    for i in range(2):
        records.append(grade_record(f"nb{i}", eliminated_by="nano_bind"))
        records.append(promote_record(f"nb{i}", "rejected"))
    write_ledger_jsonl(ledger, records)
    report = compute_failure_attribution(ledger)
    gates = {g.gate: g for g in report.gate_stats}
    assert gates["erf_density"].reached == 12
    # nano_bind reach drops by the 10 ERF kills
    assert gates["nano_bind"].reached == 2
    # late gates see zero — every proposal died at ERF or nano_bind
    assert gates["ar_easy"].reached == 0
    assert gates["ar_easy"].killed == 0


def test_failure_attribution_writes_json(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    out = tmp_path / "failure.json"
    write_ledger_jsonl(
        ledger,
        [
            grade_record("a", eliminated_by="erf_density", erf=0.05),
            promote_record("a", "rejected"),
        ],
    )
    report = compute_failure_attribution(ledger)
    write_failure_attribution(report, output_path=out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["total_graded"] == 1
    assert data["total_rejected"] == 1


def test_failure_attribution_counts_math_sweep_failures(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    rank = grade_record("rank", eliminated_by="nano_bind")
    rank["metadata"].update(
        {
            "math_sweep_passed": False,
            "math_variant_failure_reason": "rank_collapse",
        }
    )
    shape = grade_record("shape", eliminated_by="smoke")
    shape["metadata"].update(
        {
            "math_sweep_passed": False,
            "math_variant_failure_reason": "compile_failed",
        }
    )
    write_ledger_jsonl(
        ledger,
        [
            rank,
            promote_record("rank", "rejected"),
            shape,
            promote_record("shape", "rejected"),
        ],
    )

    report = compute_failure_attribution(ledger)

    assert report.math_sweep_failures == {
        "compile_failed": 1,
        "rank_collapse": 1,
    }


def test_failure_attribution_floor_bunching_unflags_correct_erf_gate(
    tmp_path: Path,
) -> None:
    """When ERF kills cluster at the 1/seq_len structural floor the gate is
    correct — generator is producing per-position-only blocks. Don't flag
    over_eager; do raise generator_floor_bunched on the ERF row."""
    ledger = tmp_path / "ledger.jsonl"
    records: list[dict] = []
    # 30 ERF kills all at the floor (1/32 ≈ 0.03125).
    for i in range(30):
        records.append(grade_record(f"e{i}", eliminated_by="erf_density", erf=1.0 / 32))
        records.append(promote_record(f"e{i}", "rejected"))
    # 10 survivors so kill_rate stays above the over_eager threshold.
    for i in range(10):
        records.append(grade_record(f"s{i}", composite=0.7, learned=True))
        records.append(promote_record(f"s{i}", "promoted"))
    write_ledger_jsonl(ledger, records)
    report = compute_failure_attribution(
        ledger, over_eager_threshold=0.7, min_n_for_over_eager=10
    )
    gates = {g.gate: g for g in report.gate_stats}
    erf = gates["erf_density"]
    assert erf.kill_rate >= 0.7  # would have been flagged
    assert not erf.over_eager  # but cleared because kills are at the floor
    assert erf.generator_floor_bunched
    assert erf.killed_at_floor == 30
    assert "erf_density" not in report.over_eager_gates


def test_failure_attribution_canonical_order_constant_matches_validator() -> None:
    """Order must include the gates the validator actually emits."""
    expected = {
        "smoke",
        "s05_causality_stability",
        "erf_density",
        "nano_bind",
        "ar_easy",
        "ar_medium",
    }
    assert expected.issubset(set(CANONICAL_GATE_ORDER))


# -------------------- proposer wiring --------------------


def test_score_knob_stack_uses_axis_lift_when_provided() -> None:
    """High-lift knobs must outscore low-lift knobs holding ledger flat."""
    from component_fab.improver.math_knob_catalog import score_knob_stack
    from component_fab.state.axis_lift import AxisLiftReport, ValueStats

    high = AxisLiftReport(
        global_promoted=1,
        global_total=10,
        global_pass_rate=0.1,
        prior_strength=5.0,
        min_n=1,
        by_axis={
            "math_knob": [
                ValueStats(value="winner", n=10, k_promoted=5, lift=5.0),
                ValueStats(value="loser", n=10, k_promoted=0, lift=0.2),
            ]
        },
    )
    score_winner = score_knob_stack(("winner",), ledger=None, axis_lift=high)
    score_loser = score_knob_stack(("loser",), ledger=None, axis_lift=high)
    score_none = score_knob_stack(("winner",), ledger=None, axis_lift=None)
    assert score_winner.score > score_loser.score
    assert score_winner.score > score_none.score
    assert "axis_lift" in score_winner.reason
