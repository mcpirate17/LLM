"""Tests for axis_lift + failure_attribution analyzers."""

from __future__ import annotations

import datetime as _dt
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
    load_failure_attribution,
    write_failure_attribution,
)


def _grade(
    pid: str,
    *,
    cycle: int = 1,
    knobs: tuple[str, ...] = (),
    eliminated_by: str | None = None,
    composite: float = 0.0,
    learned: bool = False,
    erf: float | None = None,
    nb: float | None = None,
    synthesis_kind: str = "semiring_swap",
    category: str = "lane",
    name: str | None = None,
) -> dict:
    meta: dict = {"math_knobs": list(knobs)}
    if eliminated_by is not None:
        meta["eliminated_by"] = eliminated_by
    if erf is not None:
        meta["erf_density"] = erf
    if nb is not None:
        meta["nb_max_accuracy"] = nb
    return {
        "event": "grade",
        "proposal_id": pid,
        "name": name or pid,
        "category": category,
        "synthesis_kind": synthesis_kind,
        "cycle": cycle,
        "composite_score": composite,
        "smoke_pass": eliminated_by != "smoke",
        "learned_signal": learned,
        "metadata": meta,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def _promote(pid: str, status: str) -> dict:
    return {
        "event": "promote",
        "proposal_id": pid,
        "status": status,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def _write_ledger(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps(r) + "\n")


# -------------------- axis_lift --------------------


def test_axis_lift_shrinks_toward_global_with_small_n(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    records = [
        _grade("a", knobs=("good",), composite=0.6),
        _promote("a", "promoted"),
        # five rejected proposals so global pass rate = 1/6 ≈ 0.167
        *(
            rec
            for i in range(5)
            for rec in (
                _grade(f"r{i}", knobs=("other",), eliminated_by="nano_bind"),
                _promote(f"r{i}", "rejected"),
            )
        ),
    ]
    _write_ledger(ledger, records)

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
        _grade("a", knobs=("x", "y"), composite=0.5),
        _promote("a", "promoted"),
        _grade("b", knobs=("x",), eliminated_by="nano_bind"),
        _promote("b", "rejected"),
    ]
    _write_ledger(ledger, records)
    report = compute_axis_lift(ledger, min_n=1)
    pair_values = {r.value for r in report.by_axis.get("math_knob_pair", [])}
    assert "x+y" in pair_values
    # singletons never reach the pair axis
    assert pair_values == {"x+y"}


def test_axis_lift_round_trip_json(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    out = tmp_path / "axis_lift.json"
    _write_ledger(ledger, [_grade("a", knobs=("k",)), _promote("a", "promoted")])
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
        records.append(_grade(f"e{i}", eliminated_by="erf_density"))
        records.append(_promote(f"e{i}", "rejected"))
    for i in range(5):
        records.append(_grade(f"nb{i}", eliminated_by="nano_bind"))
        records.append(_promote(f"nb{i}", "rejected"))
    for i in range(5):
        records.append(_grade(f"s{i}", composite=0.7, learned=True))
        records.append(_promote(f"s{i}", "promoted"))
    _write_ledger(ledger, records)

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
        _grade("good_late", eliminated_by="nano_bind", composite=0.6, erf=0.15),
        _promote("good_late", "rejected"),
        # high ERF, late-gate kill -> anchor candidate
        _grade("erf_late", eliminated_by="nano_bind", composite=0.1, erf=0.25),
        _promote("erf_late", "rejected"),
        # low composite, low ERF -> excluded
        _grade("noise", eliminated_by="nano_bind", composite=0.0, erf=0.01),
        _promote("noise", "rejected"),
        # promoted -> excluded
        _grade("winner", composite=0.9, erf=0.4),
        _promote("winner", "promoted"),
    ]
    _write_ledger(ledger, records)
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
        records.append(_grade(f"e{i}", eliminated_by="erf_density"))
        records.append(_promote(f"e{i}", "rejected"))
    for i in range(2):
        records.append(_grade(f"nb{i}", eliminated_by="nano_bind"))
        records.append(_promote(f"nb{i}", "rejected"))
    _write_ledger(ledger, records)
    report = compute_failure_attribution(ledger)
    gates = {g.gate: g for g in report.gate_stats}
    assert gates["erf_density"].reached == 12
    # nano_bind reach drops by the 10 ERF kills
    assert gates["nano_bind"].reached == 2
    # late gates see zero — every proposal died at ERF or nano_bind
    assert gates["ar_easy"].reached == 0
    assert gates["ar_easy"].killed == 0


def test_failure_attribution_round_trip_json(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    out = tmp_path / "failure.json"
    _write_ledger(
        ledger,
        [
            _grade("a", eliminated_by="erf_density", erf=0.05),
            _promote("a", "rejected"),
        ],
    )
    report = compute_failure_attribution(ledger)
    write_failure_attribution(report, output_path=out)
    loaded = load_failure_attribution(out)
    assert loaded is not None
    assert loaded.total_graded == 1
    assert loaded.total_rejected == 1


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
        records.append(_grade(f"e{i}", eliminated_by="erf_density", erf=1.0 / 32))
        records.append(_promote(f"e{i}", "rejected"))
    # 10 survivors so kill_rate stays above the over_eager threshold.
    for i in range(10):
        records.append(_grade(f"s{i}", composite=0.7, learned=True))
        records.append(_promote(f"s{i}", "promoted"))
    _write_ledger(ledger, records)
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
