"""Tests for the gate_calibration analyzer (WS-1)."""

from __future__ import annotations

import json

import pytest

from component_fab.state.gate_calibration import (
    GateCalibrationReport,
    _auc,
    compute_gate_calibration,
    write_gate_calibration,
)
from component_fab.tests.conftest import grade_record, write_ledger_jsonl


# --------------------------------------------------------------------------- #
# AUC primitive
# --------------------------------------------------------------------------- #
def test_auc_perfect_separation():
    # positives all score higher than negatives -> AUC 1.0
    assert _auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0


def test_auc_perfect_inversion():
    # positives all score lower than negatives -> AUC 0.0 (anti-predictive)
    assert _auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0


def test_auc_ties_half():
    # all identical scores -> chance
    assert _auc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5


def test_auc_single_class_none():
    assert _auc([0.1, 0.2], [1, 1]) is None


# --------------------------------------------------------------------------- #
# End-to-end on a synthetic ledger
# --------------------------------------------------------------------------- #
def _separating_ledger(n: int = 40) -> list[dict]:
    """erf_density and nb_max_accuracy cleanly separate learned vs not."""
    records: list[dict] = []
    for i in range(n):
        good = i % 2 == 0
        records.append(
            grade_record(
                f"p{i}",
                eliminated_by=None,  # all survive so they reach every gate
                erf=0.30 if good else 0.06,
                nb=0.70 if good else 0.20,
                can_bind=good,
                learned=good,
                composite=0.6 if good else 0.2,
            )
        )
    return records


def test_separating_signals_report_high_auc(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    write_ledger_jsonl(ledger, _separating_ledger())
    report = compute_gate_calibration(
        ledger_path=ledger,
        db_path=None,
        run_buildability_survey=False,
        min_class_n=5,
    )
    assert isinstance(report, GateCalibrationReport)
    aucs = {g.signal: g for g in report.gate_aucs["learned_signal"]}
    assert aucs["erf_density"].auc_reached == pytest.approx(1.0)
    assert aucs["nb_max_accuracy"].auc_reached == pytest.approx(1.0)
    assert aucs["erf_density"].verdict == "predictive"
    assert any("calibrated OK" in f for f in report.findings)


def test_anti_predictive_signal_flagged(tmp_path):
    # erf_density is INVERTED vs the label -> must be flagged anti_predictive.
    records: list[dict] = []
    for i in range(40):
        good = i % 2 == 0
        records.append(
            grade_record(
                f"p{i}",
                eliminated_by=None,
                erf=0.06 if good else 0.30,  # inverted
                nb=0.50,
                can_bind=good,
                learned=good,
            )
        )
    ledger = tmp_path / "ledger.jsonl"
    write_ledger_jsonl(ledger, records)
    report = compute_gate_calibration(
        ledger_path=ledger, db_path=None, run_buildability_survey=False, min_class_n=5
    )
    erf = next(
        g for g in report.gate_aucs["learned_signal"] if g.signal == "erf_density"
    )
    assert erf.verdict == "anti_predictive"
    assert any("ANTI-PREDICTIVE" in f and "erf_density" in f for f in report.findings)


def test_reachability_excludes_earlier_killed(tmp_path):
    # A candidate killed at s05 never reaches erf -> excluded from erf AUC.
    records = [
        grade_record(
            "survivor", eliminated_by=None, erf=0.3, nb=0.7, can_bind=True, learned=True
        ),
        grade_record(
            "killed_s05", eliminated_by="s05_causality_stability", learned=False
        ),
        grade_record(
            "killed_erf", eliminated_by="erf_density", erf=0.04, nb=0.0, learned=False
        ),
    ]
    ledger = tmp_path / "ledger.jsonl"
    write_ledger_jsonl(ledger, records)
    report = compute_gate_calibration(
        ledger_path=ledger, db_path=None, run_buildability_survey=False, min_class_n=1
    )
    erf = next(
        g for g in report.gate_aucs["learned_signal"] if g.signal == "erf_density"
    )
    # survivor + killed_erf reached erf (2); killed_s05 did not.
    assert erf.n_reached == 2
    # only survivor passed erf.
    assert erf.n_passed == 1


def test_threshold_sweep_present_and_monotone_kept(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    write_ledger_jsonl(ledger, _separating_ledger())
    report = compute_gate_calibration(
        ledger_path=ledger, db_path=None, run_buildability_survey=False, min_class_n=5
    )
    sweep = report.threshold_sweeps["erf_density"]
    assert sweep
    kept = [p.kept_frac for p in sweep]
    # higher threshold keeps no more than lower threshold
    assert all(kept[i] >= kept[i + 1] for i in range(len(kept) - 1))


def test_write_roundtrip(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    write_ledger_jsonl(ledger, _separating_ledger())
    report = compute_gate_calibration(
        ledger_path=ledger, db_path=None, run_buildability_survey=False, min_class_n=5
    )
    out = write_gate_calibration(report, output_path=tmp_path / "gate_calibration.json")
    data = json.loads(out.read_text())
    assert data["total_graded"] == 40
    assert "gate_aucs" in data and "learned_signal" in data["gate_aucs"]
    assert "findings" in data and data["findings"]


def test_empty_ledger_no_crash(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("")
    report = compute_gate_calibration(
        ledger_path=ledger, db_path=None, run_buildability_survey=False
    )
    assert report.total_graded == 0
    # every gate AUC is None / insufficient_n, no exception
    for g in report.gate_aucs["learned_signal"]:
        assert g.verdict == "insufficient_n"


def test_bad_primary_label_raises(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    write_ledger_jsonl(ledger, _separating_ledger())
    with pytest.raises(ValueError):
        compute_gate_calibration(
            ledger_path=ledger,
            db_path=None,
            primary_label="nonsense",
            run_buildability_survey=False,
        )
