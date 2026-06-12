"""Tests for the state-layer consolidation: gates.py, ledger helpers, the
shared spearman helper, and the ONE Tier-2 row parser feeding all three
consumers (trust evidence, training labels, proposer feedback)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from component_fab.state.gates import (
    CANONICAL_GATE_ORDER,
    GATE_ERF_DENSITY,
    GATE_NANO_BIND,
    GATE_SMOKE,
    SURVIVED,
    eliminated_by,
    gate_index,
    passed,
    reached,
)
from component_fab.state.ledger import (
    LedgerEntry,
    iter_jsonl_records,
    latest_by_key,
    write_json_report,
)
from component_fab.state.tier2_training import parse_tier2_row, tier2_label_row


# --------------------------------------------------------------------------- #
# gates.py — canonical chained-gate semantics
# --------------------------------------------------------------------------- #
def test_gate_index_canonical_and_unknown():
    assert gate_index(GATE_SMOKE) == 0
    assert gate_index(GATE_NANO_BIND) == 3
    assert gate_index("never_heard_of_it") == len(CANONICAL_GATE_ORDER)


def test_eliminated_by_resolution_order():
    assert eliminated_by({"metadata": {"eliminated_by": GATE_ERF_DENSITY}}) == (
        GATE_ERF_DENSITY
    )
    assert eliminated_by({"smoke_pass": False, "metadata": {}}) == GATE_SMOKE
    assert eliminated_by({"smoke_pass": True, "metadata": {}}) == SURVIVED


def test_reached_and_passed_chained_semantics():
    # Killed at nano_bind: reached every gate up to and including nano_bind,
    # passed everything strictly before it, and nothing at/after it.
    assert reached(GATE_NANO_BIND, GATE_ERF_DENSITY)
    assert passed(GATE_NANO_BIND, GATE_ERF_DENSITY)
    assert reached(GATE_NANO_BIND, GATE_NANO_BIND)
    assert not passed(GATE_NANO_BIND, GATE_NANO_BIND)
    assert not reached(GATE_NANO_BIND, "ar_easy")
    # Survivors reach and pass everything.
    assert reached(SURVIVED, "ar_hard")
    assert passed(SURVIVED, "ar_hard")


# --------------------------------------------------------------------------- #
# ledger helpers
# --------------------------------------------------------------------------- #
def test_latest_by_key_last_record_wins_and_skips_missing():
    records = [
        {"proposal_id": "a", "v": 1},
        {"no_key_here": True},
        {"proposal_id": "b", "v": 2},
        {"proposal_id": "a", "v": 3},
    ]
    latest = latest_by_key(records, "proposal_id")
    assert latest["a"]["v"] == 3
    assert latest["b"]["v"] == 2
    assert set(latest) == {"a", "b"}


def test_iter_jsonl_records_logs_corrupt_line_count(tmp_path: Path, caplog):
    path = tmp_path / "log.jsonl"
    path.write_text('{"ok": 1}\nnot json\n{"ok": 2}\n{broken\n', encoding="utf-8")
    with caplog.at_level(logging.DEBUG, logger="component_fab.state.ledger"):
        records = list(iter_jsonl_records(path))
    assert len(records) == 2
    assert any("skipped 2 corrupt lines" in m for m in caplog.messages)


def test_ledger_entry_composite_helpers():
    entry = LedgerEntry(
        proposal_id="p", name="p", category="lane", synthesis_kind="novel_hybrid"
    )
    assert entry.best_composite() == 0.0
    assert entry.mean_composite() == 0.0
    entry.composite_history.extend([0.2, 0.8, 0.5])
    assert entry.best_composite() == 0.8
    assert entry.mean_composite() == pytest.approx(0.5)
    assert entry.mean_composite(window=2) == pytest.approx(0.65)
    assert entry.mean_composite(window=10) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        entry.mean_composite(window=0)


def test_write_json_report_creates_parents_and_stable_format(tmp_path: Path):
    out = write_json_report({"b": 1, "a": [2, 3]}, tmp_path / "deep" / "r.json")
    text = out.read_text(encoding="utf-8")
    assert json.loads(text) == {"a": [2, 3], "b": 1}
    assert text.index('"a"') < text.index('"b"')  # sort_keys
    assert "\n  " in text  # indent=2


# --------------------------------------------------------------------------- #
# ONE Tier-2 row parser through all three consumers
# --------------------------------------------------------------------------- #
_ROW = {
    "status": "ok",
    "name": "cand_x",
    "math_axes": {
        "op_algebraic_space": "tropical",
        "op_block_template": "b",
        "op_routing_kind": "r",
    },
    "pass_count": 2,
    "n_tasks": 4,
    "tier2_passed": False,
    "tier2_passed_niche": False,
    "seed_count": 2,
    "per_task": {
        "distractor_kv_recall": {
            "candidate_eval_acc": 0.2,
            "baseline_max": 0.1,
            "delta": 0.1,
            "beats": True,
        },
        "long_gap_recall": {
            "candidate_eval_acc": 0.3,
            "baseline_max": 0.25,
            "delta": 0.05,
            "beats": True,
        },
        "compositional_binding": {
            "candidate_eval_acc": 0.1,
            "baseline_max": 0.2,
            "delta": -0.1,
            "beats": False,
        },
        "multi_query_kv_recall": {
            "candidate_eval_acc": 0.1,
            "baseline_max": 0.3,
            "delta": -0.2,
            "beats": False,
        },
    },
}


def test_parse_tier2_row_fields():
    m = parse_tier2_row(_ROW)
    assert m.ok and m.status == "ok"
    assert m.mean_delta == pytest.approx(-0.0375)
    assert m.min_delta == pytest.approx(-0.2)
    assert m.wins == ("distractor_kv_recall", "long_gap_recall")
    assert m.failures == ("compositional_binding", "multi_query_kv_recall")
    assert m.pass_count == 2 and m.n_tasks == 4 and m.seed_count == 2
    assert [r.task for r in m.task_results] == list(_ROW["per_task"])  # row order
    assert not parse_tier2_row({"status": "failed: boom"}).ok


def test_tier2_row_through_training_consumer():
    row = tier2_label_row(
        "p",
        _ROW,
        baseline_names=("softmax_attention",),
        dim=32,
        n_blocks=2,
        n_train_steps=200,
        seed_count=1,
        timestamp="t",
    )
    assert row is not None
    assert row["mean_delta"] == pytest.approx(-0.0375)
    assert row["min_delta"] == pytest.approx(-0.2)
    assert row["arch_group"] == "tropical|b|r"
    assert list(row["per_task"]) == list(_ROW["per_task"])  # JSONL key order kept
    assert row["per_task"]["long_gap_recall"] == {
        "delta": 0.05,
        "beats": True,
        "candidate_eval_acc": 0.3,
        "baseline_max": 0.25,
    }


def test_tier2_row_through_trust_consumer():
    # Lazy import: validator/__init__ pulls the harness package, which a
    # sibling refactor may have mid-flight; only this test should depend on it.
    from component_fab.validator.trust import tier2_evidence_from_summary

    ev = tier2_evidence_from_summary("p", {"results": {"p": _ROW}, "seed_count": 3})
    assert ev.present and ev.status == "ok"
    assert ev.pass_count == 2 and ev.n_tasks == 4
    assert ev.mean_delta == pytest.approx(-0.0375)
    assert ev.min_delta == pytest.approx(-0.2)
    assert not ev.niche_passed  # compositional_binding lost
    assert not ev.passed
    assert ev.seed_count == 2  # row-level beats the summary-level fallback


def test_tier2_row_through_feedback_consumer():
    from component_fab.proposer.tier2_feedback import (
        WEAK_FAIL_BROAD_KV,
        WEAK_FAIL_COMPOSITIONAL,
        WEAK_NEAR_SURVIVOR,
        WEAK_REJECTED,
        feedback_from_result,
    )

    fb = feedback_from_result("p", _ROW)
    assert fb is not None
    assert fb.name == "cand_x"
    assert fb.pass_count == 2 and fb.n_tasks == 4
    assert fb.mean_delta == pytest.approx(-0.0375)
    assert fb.wins == ("distractor_kv_recall", "long_gap_recall")
    assert fb.failures == ("compositional_binding", "multi_query_kv_recall")
    assert [r.task for r in fb.task_results] == sorted(_ROW["per_task"])  # sorted
    assert set(fb.signatures) == {
        WEAK_FAIL_COMPOSITIONAL,
        WEAK_FAIL_BROAD_KV,
        WEAK_NEAR_SURVIVOR,
        WEAK_REJECTED,
    }
    assert feedback_from_result("p", {"status": "failed: boom"}) is None
