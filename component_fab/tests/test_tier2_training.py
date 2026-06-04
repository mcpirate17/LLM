"""Tests for the Tier-2 predictor training-table accumulator."""

from __future__ import annotations

from pathlib import Path

from component_fab.state.tier2_training import (
    append_tier2_labels,
    arch_group,
    load_tier2_labels,
    tier2_label_row,
)


def _ok_result(delta: float = 0.05) -> dict:
    return {
        "status": "ok",
        "name": "cand",
        "math_axes": {
            "op_algebraic_space": "tropical",
            "op_block_template": "recursive_depth_router",
            "op_routing_kind": "depth_router",
        },
        "pass_count": 4,
        "n_tasks": 6,
        "tier2_passed": True,
        "per_task": {
            "long_gap_recall": {
                "delta": delta,
                "beats": True,
                "candidate_eval_acc": 0.3,
                "baseline_max": 0.25,
            },
            "compositional_binding": {
                "delta": -0.01,
                "beats": False,
                "candidate_eval_acc": 0.1,
                "baseline_max": 0.11,
            },
        },
    }


def test_label_row_skips_non_ok() -> None:
    assert (
        tier2_label_row(
            "p",
            {"status": "failed: boom"},
            baseline_names=("softmax_attention",),
            dim=32,
            n_blocks=2,
            n_train_steps=200,
            seed_count=1,
            timestamp="t",
        )
        is None
    )


def test_label_row_computes_mean_delta_and_provenance() -> None:
    row = tier2_label_row(
        "p1",
        _ok_result(0.05),
        baseline_names=("softmax_attention", "gpt2"),
        dim=32,
        n_blocks=2,
        n_train_steps=200,
        seed_count=2,
        timestamp="t",
    )
    assert row is not None
    assert abs(row["mean_delta"] - 0.02) < 1e-9  # mean(0.05, -0.01)
    assert row["arch_group"] == "tropical|recursive_depth_router|depth_router"
    assert row["baseline_names"] == ["softmax_attention", "gpt2"]
    assert row["tier2_passed"] is True


def test_append_and_load_roundtrip_dedup(tmp_path: Path) -> None:
    table = tmp_path / "labels.jsonl"
    n = append_tier2_labels(
        {"p1": _ok_result(0.05), "p2": _ok_result(0.02), "bad": {"status": "x"}},
        baseline_names=("softmax_attention",),
        dim=32,
        n_blocks=2,
        n_train_steps=200,
        seed_count=1,
        table_path=table,
    )
    assert n == 2  # only ok rows
    # re-run p1 with a different outcome → latest wins
    append_tier2_labels(
        {"p1": _ok_result(0.20)},
        baseline_names=("softmax_attention",),
        dim=32,
        n_blocks=2,
        n_train_steps=200,
        seed_count=1,
        table_path=table,
    )
    rows = load_tier2_labels(table)
    assert len(rows) == 2  # p1, p2 (dedup by id)
    p1 = next(r for r in rows if r["proposal_id"] == "p1")
    assert abs(p1["mean_delta"] - 0.095) < 1e-9  # mean(0.20, -0.01) from latest


def test_load_missing_table_is_empty(tmp_path: Path) -> None:
    assert load_tier2_labels(tmp_path / "nope.jsonl") == []


def test_arch_group_stable() -> None:
    a = {
        "op_algebraic_space": "tropical",
        "op_block_template": "x",
        "op_routing_kind": "y",
    }
    assert arch_group(a) == "tropical|x|y"
