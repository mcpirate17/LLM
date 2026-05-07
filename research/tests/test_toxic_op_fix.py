"""Tests for toxic op classification fix: S1-only failure tracking + op rehabilitation."""

import json
import time
import pytest
from research.scientist.notebook import LabNotebook

pytestmark = pytest.mark.unit


def _make_graph_json(ops):
    """Build a minimal graph JSON with the given op-pair bigrams.

    Creates nodes: input -> op1 -> op2 -> ... with the given op names.
    Matches the format expected by _extract_op_bigrams: {"nodes": {id: {op_name, input_ids}}}.
    """
    nodes = {"0": {"op_name": "input", "input_ids": []}}
    for i, op in enumerate(ops, start=1):
        nodes[str(i)] = {"op_name": op, "input_ids": [str(i - 1)]}
    return json.dumps({"nodes": nodes})


@pytest.fixture
def nb(tmp_path):
    """Fresh LabNotebook for each test."""
    db_path = str(tmp_path / "test_toxic.db")
    notebook = LabNotebook(db_path)
    yield notebook
    notebook.close()


def test_failure_signatures_only_count_s1_failures(nb):
    """S0.5 failures (stage0=1, stage05=0) must NOT count as failure signatures."""
    exp_id = nb.start_experiment(
        experiment_type="test",
        config={"dim": 64},
        hypothesis="test",
    )

    graph_json = _make_graph_json(["gelu", "linear_proj"])

    # Program that failed at S0.5 (causality gate) — should NOT contaminate
    # Needs error_type to pass the quality gate in record_program_result
    nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="s05_fail_1",
        graph_json=graph_json,
        stage0_passed=True,
        stage05_passed=False,
        stage1_passed=False,
        error_type="causality_violation",
        loss_ratio=0.99,
    )
    # Another S0.5 failure
    nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="s05_fail_2",
        graph_json=graph_json,
        stage0_passed=True,
        stage05_passed=False,
        stage1_passed=False,
        error_type="causality_violation",
        loss_ratio=0.99,
    )

    nb.flush_writes()
    nb.update_failure_signatures(exp_id)

    # No failure signatures should exist — S0.5 failures are excluded
    count = nb.conn.execute("SELECT COUNT(*) FROM failure_signatures").fetchone()[0]
    assert count == 0, (
        f"Expected 0 failure signatures, got {count} (S0.5 failures contaminated)"
    )


def test_failure_signatures_count_genuine_s1_failures(nb):
    """Programs that passed S0+S0.5 but failed S1 SHOULD count as failures."""
    exp_id = nb.start_experiment(
        experiment_type="test",
        config={"dim": 64},
        hypothesis="test",
    )

    graph_json = _make_graph_json(["gelu", "linear_proj"])

    # Program that passed S0+S0.5 but failed S1 — genuine learning failure
    nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="s1_fail",
        graph_json=graph_json,
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=False,
        error_type="loss_diverged",
        loss_ratio=0.99,
    )
    # Program that passed everything — success
    nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="s1_pass",
        graph_json=graph_json,
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.5,
        trust_label="test_fixture",
    )

    nb.flush_writes()
    nb.update_failure_signatures(exp_id)

    # Should have signatures with both failure and success counts
    rows = nb.conn.execute(
        "SELECT signature, n_failures, n_successes FROM failure_signatures"
    ).fetchall()
    assert len(rows) > 0, "Expected failure signatures for genuine S1 failures"

    # The bigram gelu->linear_proj should have 1 failure + 1 success
    sigs = {r[0]: (r[1], r[2]) for r in rows}
    bg = "gelu->linear_proj"
    assert bg in sigs, f"Expected bigram '{bg}' in signatures, got {list(sigs.keys())}"
    assert sigs[bg][0] == 1, f"Expected 1 failure for '{bg}', got {sigs[bg][0]}"
    assert sigs[bg][1] == 1, f"Expected 1 success for '{bg}', got {sigs[bg][1]}"


def test_recompute_failure_signatures(nb):
    """recompute_failure_signatures should clear and rebuild with S1-only filter."""
    exp_id = nb.start_experiment(
        experiment_type="test",
        config={"dim": 64},
        hypothesis="test",
    )

    graph_json = _make_graph_json(["gelu", "linear_proj"])

    # Insert a mix of S0.5 failures and genuine S1 failures
    nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="s05_fail",
        graph_json=graph_json,
        stage0_passed=True,
        stage05_passed=False,
        stage1_passed=False,
        error_type="causality_violation",
        loss_ratio=0.99,
    )
    nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="s1_fail",
        graph_json=graph_json,
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=False,
        error_type="loss_diverged",
        loss_ratio=0.99,
    )

    nb.flush_writes()

    # Manually insert a contaminated signature (as old code would have)
    nb.conn.execute(
        """INSERT INTO failure_signatures
           (signature, n_failures, n_successes, last_updated)
           VALUES ('gelu->linear_proj', 100, 0, ?)""",
        (time.time(),),
    )
    nb.conn.commit()

    # Verify contaminated data exists
    before = nb.conn.execute(
        "SELECT n_failures FROM failure_signatures WHERE signature='gelu->linear_proj'"
    ).fetchone()
    assert before[0] == 100

    # Recompute should clean it up
    count = nb.recompute_failure_signatures()
    assert count > 0

    # Now should only reflect the genuine S1 failure (1 failure, 0 successes)
    after = nb.conn.execute(
        "SELECT n_failures, n_successes FROM failure_signatures WHERE signature='gelu->linear_proj'"
    ).fetchone()
    assert after is not None, "Signature should still exist from genuine S1 failure"
    assert after[0] == 1, f"Expected 1 failure after recompute, got {after[0]}"
    assert after[1] == 0, f"Expected 0 successes after recompute, got {after[1]}"


def test_failure_signatures_skip_malformed_graph_json(nb):
    """Malformed graph_json rows must not crash failure signature refresh."""
    exp_id = nb.start_experiment(
        experiment_type="test",
        config={"dim": 64},
        hypothesis="test",
    )

    valid_graph_json = _make_graph_json(["gelu", "linear_proj"])
    nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="valid_s1_fail",
        graph_json=valid_graph_json,
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=False,
        error_type="loss_diverged",
        loss_ratio=0.99,
    )
    nb.flush_writes()

    nb.conn.execute(
        """INSERT INTO program_results
           (result_id, experiment_id, graph_fingerprint, graph_json,
            stage0_passed, stage05_passed, stage1_passed, error_type, loss_ratio, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "bad_json_row",
            exp_id,
            "bad_json_fp",
            '{"nodes": {"0": {"op_name": "input"},',
            1,
            1,
            0,
            "RuntimeError",
            0.99,
            time.time(),
        ),
    )
    nb.conn.commit()

    nb.update_failure_signatures(exp_id)

    row = nb.conn.execute(
        "SELECT n_failures, n_successes FROM failure_signatures "
        "WHERE signature = 'gelu->linear_proj'"
    ).fetchone()
    assert row is not None
    assert row[0] == 1
    assert row[1] == 0


def test_failure_signatures_skip_persistently_suppressed_pairs(nb):
    """Audited false-positive pairs should be blocked at write time."""
    exp_id = nb.start_experiment(
        experiment_type="test",
        config={"dim": 64},
        hypothesis="suppressed pair",
    )

    graph_json = _make_graph_json(["rwkv_channel", "rmsnorm"])
    nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="suppressed_pair_fail",
        graph_json=graph_json,
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=False,
        error_type="insufficient_learning",
        loss_ratio=0.99,
    )
    nb.flush_writes()
    nb.update_failure_signatures(exp_id)

    row = nb.conn.execute(
        "SELECT n_failures, n_successes FROM failure_signatures "
        "WHERE signature='rwkv_channel->rmsnorm'"
    ).fetchone()
    assert row is None

    suppression = nb.conn.execute(
        "SELECT source, active FROM failure_signature_suppressions "
        "WHERE signature='rwkv_channel->rmsnorm'"
    ).fetchone()
    assert suppression is not None
    assert suppression[0] == "audit"
    assert suppression[1] == 1


def test_op_rehabilitation_basic(nb):
    """test_op_in_isolation should compile and forward for a known-good op."""
    from research.eval.op_rehab import test_op_in_isolation

    # linear_proj has learnable parameters so gradient flow works
    result = test_op_in_isolation("linear_proj", model_dim=64, device="cpu")
    assert result["compile_passed"], (
        f"linear_proj should compile: {result['error_message']}"
    )
    assert result["forward_passed"], (
        f"linear_proj should forward: {result['error_message']}"
    )


def test_rehabilitation_prevents_exclusion(nb):
    """An op with 0% S1 rate but passing rehab should be soft-penalized, not excluded."""
    # Insert op_success_rates entry with 0% S1 rate
    nb.conn.execute(
        """INSERT INTO op_success_rates (op_name, n_used, n_stage0_passed, n_stage05_passed, n_stage1_passed)
           VALUES ('gelu', 10, 10, 10, 0)"""
    )
    nb.conn.commit()

    # Save a passing rehabilitation result
    nb.save_op_rehabilitation_result(
        op_name="gelu",
        compile_passed=True,
        forward_passed=True,
        error_message=None,
        model_dim=64,
    )

    # Check cache returns it
    cache = nb.get_op_rehabilitation_cache()
    assert "gelu" in cache
    assert cache["gelu"]["compile_passed"] is True
    assert cache["gelu"]["forward_passed"] is True

    # Simulate the runner logic: rehabilitated ops get soft penalty
    op_weights = {}
    rehab_cache = nb.get_op_rehabilitation_cache()

    # Fake analytics result: gelu has 0% S1 rate
    failed_ops = [{"op_name": "gelu", "s1_rate": 0, "n_used": 10, "confidence": 0.9}]

    for op_info in failed_ops:
        if (
            op_info.get("s1_rate", 1) == 0
            and op_info.get("n_used", 0) >= 5
            and op_info.get("confidence", 0) >= 0.7
        ):
            op_name = op_info["op_name"]
            rehab = rehab_cache.get(op_name)
            if rehab and rehab.get("compile_passed") and rehab.get("forward_passed"):
                op_weights[op_name] = 0.5
            else:
                op_weights[op_name] = 0.1

    assert op_weights.get("gelu") == 0.5, "Rehabilitated op should get 0.5 penalty"
