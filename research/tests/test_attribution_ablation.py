import pytest
import json
import os
import tempfile

from research.scientist.analytics import ExperimentAnalytics
from research.scientist.notebook import LabNotebook
from research.scientist.runner import (
    ExperimentRunner,
    RunConfig,
    propose_ablation_suite,
)
from research.synthesis.compiler import compile_model
from research.synthesis.graph import ComputationGraph
from research.synthesis.validator import validate_graph

pytestmark = pytest.mark.unit


def _graph_json_for_ops(ops, *, variant: int | None = None):
    """Build a deterministic graph JSON.

    If ``variant`` is provided, an extra linear_proj node is inserted a
    variant-specific number of times so the canonical fingerprint is
    distinct per variant. The corpus loader deduplicates by canonical
    graph fingerprint, so tests that need independent per-row analysis
    must supply distinct variants.
    """
    nodes = {"0": {"id": 0, "op_name": "input", "input_ids": []}}
    prev = 0
    for idx, op in enumerate(ops, start=1):
        nodes[str(idx)] = {"id": idx, "op_name": op, "input_ids": [prev]}
        prev = idx
    if variant is not None:
        for v in range(int(variant) + 1):
            idx = len(nodes)
            nodes[str(idx)] = {
                "id": idx,
                "op_name": "linear_proj",
                "input_ids": [prev],
                "config": {"out_dim": 16 + (int(variant) % 16)},
            }
            prev = idx
    return json.dumps({"nodes": nodes})


def _make_notebook_with_fixed_programs() -> LabNotebook:
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "attr.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment(
        "synthesis", {"n_programs": 40}, "attribution determinism"
    )

    for i in range(20):
        nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=f"math_{i}",
            graph_json=_graph_json_for_ops(["linear_proj", "conv1d_seq", "gelu"]),
            stage1_passed=1 if i < 14 else 0,
            graph_depth=7,
            graph_uses_math_spaces=1,
            trust_label="test_fixture",
        )
    for i in range(20):
        nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=f"plain_{i}",
            graph_json=_graph_json_for_ops(["linear_proj", "gelu", "tanh"]),
            stage1_passed=1 if i < 4 else 0,
            graph_depth=3,
            graph_uses_math_spaces=0,
            trust_label="test_fixture",
        )

    nb.complete_experiment(
        exp_id,
        {
            "total": 40,
            "stage0_passed": 40,
            "stage05_passed": 40,
            "stage1_passed": 18,
            "best_loss_ratio": 0.2,
            "best_novelty_score": 0.5,
        },
    )
    return nb


def test_attribution_is_deterministic_for_fixed_db():
    nb = _make_notebook_with_fixed_programs()
    analytics = ExperimentAnalytics(nb)
    report_a = analytics.grammar_weight_attribution_report()
    report_b = analytics.grammar_weight_attribution_report()

    assert json.dumps(report_a, sort_keys=True) == json.dumps(report_b, sort_keys=True)
    assert isinstance(report_a.get("factors"), list)
    assert all("q_value" in row for row in report_a.get("factors", []))
    assert isinstance(report_a.get("matched_controls"), list)
    nb.close()


def test_propose_ablation_suite_graphs_validate_and_compile():
    g = ComputationGraph(model_dim=64)
    x = g.add_input()
    n1 = g.add_op("linear_proj", [x], config={"out_dim": 64})
    n2 = g.add_op("conv1d_seq", [n1], config={})
    n3 = g.add_op("gelu", [n2], config={})
    g.set_output(n3)

    suite = propose_ablation_suite(g, "conv1d_seq improves the stage1 pass rate")
    assert suite, "Expected at least one ablation graph."

    for ablated in suite:
        validation = validate_graph(ablated, max_ops=20, max_depth=15)
        assert validation.valid, f"Invalid ablation graph: {validation.errors}"
        model = compile_model([ablated], vocab_size=256, max_seq_len=32)
        assert model is not None


def test_attribution_filters_unknown_depth_bucket_as_top_signal():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "unknown_depth.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment(
        "synthesis", {"n_programs": 60}, "unknown-depth signal quality"
    )

    for i in range(30):
        nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=f"unk_{i}",
            graph_json=_graph_json_for_ops(["linear_proj", "gelu", "tanh"], variant=i),
            stage1_passed=1 if i < 24 else 0,
            graph_depth=None,
            graph_uses_math_spaces=0,
            trust_label="test_fixture",
        )
    for i in range(30):
        nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=f"known_{i}",
            graph_json=_graph_json_for_ops(
                ["linear_proj", "gelu", "tanh"], variant=1000 + i
            ),
            stage1_passed=1 if i < 3 else 0,
            graph_depth=4,
            graph_uses_math_spaces=0,
            trust_label="test_fixture",
        )

    nb.flush_writes()
    analytics = ExperimentAnalytics(nb)
    report = analytics.grammar_weight_attribution_report()

    top_signal = report.get("top_signal")
    assert top_signal is None
    assert report.get("strong_correlational_evidence") is False
    assert report.get("uncertainty", {}).get("correlational_signal_count", 0) >= 1
    assert (
        report.get("uncertainty", {}).get("interpretable_correlational_signal_count", 0)
        == 0
    )
    nb.close()


def test_ablation_runner_skips_experiment_when_no_evaluable_graphs():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "ablation_skip.db")
    nb = LabNotebook(db_path)
    runner = ExperimentRunner(notebook_path=db_path)

    invalid_graph = ComputationGraph(model_dim=64)
    invalid_graph.add_input()

    before = nb.conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    exp_ids, outcome = runner._run_ablation_experiment(
        nb=nb,
        config=RunConfig(device="cpu", n_programs=1),
        hypothesis="signal=depth_bucket:unknown",
        ablation_graphs=[invalid_graph],
    )
    after = nb.conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]

    assert exp_ids == []
    assert outcome == "skipped_no_evaluable_graphs"
    assert before == after

    event = nb.conn.execute(
        "SELECT event_type FROM learning_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert event is not None
    assert event[0] == "ablation_skipped_no_evaluable_graphs"
    nb.close()
