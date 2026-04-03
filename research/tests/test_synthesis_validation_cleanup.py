from __future__ import annotations

import random

import pytest

from research.synthesis import templates as templates_mod
from research.synthesis._template_helpers import TemplateBuildError
from research.synthesis.graph import ComputationGraph
from research.synthesis.graph_validator import compute_kv_cacheable, validate_dim_flow


def test_apply_template_rolls_back_failed_template(monkeypatch):
    def _failing_template(graph, input_id, rng, weights=None):
        graph.add_op("linear_proj_down", [input_id], config={"out_dim": 16})
        raise TemplateBuildError("forced failure")

    monkeypatch.setitem(templates_mod.TEMPLATES, "_failing_template", _failing_template)

    graph = ComputationGraph(32)
    input_id = graph.add_input()
    before_nodes = set(graph.nodes)
    before_metadata = dict(graph.metadata)
    before_next_id = graph._next_id
    before_output_id = graph._output_node_id
    before_ir_version = graph._ir_version

    with pytest.raises(TemplateBuildError, match="forced failure"):
        templates_mod.apply_template(
            graph,
            input_id,
            random.Random(0),
            template_name="_failing_template",
        )

    assert set(graph.nodes) == before_nodes
    assert dict(graph.metadata) == before_metadata
    assert graph._next_id == before_next_id
    assert graph._output_node_id == before_output_id
    assert graph._ir_version == before_ir_version


def test_validate_dim_flow_enforces_budget_and_does_not_mutate_graph():
    graph = ComputationGraph(32)
    input_id = graph.add_input()
    hidden = graph.add_op("linear_proj", [input_id], config={"out_dim": 32})
    graph.set_output(hidden)

    before_ir_version = graph._ir_version
    before_nodes = {
        nid: (node.op_name, tuple(node.input_ids), dict(node.config))
        for nid, node in graph.nodes.items()
    }

    result = validate_dim_flow(graph, max_params=10)

    assert not result.valid
    assert any(
        error.startswith("Parameter budget exceeded:") for error in result.errors
    )
    assert graph._ir_version == before_ir_version
    after_nodes = {
        nid: (node.op_name, tuple(node.input_ids), dict(node.config))
        for nid, node in graph.nodes.items()
    }
    assert after_nodes == before_nodes


def test_validate_dim_flow_rejects_skip_only_graph_explicitly():
    graph = ComputationGraph(32)
    input_id = graph.add_input()
    graph.set_output(input_id)

    result = validate_dim_flow(graph)

    assert not result.valid
    assert "Graph is skip-only: output == input (no computation)" in result.errors


def test_validate_dim_flow_uses_native_summary_when_available():
    import research.synthesis.native_analysis as native_analysis

    graph = ComputationGraph(32)
    input_id = graph.add_input()
    hidden = graph.add_op("linear_proj", [input_id], config={"out_dim": 32})
    graph.set_output(hidden)

    native_analysis.reset_native_analysis_bindings()
    result = validate_dim_flow(graph, max_params=10)

    assert result.reachable_param_count == 1
    assert result.reachable_nontrivial_ops == 1
    assert result.reachable_ops == 1
    assert any(
        error.startswith("Parameter budget exceeded:") for error in result.errors
    )


def test_validate_dim_flow_summary_falls_back_to_python(monkeypatch):
    import research.synthesis.native_analysis as native_analysis

    monkeypatch.setattr(
        native_analysis, "_load_native_graph_analysis_lib", lambda: None
    )
    monkeypatch.setattr(native_analysis, "_try_import_aria_core", lambda: None)
    native_analysis.reset_native_analysis_bindings()

    graph = ComputationGraph(32)
    input_id = graph.add_input()
    hidden = graph.add_op("linear_proj", [input_id], config={"out_dim": 32})
    graph.set_output(hidden)

    result = validate_dim_flow(graph)

    assert result.reachable_param_count == 1
    assert result.reachable_ops == 1

    kv_graph = ComputationGraph(32)
    kv_in = kv_graph.add_input()
    kv_out = kv_graph.add_op("spectral_filter", [kv_in])
    kv_graph.set_output(kv_out)
    assert compute_kv_cacheable(kv_graph) is False


def test_validate_dim_flow_warns_on_dead_parameterized_nodes():
    graph = ComputationGraph(32)
    input_id = graph.add_input()
    live = graph.add_op("linear_proj", [input_id], config={"out_dim": 32})
    dead = graph.add_op("linear_proj", [input_id], config={"out_dim": 32})
    graph.set_output(live)

    result = validate_dim_flow(graph)

    assert any(
        f"Node {dead} (linear_proj): parameterized but unreachable" in warning
        for warning in result.warnings
    )


def test_validate_dim_flow_dead_parameterized_mask_falls_back_to_python(monkeypatch):
    import research.synthesis.native_analysis as native_analysis
    import research.synthesis.native_dim_flow as native_dim_flow

    monkeypatch.setattr(
        native_analysis, "_load_native_graph_analysis_lib", lambda: None
    )
    monkeypatch.setattr(native_analysis, "_try_import_aria_core", lambda: None)
    monkeypatch.setattr(
        native_dim_flow, "_load_native_graph_analysis_lib", lambda: None
    )
    native_analysis.reset_native_analysis_bindings()

    graph = ComputationGraph(32)
    input_id = graph.add_input()
    live = graph.add_op("linear_proj", [input_id], config={"out_dim": 32})
    dead = graph.add_op("linear_proj", [input_id], config={"out_dim": 32})
    graph.set_output(live)

    result = validate_dim_flow(graph)

    assert any(
        f"Node {dead} (linear_proj): parameterized but unreachable" in warning
        for warning in result.warnings
    )
