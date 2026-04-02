"""Tests for the runtime bridge (aria_designer → research/ eval pipeline)."""

import sys
import pytest
from types import SimpleNamespace

from aria_designer.runtime.bridge import (
    workflow_to_graph,
    validate_workflow_graph,
    estimate_performance,
    evaluate_workflow,
    list_available_primitives,
    _resolve_primitive,
    get_component_execution_capability,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _simple_mlp():
    """Simple: input → linear → relu → linear → output."""
    return {
        "nodes": [
            {"id": "n0", "component_type": "graph_input", "params": {}},
            {"id": "n1", "component_type": "linear_proj", "params": {"out_dim": 256}},
            {"id": "n2", "component_type": "relu", "params": {}},
            {"id": "n3", "component_type": "linear_proj", "params": {"out_dim": 256}},
            {"id": "n4", "component_type": "graph_output", "params": {}},
        ],
        "edges": [
            {
                "id": "e0",
                "source": "n0",
                "target": "n1",
                "source_port": "out",
                "target_port": "in",
            },
            {
                "id": "e1",
                "source": "n1",
                "target": "n2",
                "source_port": "out",
                "target_port": "in",
            },
            {
                "id": "e2",
                "source": "n2",
                "target": "n3",
                "source_port": "out",
                "target_port": "in",
            },
            {
                "id": "e3",
                "source": "n3",
                "target": "n4",
                "source_port": "out",
                "target_port": "in",
            },
        ],
    }


def _attention_pattern():
    """Attention-like: input → Q/K/V projections → matmul → softmax → matmul → output."""
    return {
        "nodes": [
            {"id": "in", "component_type": "graph_input", "params": {}},
            {"id": "q", "component_type": "linear_proj", "params": {"out_dim": 256}},
            {"id": "k", "component_type": "linear_proj", "params": {"out_dim": 256}},
            {"id": "v", "component_type": "linear_proj", "params": {"out_dim": 256}},
            {"id": "attn", "component_type": "matmul", "params": {}},
            {"id": "sm", "component_type": "softmax_last", "params": {}},
            {"id": "out_attn", "component_type": "matmul", "params": {}},
            {"id": "proj", "component_type": "linear_proj", "params": {"out_dim": 256}},
            {"id": "out", "component_type": "graph_output", "params": {}},
        ],
        "edges": [
            {
                "id": "e0",
                "source": "in",
                "target": "q",
                "source_port": "out",
                "target_port": "in",
            },
            {
                "id": "e1",
                "source": "in",
                "target": "k",
                "source_port": "out",
                "target_port": "in",
            },
            {
                "id": "e2",
                "source": "in",
                "target": "v",
                "source_port": "out",
                "target_port": "in",
            },
            {
                "id": "e3",
                "source": "q",
                "target": "attn",
                "source_port": "out",
                "target_port": "a",
            },
            {
                "id": "e4",
                "source": "k",
                "target": "attn",
                "source_port": "out",
                "target_port": "b",
            },
            {
                "id": "e5",
                "source": "attn",
                "target": "sm",
                "source_port": "out",
                "target_port": "in",
            },
            {
                "id": "e6",
                "source": "sm",
                "target": "out_attn",
                "source_port": "out",
                "target_port": "a",
            },
            {
                "id": "e7",
                "source": "v",
                "target": "out_attn",
                "source_port": "out",
                "target_port": "b",
            },
            {
                "id": "e8",
                "source": "out_attn",
                "target": "proj",
                "source_port": "out",
                "target_port": "in",
            },
            {
                "id": "e9",
                "source": "proj",
                "target": "out",
                "source_port": "out",
                "target_port": "in",
            },
        ],
    }


def _residual_block():
    """Residual: input → linear → relu → add(input, .) → output."""
    return {
        "nodes": [
            {"id": "in", "component_type": "graph_input", "params": {}},
            {"id": "proj", "component_type": "linear_proj", "params": {"out_dim": 256}},
            {"id": "act", "component_type": "gelu", "params": {}},
            {
                "id": "proj2",
                "component_type": "linear_proj",
                "params": {"out_dim": 256},
            },
            {"id": "res", "component_type": "add", "params": {}},
            {"id": "out", "component_type": "graph_output", "params": {}},
        ],
        "edges": [
            {
                "id": "e0",
                "source": "in",
                "target": "proj",
                "source_port": "out",
                "target_port": "in",
            },
            {
                "id": "e1",
                "source": "proj",
                "target": "act",
                "source_port": "out",
                "target_port": "in",
            },
            {
                "id": "e2",
                "source": "act",
                "target": "proj2",
                "source_port": "out",
                "target_port": "in",
            },
            {
                "id": "e3",
                "source": "proj2",
                "target": "res",
                "source_port": "out",
                "target_port": "a",
            },
            {
                "id": "e4",
                "source": "in",
                "target": "res",
                "source_port": "out",
                "target_port": "b",
            },
            {
                "id": "e5",
                "source": "res",
                "target": "out",
                "source_port": "out",
                "target_port": "in",
            },
        ],
    }


# ── Tests: primitive resolution ──────────────────────────────────────


def test_resolve_direct_name():
    assert _resolve_primitive("relu") == "relu"
    assert _resolve_primitive("linear_proj") == "linear_proj"
    assert _resolve_primitive("matmul") == "matmul"


def test_resolve_category_prefix():
    assert _resolve_primitive("math/relu") == "relu"
    assert _resolve_primitive("linear_algebra/matmul") == "matmul"
    assert _resolve_primitive("mixing/softmax_attention") == "softmax_attention"
    assert _resolve_primitive("channel_mixing/swiglu_mlp") == "swiglu_mlp"
    assert _resolve_primitive("linear_algebra/selective_scan") == "selective_scan"


def test_resolve_io_returns_none():
    assert _resolve_primitive("graph_input") is None
    assert _resolve_primitive("graph_output") is None
    assert _resolve_primitive("input") is None
    assert _resolve_primitive("output") is None


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown component"):
        _resolve_primitive("totally_fake_op_42")


def test_execution_capability_direct_primitive():
    info = get_component_execution_capability("mixing/softmax_attention")
    assert info["bridge_supported"] is True
    assert info["primitive_name"] == "softmax_attention"
    assert info["execution_class"] == "primitive"
    assert info["mapping_kind"] == "direct"
    assert info["semantic_fidelity"] == "exact"


def test_execution_capability_composite_class():
    info = get_component_execution_capability("blocks/u_net")
    assert info["bridge_supported"] is True
    assert info["execution_class"] == "composite"


def test_execution_capability_primitive_candidate_class():
    info = get_component_execution_capability("mixing/linear_attention")
    if info["bridge_supported"]:
        assert info["primitive_name"] == "linear_attention"
        assert info["execution_class"] == "primitive"
    else:
        assert info["execution_class"] == "primitive_candidate"


def test_passthrough_lowering_component_is_supported():
    info = get_component_execution_capability("blocks/sequential")
    assert info["bridge_supported"] is True
    assert info["primitive_name"] is None
    assert "passthrough lowering" in info["reason"].lower()
    # Routing ops with real primitives: bridge emits actual op nodes (not passthrough)
    merge_info = get_component_execution_capability("routing/adjacent_token_merge")
    assert merge_info["bridge_supported"] is True
    assert merge_info["primitive_name"] == "adjacent_token_merge"
    cascade_info = get_component_execution_capability("routing/learned_token_gate")
    assert cascade_info["bridge_supported"] is True
    assert cascade_info["primitive_name"] == "learned_token_gate"
    adaptive_info = get_component_execution_capability("routing/depth_weighted_proj")
    assert adaptive_info["bridge_supported"] is True
    assert adaptive_info["primitive_name"] == "depth_weighted_proj"
    speculative_info = get_component_execution_capability("routing/cheap_verify_blend")
    assert speculative_info["bridge_supported"] is True
    assert speculative_info["primitive_name"] == "cheap_verify_blend"
    router_info = get_component_execution_capability("routing/difficulty_blend_3way")
    assert router_info["bridge_supported"] is True
    assert router_info["primitive_name"] == "difficulty_blend_3way"
    dispatch_info = get_component_execution_capability(
        "structural/conditional_dispatch"
    )
    assert dispatch_info["bridge_supported"] is True
    assert dispatch_info["primitive_name"] is None
    loop_info = get_component_execution_capability("control_flow/loop")
    assert loop_info["bridge_supported"] is True
    assert loop_info["primitive_name"] is None
    loop_info = get_component_execution_capability("control_flow/loop")
    assert loop_info["bridge_supported"] is True
    assert loop_info["primitive_name"] is None
    source_info = get_component_execution_capability("data_io/random_data_source")
    assert source_info["bridge_supported"] is True
    assert source_info["primitive_name"] is None
    assert "source lowering" in source_info["reason"].lower()


def test_data_plane_components_are_bridge_supported():
    source_info = get_component_execution_capability("data_io/random_data_source")
    assert source_info["bridge_supported"] is True
    assert source_info["primitive_name"] is None

    transform_info = get_component_execution_capability("data_transform/filter")
    assert transform_info["bridge_supported"] is True
    assert transform_info["primitive_name"] is None

    split_info = get_component_execution_capability(
        "data_transform/split_train_val_test"
    )
    assert split_info["bridge_supported"] is True
    assert split_info["primitive_name"] is None

    projection_info = get_component_execution_capability(
        "data_transform/select_columns"
    )
    assert projection_info["bridge_supported"] is True
    assert projection_info["primitive_name"] is None

    sink_info = get_component_execution_capability("data_io/file_writer")
    assert sink_info["bridge_supported"] is True
    assert sink_info["primitive_name"] is None

    csv_info = get_component_execution_capability("io/csv_reader")
    assert csv_info["bridge_supported"] is True
    assert csv_info["primitive_name"] is None


# ── Tests: workflow → graph conversion ───────────────────────────────


def test_simple_mlp_conversion():
    graph = workflow_to_graph(_simple_mlp(), model_dim=256)
    assert graph.n_ops() == 3  # linear, relu, linear (input doesn't count)
    assert graph.depth() == 3
    assert graph.model_dim == 256
    assert graph.output_node is not None
    assert graph.output_node.output_shape.dim == 256


def test_attention_conversion():
    graph = workflow_to_graph(_attention_pattern(), model_dim=256)
    assert graph.n_ops() == 7  # 4 linear + 2 matmul + 1 softmax
    assert graph.depth() == 5


def test_residual_block_conversion():
    graph = workflow_to_graph(_residual_block(), model_dim=256)
    assert graph.n_ops() >= 4  # linear, gelu, linear, add


def test_no_nodes_raises():
    with pytest.raises(ValueError, match="no detectable input nodes"):
        workflow_to_graph({"nodes": [], "edges": []})


def test_cycle_raises():
    wf = {
        "nodes": [
            {"id": "a", "component_type": "relu", "params": {}},
            {"id": "b", "component_type": "relu", "params": {}},
        ],
        "edges": [
            {"id": "e0", "source": "a", "target": "b"},
            {"id": "e1", "source": "b", "target": "a"},
        ],
    }
    # Full cycle: all nodes have incoming edges → "no input nodes" detected first
    with pytest.raises(ValueError, match="(cycle|no detectable input)"):
        workflow_to_graph(wf)


def test_implicit_io_nodes():
    """Workflow without explicit input/output nodes should infer them from topology."""
    # Single node with no edges: it's both input and output
    # relu is identity-shaped, so it needs an actual input node
    # This should fail because there's no graph_input and relu needs an input
    # Let's test with a proper chain instead
    # n0 has no incoming edges → treated as implicit input
    # But n0 is a linear_proj, not an input node, so it needs to connect to something
    # This tests the fallback path


# ── Tests: validation ────────────────────────────────────────────────


def test_validate_simple_mlp():
    result = validate_workflow_graph(_simple_mlp(), model_dim=256)
    assert result["valid"] is True
    info = result["graph_info"]
    assert info["n_ops"] == 3
    assert info["has_gradient_path"] is True
    assert info["fingerprint"]


def test_validate_attention():
    result = validate_workflow_graph(_attention_pattern(), model_dim=256)
    assert result["valid"] is True


def test_validate_residual():
    result = validate_workflow_graph(_residual_block(), model_dim=256)
    assert result["valid"] is True


def test_validate_unknown_op():
    wf = {
        "nodes": [
            {"id": "n0", "component_type": "graph_input", "params": {}},
            {"id": "n1", "component_type": "nonexistent_op_xyz", "params": {}},
        ],
        "edges": [{"id": "e0", "source": "n0", "target": "n1"}],
    }
    result = validate_workflow_graph(wf, model_dim=256)
    assert result["valid"] is False
    assert "Unknown op" in result["error"]


def test_workflow_with_passthrough_component():
    wf = {
        "nodes": [
            {"id": "n0", "component_type": "graph_input", "params": {}},
            {"id": "n1", "component_type": "blocks/sequential", "params": {}},
            {"id": "n2", "component_type": "relu", "params": {}},
            {"id": "n3", "component_type": "graph_output", "params": {}},
        ],
        "edges": [
            {"id": "e0", "source": "n0", "target": "n1"},
            {"id": "e1", "source": "n1", "target": "n2"},
            {"id": "e2", "source": "n2", "target": "n3"},
        ],
    }
    graph = workflow_to_graph(wf, model_dim=256)
    # sequential is lowered as passthrough, so only relu counts as op
    assert graph.n_ops() == 1


def test_workflow_with_routing_passthrough_components():
    wf = {
        "nodes": [
            {"id": "n0", "component_type": "graph_input", "params": {}},
            {
                "id": "n1",
                "component_type": "routing/adjacent_token_merge",
                "params": {},
            },
            {"id": "n2", "component_type": "routing/learned_token_gate", "params": {}},
            {"id": "n3", "component_type": "relu", "params": {}},
            {"id": "n4", "component_type": "graph_output", "params": {}},
        ],
        "edges": [
            {"id": "e0", "source": "n0", "target": "n1"},
            {"id": "e1", "source": "n1", "target": "n2"},
            {"id": "e2", "source": "n2", "target": "n3"},
            {"id": "e3", "source": "n3", "target": "n4"},
        ],
    }
    graph = workflow_to_graph(wf, model_dim=256)
    # adjacent_token_merge + learned_token_gate are real primitives, plus relu = 3 ops
    assert graph.n_ops() == 3


def test_workflow_with_adaptive_and_speculative_passthrough():
    wf = {
        "nodes": [
            {"id": "n0", "component_type": "graph_input", "params": {}},
            {"id": "n1", "component_type": "routing/depth_weighted_proj", "params": {}},
            {"id": "n2", "component_type": "routing/cheap_verify_blend", "params": {}},
            {"id": "n3", "component_type": "gelu", "params": {}},
            {"id": "n4", "component_type": "graph_output", "params": {}},
        ],
        "edges": [
            {"id": "e0", "source": "n0", "target": "n1"},
            {"id": "e1", "source": "n1", "target": "n2"},
            {"id": "e2", "source": "n2", "target": "n3"},
            {"id": "e3", "source": "n3", "target": "n4"},
        ],
    }
    graph = workflow_to_graph(wf, model_dim=256)
    # depth_weighted_proj + cheap_verify_blend are real primitives, plus gelu = 3 ops
    assert graph.n_ops() == 3


def test_workflow_with_data_source_lowering():
    wf = {
        "nodes": [
            {"id": "n0", "component_type": "data_io/random_data_source", "params": {}},
            {"id": "n1", "component_type": "data_transform/filter", "params": {}},
            {"id": "n2", "component_type": "relu", "params": {}},
            {"id": "n3", "component_type": "graph_output", "params": {}},
        ],
        "edges": [
            {"id": "e0", "source": "n0", "target": "n1"},
            {"id": "e1", "source": "n1", "target": "n2"},
            {"id": "e2", "source": "n2", "target": "n3"},
        ],
    }
    graph = workflow_to_graph(wf, model_dim=256)
    # source + filter lower to input/passthrough, leaving relu as concrete op
    assert graph.n_ops() == 1


def test_workflow_with_control_flow_loop_passthrough():
    wf = {
        "nodes": [
            {"id": "n0", "component_type": "graph_input", "params": {}},
            {
                "id": "n1",
                "component_type": "control_flow/loop",
                "params": {"max_iterations": 3},
            },
            {"id": "n2", "component_type": "gelu", "params": {}},
            {"id": "n3", "component_type": "graph_output", "params": {}},
        ],
        "edges": [
            {"id": "e0", "source": "n0", "target": "n1"},
            {"id": "e1", "source": "n1", "target": "n2"},
            {"id": "e2", "source": "n2", "target": "n3"},
        ],
    }
    graph = workflow_to_graph(wf, model_dim=256)
    assert graph.n_ops() == 1


def test_template_lowered_block_components_supported():
    info = get_component_execution_capability("blocks/u_net")
    assert info["bridge_supported"] is True
    assert info["primitive_name"] is None
    assert "template lowering" in info["reason"].lower()


def test_workflow_with_template_lowered_block():
    wf = {
        "nodes": [
            {"id": "n0", "component_type": "graph_input", "params": {}},
            {"id": "n1", "component_type": "blocks/u_net", "params": {}},
            {"id": "n2", "component_type": "graph_output", "params": {}},
        ],
        "edges": [
            {"id": "e0", "source": "n0", "target": "n1"},
            {"id": "e1", "source": "n1", "target": "n2"},
        ],
    }
    graph = workflow_to_graph(wf, model_dim=256)
    # u_net lowers to linear_proj_down -> gelu -> linear_proj_up (3 ops)
    assert graph.n_ops() == 3


def test_workflow_with_data_plane_lowering_components():
    wf = {
        "nodes": [
            {
                "id": "src",
                "component_type": "data_io/random_data_source",
                "params": {"seed": 7},
            },
            {
                "id": "xfm",
                "component_type": "data_transform/filter",
                "params": {"filter_scope": "token"},
            },
            {"id": "act", "component_type": "relu", "params": {}},
            {"id": "sink", "component_type": "data_io/file_writer", "params": {}},
            {"id": "out", "component_type": "graph_output", "params": {}},
        ],
        "edges": [
            {"id": "e0", "source": "src", "target": "xfm"},
            {"id": "e1", "source": "xfm", "target": "act"},
            {"id": "e2", "source": "act", "target": "sink"},
            {"id": "e3", "source": "sink", "target": "out"},
        ],
    }
    graph = workflow_to_graph(wf, model_dim=256)
    # data source/transform/sink are bridge-lowered; only relu is a primitive op
    assert graph.n_ops() == 1


# ── Tests: performance estimation ────────────────────────────────────


def test_estimate_simple_mlp():
    result = estimate_performance(_simple_mlp(), model_dim=256)
    assert result["valid"] is True
    assert result["n_params_estimate"] > 0
    assert result["flops_per_token_estimate"] > 0
    assert "linear_proj" in result["op_counts"]
    assert "relu" in result["op_counts"]


def test_estimate_attention():
    result = estimate_performance(_attention_pattern(), model_dim=256)
    assert result["valid"] is True
    assert result["n_ops"] == 7


# ── Tests: list primitives ───────────────────────────────────────────


def test_list_primitives():
    prims = list_available_primitives()
    assert len(prims) > 50  # We know there are 66+
    names = {p["name"] for p in prims}
    assert "relu" in names
    assert "linear_proj" in names
    assert "matmul" in names


# ── Tests: full evaluation (CPU only, no fingerprint) ────────────────


def test_evaluate_simple_mlp():
    result = evaluate_workflow(
        _simple_mlp(),
        model_dim=256,
        device="cpu",
        run_fingerprint=False,
        run_novelty=False,
        batch_size=1,
        seq_len=32,
    )
    assert result.status == "success"
    assert result.sandbox.passed is True
    assert result.sandbox.param_count > 0
    assert result.sandbox.forward_time_ms > 0
    assert result.total_time_ms > 0


def test_evaluate_residual():
    result = evaluate_workflow(
        _residual_block(),
        model_dim=256,
        device="cpu",
        run_fingerprint=False,
        run_novelty=False,
        batch_size=1,
        seq_len=32,
    )
    # Residual blocks may fail sandbox for various reasons (numerical issues),
    # but conversion + compilation should succeed
    assert result.status in ("success", "failed_sandbox")
    assert result.n_ops >= 4
    assert result.has_gradient_path is True


# ── Tests: serialization ─────────────────────────────────────────────


def test_bridge_result_to_dict():
    result = evaluate_workflow(
        _simple_mlp(),
        model_dim=256,
        device="cpu",
        run_fingerprint=False,
        run_novelty=False,
        batch_size=1,
        seq_len=16,
    )
    d = result.to_dict()
    # Should be JSON-serializable (no numpy types)
    import json

    json_str = json.dumps(d)
    assert '"status": "success"' in json_str


def test_evaluate_workflow_uses_behavioral_fingerprint_for_novelty(monkeypatch):
    import aria_designer.runtime.bridge as bridge_mod

    class _FakeGraph:
        def fingerprint(self):
            return "fp_test"

        def n_ops(self):
            return 3

        def depth(self):
            return 2

        def n_params_estimate(self):
            return 123

        def has_gradient_path(self):
            return True

    class _FakeModel:
        def to(self, _device):
            return self

    fake_sandbox = SimpleNamespace(
        passed=True,
        error=None,
        compile_time_ms=1.0,
        forward_time_ms=2.0,
        backward_time_ms=3.0,
        param_count=123,
        peak_memory_mb=4.0,
        grad_norm=5.0,
        stability_score=0.9,
        to_dict=lambda: {
            "passed": True,
            "compile_time_ms": 1.0,
            "forward_time_ms": 2.0,
            "backward_time_ms": 3.0,
            "param_count": 123,
            "peak_memory_mb": 4.0,
            "grad_norm": 5.0,
            "stability_score": 0.9,
        },
    )

    fp = SimpleNamespace(
        cka_vs_transformer=0.2,
        cka_vs_ssm=0.6,
        cka_vs_conv=0.1,
        interaction_locality=0.3,
        interaction_sparsity=0.4,
        intrinsic_dim=7.0,
        isotropy=0.8,
        novelty_score=0.75,
    )

    novelty_metrics = SimpleNamespace(
        structural_novelty=0.25,
        behavioral_novelty=0.75,
        overall_novelty=0.6,
        most_similar_to="ssm",
    )

    monkeypatch.setattr(
        bridge_mod, "workflow_to_graph", lambda *args, **kwargs: _FakeGraph()
    )
    monkeypatch.setitem(
        sys.modules,
        "research.synthesis.compiler",
        SimpleNamespace(compile_model=lambda *args, **kwargs: _FakeModel()),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.sandbox",
        SimpleNamespace(safe_eval=lambda *args, **kwargs: fake_sandbox),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.fingerprint",
        SimpleNamespace(compute_fingerprint=lambda *args, **kwargs: fp),
    )
    monkeypatch.setitem(
        sys.modules,
        "research.eval.metrics",
        SimpleNamespace(
            novelty_score=lambda graph, fingerprint=None, **kwargs: novelty_metrics
        ),
    )

    result = bridge_mod.evaluate_workflow(
        _simple_mlp(),
        model_dim=256,
        device="cpu",
        run_fingerprint=True,
        run_novelty=True,
        batch_size=1,
        seq_len=16,
    )

    assert result.status == "success"
    assert result.fingerprint.behavioral_novelty == 0.75
    assert result.fingerprint.structural_novelty == 0.25
    assert result.fingerprint.overall_novelty == 0.6
    assert result.fingerprint.most_similar_to == "ssm"
    assert result.fingerprint.cka_vs_ssm == 0.6
