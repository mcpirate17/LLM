from __future__ import annotations

import json

import pytest
import torch

from research.synthesis.graph import ComputationGraph
from research.synthesis.compiler import compile_graph, compile_model
from research.synthesis.grammar import GrammarConfig, generate_layer_graph
from research.synthesis.reference_architectures import build_reference
from research.synthesis.ir_executor import IRExecutor
from research.synthesis.native_analysis import reset_native_analysis_bindings
from research.synthesis.validator import validate_graph
from research.tests.test_component_graphs import graph_token_routing


def _make_relu_graph(model_dim: int = 8) -> ComputationGraph:
    graph = ComputationGraph(model_dim)
    inp = graph.add_input()
    out = graph.add_op("relu", [inp])
    graph.set_output(out)
    return graph


def _make_chain_graph(model_dim: int = 8) -> ComputationGraph:
    graph = ComputationGraph(model_dim)
    inp = graph.add_input()
    relu = graph.add_op("relu", [inp])
    out = graph.add_op("exp", [relu])
    graph.set_output(out)
    return graph


def test_structural_analysis_uses_native_runtime_when_available():
    graph = _make_relu_graph()
    dead = graph.add_op("relu", [graph.input_node.id])
    assert dead in graph.nodes

    reset_native_analysis_bindings()
    analysis = graph._analysis_ir().analyze_structure(include_reachable=True)

    assert analysis.backend in {"native", "aria_core"}
    assert analysis.has_gradient_path is True
    assert analysis.reachable_count == 2
    assert analysis.depth == 1


def test_token_merge_conv_lowers_forced_template():
    graph = generate_layer_graph(
        GrammarConfig(
            max_depth=8,
            model_dim=64,
            forced_template="token_merge_conv",
            routing_mandatory=False,
        ),
        seed=123,
    )
    op_names = [node.op_name for node in graph.nodes.values() if not node.is_input]
    assert op_names[:9] == [
        "rmsnorm",
        "adjacent_token_merge",
        "add",
        "rmsnorm",
        "conv1d_seq",
        "swiglu_mlp",
        "gelu",
        "nm_sparse_linear",
        "add",
    ]
    assert graph.output_node is not None
    assert graph.output_node.output_shape.dim == 64


def test_structural_analysis_falls_back_to_python(monkeypatch):
    import research.synthesis.native_analysis as native_analysis

    monkeypatch.setattr(
        native_analysis, "_load_native_graph_analysis_lib", lambda: None
    )
    monkeypatch.setattr(native_analysis, "_try_import_aria_core", lambda: None)
    reset_native_analysis_bindings()

    graph = _make_relu_graph()
    analysis = graph._analysis_ir().analyze_structure(include_reachable=True)

    assert analysis.backend == "python"
    assert analysis.has_gradient_path is True
    assert analysis.reachable_count == 2
    assert analysis.depth == 1


def test_ir_executor_falls_back_to_python_loop(monkeypatch):
    import research.scientist.native.dispatch as native_dispatch

    monkeypatch.setattr(
        native_dispatch,
        "_check_native_op_support",
        lambda graphs, native_lib=None: {"supported": []},
    )

    graph = _make_relu_graph()
    executor = IRExecutor(graph.lower_to_ir(), source_graph=graph)
    x = torch.tensor([[[-1.0] * 8, [2.0] * 8]], dtype=torch.float32)
    out = executor(x)

    assert torch.equal(out, torch.relu(x))
    assert executor.execution_stats["last_execution_path"] == "python_ir_loop"
    assert executor.execution_stats["python_ir_loop_fallbacks"] == 1
    assert executor.execution_stats["native_subgraph_available"] is False


def test_bound_dispatcher_rmsnorm_uses_flattened_rows(monkeypatch):
    import research.synthesis.native_compile as native_compile

    monkeypatch.setattr(
        native_compile, "get_supported_native_ops", lambda graph: {"rmsnorm"}
    )

    graph = ComputationGraph(8)
    inp = graph.add_input()
    out = graph.add_op("rmsnorm", [inp])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    dispatcher = layer._subgraph_dispatcher
    assert dispatcher is not None

    x = torch.zeros(2, 3, 8)
    plan = dispatcher._plan_for_input(x)
    nodes = json.loads(plan.ir_json)["nodes"]
    rmsnorm_node = next(node for node in nodes if node["op_name"] == "rmsnorm")

    assert rmsnorm_node["config"]["batch"] == 6
    assert rmsnorm_node["config"]["dim"] == 8


def test_bound_native_chain_skips_host_bridge_for_non_cpu_payloads(monkeypatch):
    import research.synthesis.native_bound_segments as native_bound_segments

    graph = ComputationGraph(8)
    inp = graph.add_input()
    proj = graph.add_op("linear_proj", [inp], {"out_dim": 8})
    relu = graph.add_op("relu", [proj])
    out = graph.add_op("cumsum", [relu])
    graph.set_output(out)

    layer = compile_graph(graph, use_ir=False)
    dispatcher = native_bound_segments.BoundNativeChainDispatcher(
        [
            native_bound_segments._BoundChainNode("linear_proj", layer.ops[str(proj)]),
            native_bound_segments._BoundChainNode("relu", layer.ops[str(relu)]),
        ]
    )

    monkeypatch.setattr(
        native_bound_segments,
        "supports_host_array_bridge",
        lambda *values: False,
    )
    monkeypatch.setattr(
        native_bound_segments,
        "dispatch_graph_native_multi_input_cached",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("host bridge should be skipped")
        ),
    )

    x = torch.randn(2, 3, 8, dtype=torch.float32)
    assert dispatcher.try_dispatch(x) is None


def test_fingerprint_representations_preserve_rank_under_no_grad():
    model = compile_model(
        [build_reference("gpt2", d_model=64)],
        vocab_size=1000,
        max_seq_len=32,
    ).eval()
    input_ids = torch.randint(0, 1000, (8, 32))

    with torch.no_grad():
        logits, reps = model._fingerprint_representations(input_ids)

    assert logits.shape == (8, 32, 1000)
    assert reps.shape == (8, 32, 64)


def test_bound_native_dispatch_skips_host_bridge_for_non_cpu_tensors(monkeypatch):
    import research.scientist.native.dispatch as native_dispatch
    import research.synthesis.native_bound_graph as native_bound_graph

    monkeypatch.setattr(
        native_dispatch,
        "_check_native_op_support",
        lambda graphs, native_lib=None: {"supported": ["linear_proj", "relu"]},
    )

    graph = ComputationGraph(8)
    inp = graph.add_input()
    proj = graph.add_op("linear_proj", [inp], {"out_dim": 8})
    out = graph.add_op("relu", [proj])
    graph.set_output(out)

    executor = IRExecutor(graph.lower_to_ir(), source_graph=graph)
    dispatcher = executor._subgraph_dispatcher
    assert dispatcher is not None

    monkeypatch.setattr(
        native_bound_graph,
        "supports_host_array_bridge",
        lambda *values: False,
    )
    monkeypatch.setattr(
        native_bound_graph,
        "dispatch_graph_native_multi_input_cached",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("host bridge should be skipped")
        ),
    )

    x = torch.randn(2, 3, 8, dtype=torch.float32)
    result = dispatcher.try_dispatch(x)

    assert result is None
    assert dispatcher.stats["subgraph_dispatches"] == 0
    assert (
        dispatcher.stats["last_refusal_reason"]
        == "host_array_bridge_unsupported_device"
    )


def test_topological_order_uses_rust_scheduler_when_aria_core_unavailable(
    monkeypatch,
):
    import research.synthesis.native_topology as native_topology

    calls = {}

    class FakeRustScheduler:
        @staticmethod
        def topological_order(ir_json):
            calls["ir_json"] = ir_json
            return [0, 1, 2]

    monkeypatch.setattr(native_topology, "_try_import_aria_core", lambda: None)
    monkeypatch.setattr(
        native_topology, "_try_import_rust_scheduler", lambda: FakeRustScheduler()
    )
    monkeypatch.setattr(
        native_topology,
        "_graph_to_native_ir_json",
        lambda graph: "fake-native-ir",
    )

    graph = _make_chain_graph()

    assert graph.topological_order() == [0, 1, 2]
    assert calls["ir_json"] == "fake-native-ir"


def test_topological_order_falls_back_to_python_when_native_unavailable(monkeypatch):
    import research.synthesis.native_topology as native_topology

    calls = {"python": 0}
    original_python_topology = native_topology._python_topological_order

    def _tracked_python_topology(graph):
        calls["python"] += 1
        return original_python_topology(graph)

    monkeypatch.setattr(native_topology, "_try_import_aria_core", lambda: None)
    monkeypatch.setattr(native_topology, "_try_import_rust_scheduler", lambda: None)
    monkeypatch.setattr(
        native_topology, "_python_topological_order", _tracked_python_topology
    )

    graph = _make_chain_graph()

    assert graph.topological_order() == [0, 1, 2]
    assert calls["python"] == 1


def test_validate_graph_uses_native_validation_summary_when_available():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    a = graph.add_op("linear_proj", [inp], config={"out_dim": 8})
    b = graph.add_op("linear_proj", [a], config={"out_dim": 8})
    c = graph.add_op("linear_proj", [b], config={"out_dim": 8})
    d = graph.add_op("linear_proj", [c], config={"out_dim": 8})
    graph.set_output(d)

    reset_native_analysis_bindings()
    result = validate_graph(graph)

    assert result.n_parameterized_ops == 4
    assert any("projection chain" in warning.lower() for warning in result.warnings)


def test_validate_graph_validation_summary_falls_back_to_python(monkeypatch):
    import research.synthesis.native_analysis as native_analysis
    import research.synthesis.native_validation as native_validation

    monkeypatch.setattr(
        native_analysis, "_load_native_graph_analysis_lib", lambda: None
    )
    monkeypatch.setattr(
        native_validation, "_load_native_graph_analysis_lib", lambda: None
    )
    reset_native_analysis_bindings()

    graph = ComputationGraph(8)
    inp = graph.add_input()
    a = graph.add_op("linear_proj", [inp], config={"out_dim": 8})
    b = graph.add_op("linear_proj", [a], config={"out_dim": 8})
    c = graph.add_op("linear_proj", [b], config={"out_dim": 8})
    d = graph.add_op("linear_proj", [c], config={"out_dim": 8})
    graph.set_output(d)

    result = validate_graph(graph)

    assert result.n_parameterized_ops == 4
    assert any("projection chain" in warning.lower() for warning in result.warnings)


def test_effective_depth_native_matches_python(monkeypatch):
    from research.synthesis.native_validation import effective_depth_natively
    from research.synthesis.validation_opcode_tables import validation_opcode_tables
    import research.synthesis.validator as validator

    graph = ComputationGraph(8)
    inp = graph.add_input()
    a = graph.add_op("linear_proj", [inp], config={"out_dim": 8})
    b = graph.add_op("rmsnorm", [a])
    c = graph.add_op("linear_proj", [b], config={"out_dim": 8})
    graph.set_output(c)
    ir = graph._analysis_ir()
    tables = validation_opcode_tables()

    native_depth = effective_depth_natively(
        op_codes=ir.op_codes,
        input_indices=ir.input_indices,
        effective_depth_weights=tables.effective_depth_weight,
        discount_successor_u8=tables.discount_successor_u8,
    )
    if native_depth is None:
        pytest.skip("native graph effective-depth runtime unavailable")

    monkeypatch.setattr(
        validator,
        "effective_depth_natively",
        lambda **_: None,
    )
    ir.analysis_cache.clear()
    python_depth = validator.compute_effective_depth(ir)

    assert native_depth == pytest.approx(python_depth)


def test_dead_parameterized_mask_uses_native_runtime(monkeypatch):
    import numpy as np

    import research.synthesis.native_dim_flow as native_dim_flow

    calls = {}

    class FakeLib:
        def aria_graph_dead_parameterized_mask(
            self,
            n_nodes,
            reachable_ptr,
            parameterized_ptr,
            dead_ptr,
        ):
            reachable = np.ctypeslib.as_array(reachable_ptr, shape=(n_nodes,))
            parameterized = np.ctypeslib.as_array(parameterized_ptr, shape=(n_nodes,))
            dead = np.ctypeslib.as_array(dead_ptr, shape=(n_nodes,))
            dead[:] = ((reachable == 0) & (parameterized != 0)).astype(np.int32)
            calls["n_nodes"] = n_nodes
            return 0

    monkeypatch.setattr(
        native_dim_flow,
        "_load_native_graph_analysis_lib",
        lambda: FakeLib(),
    )

    result = native_dim_flow.dead_parameterized_mask(
        reachable_mask=np.array([1, 0, 0, 1], dtype=np.int32),
        parameterized_flags=np.array([0, 1, 0, 1], dtype=np.int32),
    )

    assert calls["n_nodes"] == 4
    assert result.backend == "native"
    assert result.mask.tolist() == [False, True, False, False]


def test_native_wrapper_softmax_attention_matches_python_path():
    import torch

    from research.scientist.native_runner import NativeForwardWrapper
    from research.synthesis.compiler import CompiledOp
    from research.synthesis.graph import ShapeInfo

    shape = ShapeInfo(dim=8)
    native_op = CompiledOp("softmax_attention", {}, shape, shape, 8)
    python_op = CompiledOp("softmax_attention", {}, shape, shape, 8)
    python_op.load_state_dict(native_op.state_dict())
    native_op._native_wrapper = NativeForwardWrapper(None, set())

    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        native_out = native_op(x)
        python_out = python_op(x)

    torch.testing.assert_close(native_out, python_out, atol=1e-6, rtol=1e-6)


def test_native_wrapper_selective_scan_matches_python_path():
    import torch

    from research.scientist.native_runner import NativeForwardWrapper
    from research.synthesis.compiler import CompiledOp
    from research.synthesis.graph import ShapeInfo

    shape = ShapeInfo(dim=8)
    native_op = CompiledOp("selective_scan", {}, shape, shape, 8)
    python_op = CompiledOp("selective_scan", {}, shape, shape, 8)
    python_op.load_state_dict(native_op.state_dict())
    native_op._native_wrapper = NativeForwardWrapper(None, set())

    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        native_out = native_op(x)
        python_out = python_op(x)

    torch.testing.assert_close(native_out, python_out, atol=1e-5, rtol=1e-5)


def test_native_wrapper_state_space_matches_python_path():
    import torch

    from research.scientist.native_runner import NativeForwardWrapper
    from research.synthesis.compiler import CompiledOp
    from research.synthesis.graph import ShapeInfo

    shape = ShapeInfo(dim=8)
    native_op = CompiledOp("state_space", {}, shape, shape, 8)
    python_op = CompiledOp("state_space", {}, shape, shape, 8)
    python_op.load_state_dict(native_op.state_dict())
    native_op._native_wrapper = NativeForwardWrapper(None, set())

    x = torch.randn(2, 4, 8)
    with torch.no_grad():
        native_out = native_op(x)
        python_out = python_op(x)

    torch.testing.assert_close(native_out, python_out, atol=1e-5, rtol=1e-5)


def test_native_wrapper_gated_delta_matches_python_path():
    import torch

    from research.scientist.native_runner import NativeForwardWrapper
    from research.synthesis.compiler import CompiledOp
    from research.synthesis.graph import ShapeInfo

    shape = ShapeInfo(dim=16)
    native_op = CompiledOp("gated_delta", {}, shape, shape, 16)
    python_op = CompiledOp("gated_delta", {}, shape, shape, 16)
    python_op.load_state_dict(native_op.state_dict())
    native_op._gated_delta_heads = 4
    python_op._gated_delta_heads = 4
    native_op._native_wrapper = NativeForwardWrapper(None, set())

    x = torch.randn(2, 37, 16)
    with torch.no_grad():
        native_out = native_op(x)
        python_out = python_op(x)

    torch.testing.assert_close(native_out, python_out, atol=1e-5, rtol=1e-5)


def test_compile_graph_uses_bound_native_subgraph_for_weighted_chain_without_prewarm():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("softmax_attention", [n1])
    n3 = graph.add_op("selective_scan", [n2])
    n4 = graph.add_op("state_space", [n3])
    n5 = graph.add_op("gated_delta", [n4])
    graph.set_output(n5)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 2, 8)
    with torch.no_grad():
        y = layer(x)

    assert y.shape == x.shape
    assert layer._subgraph_dispatcher is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0


def test_compile_graph_uses_bound_native_subgraph_backward_for_weighted_chain():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("softmax_attention", [n1])
    n3 = graph.add_op("selective_scan", [n2])
    n4 = graph.add_op("state_space", [n3])
    n5 = graph.add_op("gated_delta", [n4])
    graph.set_output(n5)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 2, 8, requires_grad=True)
    y = layer(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.ops[str(n1)].weight.grad is not None
    assert layer.ops[str(n2)].q_proj.weight.grad is not None
    assert layer.ops[str(n3)].A_log.grad is not None
    assert layer.ops[str(n4)].ssm_A.grad is not None
    assert layer.ops[str(n5)].q_proj.weight.grad is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_for_rwkv_time_mixing_inference():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    out = graph.add_op("rwkv_time_mixing", [inp])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 3, 8)
    with torch.no_grad():
        y = layer(x)

    assert y.shape == x.shape
    assert layer._subgraph_dispatcher is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0


def test_compile_graph_uses_bound_native_subgraph_backward_for_rwkv_time_mixing():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    out = graph.add_op("rwkv_time_mixing", [inp])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 3, 8, requires_grad=True)
    y = layer(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    op = layer.ops[str(out)]
    assert op.w_decay.grad is not None
    assert op.u_bonus.grad is not None
    assert op.W_k.grad is not None
    assert op.W_v.grad is not None
    assert op.W_r.grad is not None
    assert op.W_o.grad is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_for_transformer_standard():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("layernorm", [inp])
    n2 = graph.add_op("softmax_attention", [n1])
    n3 = graph.add_op("add", [inp, n2])
    n4 = graph.add_op("layernorm", [n3])
    n5 = graph.add_op("linear_proj", [n4], {"out_dim": 32})
    n6 = graph.add_op("gelu", [n5])
    n7 = graph.add_op("linear_proj", [n6], {"out_dim": 8})
    out = graph.add_op("add", [n3, n7])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 3, 8)
    with torch.no_grad():
        y = layer(x)

    assert y.shape == x.shape
    assert layer._subgraph_dispatcher is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0


def test_compile_graph_uses_bound_native_subgraph_backward_for_transformer_standard():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("layernorm", [inp])
    n2 = graph.add_op("softmax_attention", [n1])
    n3 = graph.add_op("add", [inp, n2])
    n4 = graph.add_op("layernorm", [n3])
    n5 = graph.add_op("linear_proj", [n4], {"out_dim": 32})
    n6 = graph.add_op("gelu", [n5])
    n7 = graph.add_op("linear_proj", [n6], {"out_dim": 8})
    out = graph.add_op("add", [n3, n7])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 3, 8, requires_grad=True)
    y = layer(x)
    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.ops[str(n1)].weight.grad is not None
    assert layer.ops[str(n1)].bias.grad is not None
    assert layer.ops[str(n2)].q_proj.weight.grad is not None
    assert layer.ops[str(n4)].weight.grad is not None
    assert layer.ops[str(n4)].bias.grad is not None
    assert layer.ops[str(n5)].weight.grad is not None
    assert layer.ops[str(n7)].weight.grad is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_for_ssm_rwkv_inference():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("layernorm", [inp])
    n2 = graph.add_op("rwkv_time_mixing", [n1])
    n3 = graph.add_op("add", [inp, n2])
    n4 = graph.add_op("layernorm", [n3])
    n5 = graph.add_op("rwkv_channel", [n4])
    out = graph.add_op("add", [n3, n5])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 3, 8)
    with torch.no_grad():
        y = layer(x)

    assert y.shape == x.shape
    assert layer._subgraph_dispatcher is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_backward_for_ssm_rwkv():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("layernorm", [inp])
    n2 = graph.add_op("rwkv_time_mixing", [n1])
    n3 = graph.add_op("add", [inp, n2])
    n4 = graph.add_op("layernorm", [n3])
    n5 = graph.add_op("rwkv_channel", [n4])
    out = graph.add_op("add", [n3, n5])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 3, 8, requires_grad=True)
    y = layer(x)
    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.ops[str(n1)].weight.grad is not None
    assert layer.ops[str(n1)].bias.grad is not None
    assert layer.ops[str(n2)].W_o.grad is not None
    assert layer.ops[str(n5)].mix_k.grad is not None
    assert layer.ops[str(n5)].mix_r.grad is not None
    assert layer.ops[str(n5)].key_proj.weight.grad is not None
    assert layer.ops[str(n5)].receptance_proj.weight.grad is not None
    assert layer.ops[str(n5)].value_proj.weight.grad is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_for_swiglu_inference():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("softmax_attention", [n1])
    n3 = graph.add_op("add", [inp, n2])
    n4 = graph.add_op("rmsnorm", [n3])
    n5 = graph.add_op("swiglu_mlp", [n4])
    out = graph.add_op("add", [n3, n5])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 3, 8)
    with torch.no_grad():
        y = layer(x)

    assert y.shape == x.shape
    assert layer._subgraph_dispatcher is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_backward_for_swiglu():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("softmax_attention", [n1])
    n3 = graph.add_op("add", [inp, n2])
    n4 = graph.add_op("rmsnorm", [n3])
    n5 = graph.add_op("swiglu_mlp", [n4])
    out = graph.add_op("add", [n3, n5])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 3, 8, requires_grad=True)
    y = layer(x)
    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.ops[str(n1)].weight.grad is not None
    assert layer.ops[str(n2)].q_proj.weight.grad is not None
    assert layer.ops[str(n4)].weight.grad is not None
    assert layer.ops[str(n5)].gate_proj.weight.grad is not None
    assert layer.ops[str(n5)].up_proj.weight.grad is not None
    assert layer.ops[str(n5)].down_proj.weight.grad is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_for_linear_attention_inference():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("linear_attention", [n1])
    out = graph.add_op("add", [inp, n2])
    graph.set_output(out)

    layer = compile_graph(graph, use_ir=True)
    assert layer._subgraph_dispatcher is not None

    with torch.no_grad():
        y = layer(torch.randn(1, 3, 8))

    assert y.shape == (1, 3, 8)
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is False


def test_compile_graph_uses_bound_native_subgraph_for_adaptive_recursion_inference():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("adaptive_recursion", [n1], {"max_depth": 3})
    out = graph.add_op("add", [inp, n2])
    graph.set_output(out)

    layer = compile_graph(graph, use_ir=True)
    assert layer._subgraph_dispatcher is not None

    with torch.no_grad():
        y = layer(torch.randn(1, 3, 8))

    assert y.shape == (1, 3, 8)
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is False


def test_compile_graph_falls_back_cleanly_for_adaptive_recursion_backward():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("adaptive_recursion", [n1], {"max_depth": 3})
    out = graph.add_op("add", [inp, n2])
    graph.set_output(out)

    layer = compile_graph(graph, use_ir=True)
    x = torch.randn(1, 3, 8, requires_grad=True)
    y = layer(x)
    y.sum().backward()

    assert x.grad is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 0
    assert stats["subgraph_fallbacks"] == 1
    assert stats["native_backward_supported"] is False
    assert stats["last_refusal_reason"] == "native_backward_unavailable"


def test_compile_graph_uses_bound_native_subgraph_for_token_routing_inference():
    graph, _, _ = graph_token_routing()
    layer = compile_graph(graph, use_ir=True)
    assert layer._subgraph_dispatcher is not None

    with torch.no_grad():
        y = layer(torch.randn(1, 3, graph.model_dim))

    assert y.shape == (1, 3, graph.model_dim)
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is False


def test_compile_graph_uses_bound_native_subgraph_for_ssm_state_space_inference():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("state_space", [n1])
    n3 = graph.add_op("add", [inp, n2])
    n4 = graph.add_op("rmsnorm", [n3])
    n5 = graph.add_op("conv_only", [n4])
    out = graph.add_op("add", [n3, n5])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 3, 8)
    with torch.no_grad():
        y = layer(x)

    assert y.shape == x.shape
    assert layer._subgraph_dispatcher is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_backward_for_ssm_state_space():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("state_space", [n1])
    n3 = graph.add_op("add", [inp, n2])
    n4 = graph.add_op("rmsnorm", [n3])
    n5 = graph.add_op("conv_only", [n4])
    out = graph.add_op("add", [n3, n5])
    graph.set_output(out)

    layer = compile_model([graph], vocab_size=16, max_seq_len=4).layers[0]
    x = torch.randn(1, 3, 8, requires_grad=True)
    y = layer(x)
    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.ops[str(n2)].ssm_A.grad is not None
    assert layer.ops[str(n5)].conv_dw.weight.grad is not None
    assert layer.ops[str(n5)].conv_proj.weight.grad is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_ir_executor_skips_plain_subgraph_dispatch_for_mixed_parameterized_graph():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("split2", [n1], {"n_splits": 2})
    n3 = graph.add_op("split2", [n1], {"n_splits": 2})
    n4 = graph.add_op("gelu", [n2])
    n5 = graph.add_op("tanh", [n3])
    n6 = graph.add_op("concat", [n4, n5])
    out = graph.add_op("add", [inp, n6])
    graph.set_output(out)

    layer = compile_graph(graph, use_ir=True)
    assert getattr(layer, "_subgraph_dispatcher", None) is None

    x = torch.randn(1, 3, 8)
    with torch.no_grad():
        y = layer(x)

    assert y.shape == x.shape
    assert layer.execution_stats["native_subgraph_available"] is False
    assert layer.execution_stats["partial_native_available"] is True


def test_compile_graph_uses_bound_native_subgraph_for_ssm_mamba_inference():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("conv1d_seq", [n1])
    n3 = graph.add_op("silu", [n2])
    n4 = graph.add_op("selective_scan", [n3])
    n5 = graph.add_op("gated_linear", [n4], {"out_dim": 8})
    out = graph.add_op("add", [inp, n5])
    graph.set_output(out)

    layer = compile_graph(graph, use_ir=True)
    assert layer._subgraph_dispatcher is not None

    with torch.no_grad():
        y = layer(torch.randn(1, 3, 8))

    assert y.shape == (1, 3, 8)
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_backward_for_ssm_mamba():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("conv1d_seq", [n1])
    n3 = graph.add_op("silu", [n2])
    n4 = graph.add_op("selective_scan", [n3])
    n5 = graph.add_op("gated_linear", [n4], {"out_dim": 8})
    out = graph.add_op("add", [inp, n5])
    graph.set_output(out)

    layer = compile_graph(graph, use_ir=True)
    x = torch.randn(1, 3, 8, requires_grad=True)
    y = layer(x)
    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.ops[str(n1)].weight.grad is not None
    assert layer.ops[str(n2)].conv_weight.grad is not None
    assert layer.ops[str(n4)].A_log.grad is not None
    assert layer.ops[str(n5)].linear_weight.grad is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_for_hybrid_ssm_attention_inference():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("softmax_attention", [n1])
    n3 = graph.add_op("add", [inp, n2])
    n4 = graph.add_op("rmsnorm", [n3])
    n5 = graph.add_op("conv1d_seq", [n4])
    n6 = graph.add_op("selective_scan", [n5])
    n7 = graph.add_op("add", [n3, n6])
    n8 = graph.add_op("rmsnorm", [n7])
    n9 = graph.add_op("swiglu_mlp", [n8])
    out = graph.add_op("add", [n7, n9])
    graph.set_output(out)

    layer = compile_graph(graph, use_ir=True)
    assert layer._subgraph_dispatcher is not None

    with torch.no_grad():
        y = layer(torch.randn(1, 3, 8))

    assert y.shape == (1, 3, 8)
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True


def test_compile_graph_uses_bound_native_subgraph_backward_for_hybrid_ssm_attention():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    n1 = graph.add_op("rmsnorm", [inp])
    n2 = graph.add_op("softmax_attention", [n1])
    n3 = graph.add_op("add", [inp, n2])
    n4 = graph.add_op("rmsnorm", [n3])
    n5 = graph.add_op("conv1d_seq", [n4])
    n6 = graph.add_op("selective_scan", [n5])
    n7 = graph.add_op("add", [n3, n6])
    n8 = graph.add_op("rmsnorm", [n7])
    n9 = graph.add_op("swiglu_mlp", [n8])
    out = graph.add_op("add", [n7, n9])
    graph.set_output(out)

    layer = compile_graph(graph, use_ir=True)
    x = torch.randn(1, 3, 8, requires_grad=True)
    y = layer(x)
    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert layer.ops[str(n2)].q_proj.weight.grad is not None
    assert layer.ops[str(n5)].conv_weight.grad is not None
    assert layer.ops[str(n6)].A_log.grad is not None
    assert layer.ops[str(n9)].gate_proj.weight.grad is not None
    stats = layer._subgraph_dispatcher.stats
    assert stats["subgraph_dispatches"] == 1
    assert stats["subgraph_fallbacks"] == 0
    assert stats["native_backward_supported"] is True
