from __future__ import annotations

import numpy as np

from research.synthesis.dim_flow_support import build_dim_flow_inputs
from research.synthesis.graph import ComputationGraph
from research.synthesis.native_dim_flow_flags import build_dim_flow_flags_natively


def test_build_dim_flow_flags_natively_uses_runtime(monkeypatch):
    calls = {}

    class FakeLib:
        def aria_graph_build_dim_flow_flags(
            self,
            n_nodes,
            op_codes_ptr,
            param_estimates_ptr,
            opcode_has_params_ptr,
            opcode_nontrivial_ptr,
            opcode_kv_breaking_ptr,
            opcode_kind_ptr,
            opcode_full_dim_ptr,
            has_params_ptr,
            nontrivial_ptr,
            kv_breaking_ptr,
            op_kind_ptr,
            full_dim_ptr,
        ):
            calls["n_nodes"] = n_nodes
            np.ctypeslib.as_array(has_params_ptr, shape=(n_nodes,))[:] = [0, 1]
            np.ctypeslib.as_array(nontrivial_ptr, shape=(n_nodes,))[:] = [0, 1]
            np.ctypeslib.as_array(kv_breaking_ptr, shape=(n_nodes,))[:] = [0, 0]
            np.ctypeslib.as_array(op_kind_ptr, shape=(n_nodes,))[:] = [0, 3]
            np.ctypeslib.as_array(full_dim_ptr, shape=(n_nodes,))[:] = [0, 1]
            return 0

    monkeypatch.setattr(
        "research.synthesis.native_dim_flow_flags.load_native_graph_analysis_lib",
        lambda: FakeLib(),
    )

    result = build_dim_flow_flags_natively(
        op_codes=np.array([0, 7], dtype=np.int32),
        param_estimates=np.array([0, 64], dtype=np.int64),
        opcode_has_params=np.array([0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
        opcode_nontrivial=np.array([0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
        opcode_kv_breaking=np.zeros(8, dtype=np.int32),
        opcode_kind=np.zeros(8, dtype=np.int32),
        opcode_full_dim=np.zeros(8, dtype=np.int32),
    )

    assert calls["n_nodes"] == 2
    assert result is not None
    assert result["has_params_flags"].tolist() == [0, 1]
    assert result["op_kind_flags"].tolist() == [0, 3]


def test_build_dim_flow_inputs_reuses_ir_param_estimates(monkeypatch):
    graph = ComputationGraph(8)
    inp = graph.add_input()
    proj = graph.add_op("linear_proj", [inp], config={"out_dim": 8})
    graph.set_output(proj)

    analysis_ir = graph._analysis_ir()
    analysis_ir.param_estimates = np.array([0, 123], dtype=np.int64)
    monkeypatch.setattr(graph, "_analysis_ir", lambda: analysis_ir)
    monkeypatch.setattr(
        "research.synthesis.dim_flow_support.build_dim_flow_flags_natively",
        lambda **kwargs: None,
    )

    result = build_dim_flow_inputs(
        graph,
        op_kind_default=0,
        op_kind_irfft=1,
        op_kind_identity=2,
        op_kind_binary_broadcast=3,
    )

    assert result.param_estimates.tolist() == [0, 123]
    assert result.has_params_flags.tolist() == [0, 1]
