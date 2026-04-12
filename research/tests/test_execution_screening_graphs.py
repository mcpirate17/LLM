from types import SimpleNamespace

from research.scientist.runner.execution_screening_graphs import (
    analyze_graph_for_screening,
    structural_gate_failure,
    toxic_failure_ratio,
)
from research.scientist.runner._helpers import (
    graph_observed_routing_ops,
    graph_routing_ops,
)


def _node(op_name, input_ids=(), *, is_input=False, is_output=False):
    return SimpleNamespace(
        op_name=op_name,
        input_ids=list(input_ids),
        is_input=is_input,
        is_output=is_output,
    )


def _graph(nodes, *, n_ops=8, has_gradient_path=True, has_residual_path=True):
    return SimpleNamespace(
        nodes=nodes,
        n_ops=lambda: n_ops,
        has_gradient_path=lambda: has_gradient_path,
        has_residual_path=lambda: has_residual_path,
    )


def test_analyze_graph_for_screening_collects_reusable_facts():
    graph = _graph(
        {
            0: _node("input", is_input=True),
            1: _node("linear_proj", [0]),
            2: _node("moe_topk", [1]),
            3: _node("output", [2], is_output=True),
        }
    )
    primitives = {
        "linear_proj": SimpleNamespace(has_params=True),
        "moe_topk": SimpleNamespace(has_params=False),
    }

    analysis = analyze_graph_for_screening(graph, primitives.get)

    assert analysis.has_parameterized_op is True
    assert analysis.op_names == frozenset({"linear_proj", "moe_topk"})
    assert analysis.counted_ops == ("linear_proj", "moe_topk", "output")
    assert analysis.toxic_bigrams == ("linear_proj->moe_topk",)


def test_structural_gate_failure_uses_cached_analysis():
    graph = _graph(
        {
            0: _node("input", is_input=True),
            1: _node("relu", [0]),
            2: _node("output", [1], is_output=True),
        },
        has_gradient_path=True,
        has_residual_path=True,
    )
    analysis = analyze_graph_for_screening(graph, lambda _: None)

    failure = structural_gate_failure(
        graph,
        routing_mandatory=True,
        efficiency_ops=frozenset({"moe_topk"}),
        analysis=analysis,
    )

    assert failure == "gate4_no_params"


def test_toxic_failure_ratio_reuses_cached_bigrams():
    graph = _graph(
        {
            0: _node("input", is_input=True),
            1: _node("linear_proj", [0]),
            2: _node("relu", [1]),
            3: _node("moe_topk", [2]),
            4: _node("output", [3], is_output=True),
        }
    )
    analysis = analyze_graph_for_screening(
        graph,
        {
            "linear_proj": SimpleNamespace(has_params=True),
            "relu": SimpleNamespace(has_params=False),
            "moe_topk": SimpleNamespace(has_params=False),
        }.get,
    )

    ratio = toxic_failure_ratio(
        {
            "linear_proj->relu": 0.25,
            "relu->moe_topk": 0.75,
        },
        analysis,
    )

    assert ratio == 0.5


def test_graph_routing_ops_reports_trigger_ops_and_observed_ops_separately():
    graph = _graph(
        {
            0: _node("input", is_input=True),
            1: _node("hybrid_token_gate", [0]),
            2: _node("hybrid_sparse_router", [1]),
            3: _node("signal_conditioned_compression", [2]),
            4: _node("adjacent_token_merge", [3]),
            5: _node("output", [4], is_output=True),
        }
    )

    assert graph_routing_ops(graph) == ["signal_conditioned_compression"]
    assert graph_observed_routing_ops(graph) == [
        "adjacent_token_merge",
        "hybrid_sparse_router",
        "hybrid_token_gate",
        "signal_conditioned_compression",
    ]
