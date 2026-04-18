from research.eval.flops import estimate_flops
from research.synthesis.graph import ComputationGraph
from research.synthesis.primitives import estimate_op_params, get_primitive


def _single_op_graph(op_name: str, model_dim: int = 32) -> ComputationGraph:
    graph = ComputationGraph(model_dim=model_dim)
    inp = graph.add_input()
    out = graph.add_op(op_name, [inp])
    graph.set_output(out)
    return graph


def test_new_mixing_op_param_formulas_match_initialized_modules():
    expected = {
        "difficulty_routed_attention": 32 * 32 * 5 + 32 + 1,
        "strided_attention": 32 * 32 * 4,
        "gated_progressive_attention": 32 * 32 * 5 + 32,
        "gated_linear_attention": 32 * 32 * 5,
        "long_conv_hyena": 32 * 32 * 3 + 32 * 33 + 64,
        "associative_memory": 32 * 32 * 4 + 1,
        "mixture_of_recursions": 32 * 32 * 6 + 32 * 6 + 4,
    }
    for op_name, n_params in expected.items():
        assert estimate_op_params(get_primitive(op_name), 32) == n_params


def test_new_mixing_op_flop_estimates_reflect_actual_kernel_shapes():
    seq_len = 128
    d_model = 32
    estimates = {
        op_name: estimate_flops(_single_op_graph(op_name, model_dim=d_model), seq_len=seq_len, d_model=d_model).flops_forward
        for op_name in (
            "difficulty_routed_attention",
            "strided_attention",
            "gated_progressive_attention",
            "gated_linear_attention",
            "long_conv_hyena",
            "associative_memory",
            "mixture_of_recursions",
        )
    }

    assert estimates["difficulty_routed_attention"] > estimates["gated_linear_attention"]
    assert estimates["gated_progressive_attention"] > estimates["gated_linear_attention"]
    assert estimates["associative_memory"] > estimates["gated_linear_attention"]
    assert estimates["strided_attention"] < estimates["difficulty_routed_attention"]
    assert estimates["long_conv_hyena"] > 0
    assert estimates["mixture_of_recursions"] > estimates["gated_linear_attention"]
