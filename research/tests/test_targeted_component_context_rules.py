from __future__ import annotations


from research.synthesis.graph import ComputationGraph
from research.synthesis.validator import validate_graph


def test_depth_token_mask_requires_post_mask_redensifying_context():
    g = ComputationGraph(64)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    masked = g.add_op("depth_token_mask", [norm], {"capacity_factor": 0.75})
    out = g.add_op("add", [inp, masked])
    g.set_output(out)

    result = validate_graph(g)

    assert not result.valid
    assert any(
        "depth_token_mask requires immediate projection/norm successor" in err
        for err in result.errors
    )


def test_depth_token_mask_valid_inside_residual_refinement_block():
    g = ComputationGraph(64)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    masked = g.add_op("depth_token_mask", [norm], {"capacity_factor": 0.75})
    proj = g.add_op("linear_proj", [masked], {"out_dim": 64})
    out = g.add_op("add", [inp, proj])
    g.set_output(out)

    result = validate_graph(g)

    assert result.valid, result.errors


def test_grade_mix_requires_clifford_context():
    g = ComputationGraph(64)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    mixed = g.add_op("grade_mix", [norm])
    proj = g.add_op("linear_proj", [mixed], {"out_dim": 64})
    out = g.add_op("add", [inp, proj])
    g.set_output(out)

    result = validate_graph(g)

    assert not result.valid
    assert any(
        "grade_mix requires Clifford predecessor context" in err
        for err in result.errors
    )


def test_grade_mix_accepts_geometric_product_context():
    g = ComputationGraph(64)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    gp = g.add_op("geometric_product", [norm, norm])
    mixed = g.add_op("grade_mix", [gp])
    proj = g.add_op("linear_proj", [mixed], {"out_dim": 64})
    out = g.add_op("add", [inp, proj])
    g.set_output(out)

    result = validate_graph(g)

    assert result.valid, result.errors


def test_confidence_token_gate_requires_residual_recovery():
    g = ComputationGraph(64)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    gated = g.add_op("confidence_token_gate", [norm], {"threshold": 0.5})
    g.set_output(gated)

    result = validate_graph(g)

    assert not result.valid
    assert any(
        "confidence_token_gate must sit inside a residual/routing block" in err
        for err in result.errors
    )
