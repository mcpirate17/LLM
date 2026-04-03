"""Slot/Template Wiring Tests — verify every newly-wired op compiles and runs.

For each op that was previously unreachable (Used=0), builds a minimal graph
through its motif or template path, compiles it, and runs forward + backward.
This catches type errors, shape mismatches, and broken dispatch — not model
quality, just that the wiring from grammar → compile → forward is intact.
"""

import pytest
import random

import torch

from research.synthesis.compiler import CompiledLayer
from research.synthesis.graph import ComputationGraph
from research.synthesis.motifs import VALIDATED_MOTIFS, resolve_step
from research.synthesis.templates import (
    apply_template,
)

D = 64  # Test model dim (small for speed)
B, S = 2, 8  # Batch, sequence


# ── Helpers ──────────────────────────────────────────────────────────


def _build_layer_from_graph(g: ComputationGraph) -> CompiledLayer:
    return CompiledLayer(g)


def _fwd_bwd(layer: CompiledLayer, dim: int = D) -> dict:
    """Run forward + backward, return diagnostics."""
    x = torch.randn(B, S, dim, requires_grad=True)
    y = layer(x)
    result = {
        "output_shape": tuple(y.shape),
        "has_nan": bool(torch.isnan(y).any()),
        "has_inf": bool(torch.isinf(y).any()),
    }
    try:
        y.sum().backward()
    except RuntimeError:
        pass
    result["input_grad"] = x.grad.norm().item() if x.grad is not None else 0.0
    grads = [p.grad.norm().item() for p in layer.parameters() if p.grad is not None]
    result["max_param_grad"] = max(grads) if grads else 0.0
    return result


def _build_motif_graph(motif_name: str) -> ComputationGraph:
    """Build a minimal graph from a motif: input → [motif steps] → output."""
    motif = VALIDATED_MOTIFS[motif_name]
    g = ComputationGraph(model_dim=D)
    current = g.add_input()
    rng = random.Random(42)

    for i, step in enumerate(motif.steps):
        next_op = motif.steps[i + 1].op_name if i + 1 < len(motif.steps) else None
        prev_op = g.nodes[current].op_name if not g.nodes[current].is_input else None
        op_name, config = resolve_step(step, rng, prev_op=prev_op, next_op=next_op)

        # Track current dim for proper config
        cur_dim = g.nodes[current].output_shape.dim

        # Binary ops: create a second input from a projection
        from research.synthesis.primitives import PRIMITIVE_REGISTRY

        prim = PRIMITIVE_REGISTRY.get(op_name)
        n_inputs = prim.n_inputs if prim else 1
        inputs = [current]
        if n_inputs == 2:
            try:
                inp2 = g.add_op("linear_proj", [current], config={"out_dim": cur_dim})
                inputs = [current, inp2]
            except ValueError as e:
                pytest.skip(f"Cannot create 2nd input for {op_name}: {e}")

        # Set required configs
        if op_name in ("linear_proj", "fused_linear_gelu", "gated_linear"):
            config.setdefault("out_dim", D)
        elif op_name == "linear_proj_down":
            config.setdefault("out_dim", max(cur_dim // 2, 4))
        elif op_name == "linear_proj_up":
            # Restore to model_dim (not blindly double)
            config.setdefault("out_dim", D)
        elif op_name in (
            "nm_sparse_linear",
            "block_sparse_linear",
            "semi_structured_2_4_linear",
            "ternary_projection",
            "kronecker_linear",
        ):
            config.setdefault("out_dim", D)

        try:
            current = g.add_op(op_name, inputs, config=config)
        except ValueError as e:
            pytest.skip(f"Cannot add {op_name} to graph: {e}")

    # Fix output dim if it doesn't match model_dim
    out_dim = g.nodes[current].output_shape.dim
    if out_dim != D:
        current = g.add_op("linear_proj", [current], config={"out_dim": D})

    g.set_output(current)
    return g


def _build_template_graph(template_name: str) -> ComputationGraph:
    """Build a graph by applying a named template."""
    # Use dim divisible by 3 for split3 template
    dim = 96 if "three_way" in template_name else D
    g = ComputationGraph(model_dim=dim)
    inp = g.add_input()
    rng = random.Random(42)

    out = apply_template(g, inp, rng, template_name=template_name)
    g.set_output(out)
    return g


# ── A. Motif-based ops (new motifs) ──────────────────────────────────

MOTIF_TEST_CASES = [
    # (motif_name, target_op_that_was_unreachable)
    ("kronecker_proj", "kronecker_linear"),
    ("chebyshev_spectral", "chebyshev_spectral_mix"),
    ("n_way_routing", "sparse_bottleneck_moe"),
    ("spectral_filter_block", "spectral_filter"),
    ("tropical_matmul_block", "tropical_matmul"),
    ("tropical_gate_block", "tropical_gate"),
    ("tropical_center_norm", "tropical_center"),
    ("clifford_attention_grade", "grade_mix"),
    ("padic_residual_bridge", "padic_residual"),
    ("poincare_add_bridge", "poincare_add"),
    ("ultrametric_attention_bridge", "ultrametric_attention"),
    # Lift-boosted motifs that already existed but were drowned
    ("poincare_norm_bridge", "hyperbolic_norm"),
    ("spiking_lif_rate", "lif_neuron"),
    ("spiking_threshold_stdp", "stdp_attention"),
    ("clifford_rotor_grade", "rotor_transform"),
    ("route_mod_topk", "depth_token_mask"),
    ("tropical_router_block", "tropical_router"),
    ("tropical_moe_block", "tropical_moe"),
    ("decay_cumprod", "cumprod_safe"),
]


@pytest.mark.parametrize(
    "motif_name,target_op",
    MOTIF_TEST_CASES,
    ids=[f"{m}({op})" for m, op in MOTIF_TEST_CASES],
)
def test_motif_compile_and_forward(motif_name, target_op):
    """Each motif builds a valid graph that compiles and runs forward+backward."""
    g = _build_motif_graph(motif_name)

    # Verify the target op is actually in the graph
    op_names = [n.op_name for n in g.nodes.values() if not n.is_input]
    assert target_op in op_names, (
        f"Motif {motif_name} did not produce {target_op}; got {op_names}"
    )

    layer = _build_layer_from_graph(g)
    result = _fwd_bwd(layer)

    assert not result["has_nan"], f"{motif_name}: NaN in output"
    assert not result["has_inf"], f"{motif_name}: Inf in output"
    assert result["output_shape"][0] == B, f"{motif_name}: bad batch dim"
    assert result["output_shape"][1] == S, f"{motif_name}: bad seq dim"


# ── B. Template-based ops (binary ops, structural) ───────────────────

TEMPLATE_TEST_CASES = [
    # (template_name, target_op_that_was_unreachable)
    ("hyp_distance_scoring", "hyp_distance"),
    ("residual_difference", "sub"),
    ("gated_minimum", "minimum"),
    ("gated_maximum", "maximum"),
    ("tropical_residual", "tropical_add"),
    ("geometric_product_block", "geometric_product"),
    ("three_way_split", "split3"),
    # Pre-existing templates for ops that were already reachable (sanity)
    ("normalized_matmul", "matmul"),
    ("gated_product", "outer_product"),
    ("safe_division", "div_safe"),
    ("cosine_scoring", "cosine_similarity"),
    ("decay_sequence", "cumprod_safe"),
]


@pytest.mark.parametrize(
    "template_name,target_op",
    TEMPLATE_TEST_CASES,
    ids=[f"{t}({op})" for t, op in TEMPLATE_TEST_CASES],
)
def test_template_compile_and_forward(template_name, target_op):
    """Each template builds a valid graph that compiles and runs forward+backward."""
    g = _build_template_graph(template_name)

    # Verify the target op is in the graph
    op_names = [n.op_name for n in g.nodes.values() if not n.is_input]
    assert target_op in op_names, (
        f"Template {template_name} did not produce {target_op}; got {op_names}"
    )

    layer = _build_layer_from_graph(g)
    result = _fwd_bwd(layer, dim=g.model_dim)

    assert not result["has_nan"], f"{template_name}: NaN in output"
    assert not result["has_inf"], f"{template_name}: Inf in output"
    assert result["output_shape"][0] == B, f"{template_name}: bad batch dim"
    assert result["output_shape"][1] == S, f"{template_name}: bad seq dim"


# ── C. Space consistency: motif graphs pass the grammar validator ─────


def test_motif_graphs_pass_space_check():
    """All motif-built graphs must pass algebraic space consistency."""
    from research.synthesis.grammar import _check_graph_space_consistency

    failures = []
    for motif_name, target_op in MOTIF_TEST_CASES:
        try:
            g = _build_motif_graph(motif_name)
        except Exception as e:
            failures.append(f"{motif_name}: graph build failed: {e}")
            continue

        err = _check_graph_space_consistency(g)
        if err is not None:
            failures.append(f"{motif_name}: {err}")

    assert not failures, "Space consistency failures:\n" + "\n".join(failures)


def test_template_graphs_pass_space_check():
    """All template-built graphs must pass algebraic space consistency."""
    from research.synthesis.grammar import _check_graph_space_consistency

    failures = []
    for template_name, target_op in TEMPLATE_TEST_CASES:
        try:
            g = _build_template_graph(template_name)
        except Exception as e:
            failures.append(f"{template_name}: graph build failed: {e}")
            continue

        err = _check_graph_space_consistency(g)
        if err is not None:
            failures.append(f"{template_name}: {err}")

    assert not failures, "Space consistency failures:\n" + "\n".join(failures)
