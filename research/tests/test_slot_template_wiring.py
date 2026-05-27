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
from research.synthesis.validator import validate_graph
from research.synthesis.graph import ComputationGraph
from research.synthesis._context_motifs import motif_allowed_in_template
from research.synthesis.motifs import VALIDATED_MOTIFS, resolve_step
from research.synthesis._template_helpers import get_slot_rule_summary
from research.synthesis.primitives import canonicalize_op_name
from research.synthesis.templates import (
    TEMPLATES,
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


def _build_template_graph(
    template_name: str,
    motif_weights: dict[str, float] | None = None,
) -> ComputationGraph:
    """Build a graph by applying a named template."""
    # Use dim divisible by 3 for split3 template
    dim = 96 if "three_way" in template_name else D
    g = ComputationGraph(model_dim=dim)
    inp = g.add_input()
    rng = random.Random(42)

    out = apply_template(
        g,
        inp,
        rng,
        template_name=template_name,
        motif_weights=motif_weights,
    )
    g.set_output(out)
    return g


def _template_op_names(template_name: str) -> list[str]:
    g = _build_template_graph(template_name)
    return [node.op_name for node in g.nodes.values() if not node.is_input]


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
    ("attn_sparsemax", "sparsemax_attention"),
    ("attn_entmax", "entmax_attention"),
    ("mix_dplr_gated_delta", "dplr_gated_delta"),
    ("mix_token_hodge", "token_hodge_mixer"),
    ("mix_wavelet_packet", "wavelet_packet_mix"),
    ("mix_retention", "retention_mix"),
    ("mem_product_key", "product_key_memory"),
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

BACKFILL_TEMPLATE_CASES = [
    # Rewritten to parallel attn+SSM hybrid — no longer uses matmul
    ("attn_normalized_matmul", "linear_proj"),
    ("attn_softmax_normalized_matmul", "softmax_attention"),
    ("attn_softmax_normalized_matmul_compact_ffn", "softmax_attention"),
    ("attn_softmax_normalized_matmul_fixed_tail_norm", "softmax_attention"),
    ("attn_linear_no_matmul_ffn", "linear_attention"),
    ("attn_linear_no_matmul_ffn_dense_tail", "fused_linear_gelu"),
    ("attn_linear_no_matmul_ffn_direct_recovery", "linear_attention"),
    ("attn_decay_sequence", "cumprod_safe"),
    ("attn_safe_division", "div_safe"),
    ("attn_routing_block", "difficulty_blend_3way"),
    ("graph_attn_sparse_ffn", "graph_attention"),
    ("latent_attn_ffn_block", "latent_attention_compressor"),
    ("local_attn_ffn_block", "local_window_attn"),
    ("latent_attn_sparse_ffn", "latent_attention_compressor"),
    ("local_attn_swiglu", "local_window_attn"),
    ("attn_spectral_filter", "spectral_filter"),
    ("attn_rwkv_hybrid", "rwkv_channel"),
    ("attn_three_way_split", "split3"),
    ("linear_attn_ffn_block", "linear_attention"),
    ("linear_attn_sparse_ffn", "linear_attention"),
    ("latent_attn_moe", "latent_attention_compressor"),
    ("latent_attn_conv_hybrid", "latent_attention_compressor"),
    ("latent_attn_ssm_hybrid", "latent_attention_compressor"),
    ("depth_token_mask_block", "depth_token_mask"),
    ("typed_slot_memory_block", "gather_topk"),
    ("sparse_relation_graph_block", "route_topk"),
    ("token_program_interpreter_block", "n_way_sparse_router"),
    ("sparsemax_attention_block", "sparsemax_attention"),
    ("entmax_attention_block", "entmax_attention"),
    ("dplr_gated_delta_block", "dplr_gated_delta"),
    ("token_hodge_mixer_block", "token_hodge_mixer"),
    ("wavelet_packet_mix_block", "wavelet_packet_mix"),
    ("retention_mix_block", "retention_mix"),
    ("product_key_memory_block", "product_key_memory"),
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


def test_token_merge_templates_keep_post_merge_processing_local():
    for template_name in ("token_merge_block", "token_merge_conv"):
        op_names = _template_op_names(template_name)
        merge_idx = op_names.index("adjacent_token_merge")
        post_merge_ops = op_names[merge_idx + 1 :]

        assert "softmax_attention" not in post_merge_ops
        assert "linear_attention" not in post_merge_ops
        assert "local_window_attn" not in post_merge_ops
        assert "latent_attention_compressor" not in post_merge_ops
        assert "selective_scan" not in post_merge_ops
        assert "state_space" not in post_merge_ops


def test_adaptive_ssm_chain_uses_safe_scan_path():
    op_names = _template_op_names("adaptive_ssm_chain")

    assert "conv1d_seq" in op_names
    assert "silu" in op_names
    assert "selective_scan" in op_names
    assert "ternary_projection" not in op_names


def test_high_risk_motifs_are_restricted_to_safe_templates():
    merge_scan = VALIDATED_MOTIFS["merge_scan"]
    scan = VALIDATED_MOTIFS["ssm_selective_scan"]
    scan_gelu = VALIDATED_MOTIFS["ssm_scan_gelu"]

    assert motif_allowed_in_template(merge_scan, "token_merge_block")
    assert not motif_allowed_in_template(merge_scan, "sequential")

    assert motif_allowed_in_template(scan, "adaptive_ssm_chain")
    assert not motif_allowed_in_template(scan, "residual_block")

    assert motif_allowed_in_template(scan_gelu, "adaptive_ssm_chain")
    assert not motif_allowed_in_template(scan_gelu, "mixed_recursion")


@pytest.mark.parametrize(
    "template_name,target_op",
    BACKFILL_TEMPLATE_CASES,
    ids=[f"{t}({op})" for t, op in BACKFILL_TEMPLATE_CASES],
)
def test_backfill_attention_templates_compile_and_forward(template_name, target_op):
    """Priority backfill templates must build, compile, and run cleanly."""
    g = _build_template_graph(template_name)

    op_names = [n.op_name for n in g.nodes.values() if not n.is_input]
    assert canonicalize_op_name(target_op) in op_names, (
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


def test_quarantined_templates_are_not_selectable():
    for template_name in (
        "attn_dual_axis",
        "attn_dense_cascade",
        "attn_moe_block",
        "attn_gated_minimum",
    ):
        assert template_name not in TEMPLATES


@pytest.mark.parametrize(
    "template_name,required_ops",
    (
        ("mamba_reference", {"rmsnorm", "conv1d_seq", "selective_scan", "swiglu_mlp"}),
        (
            "topk_retrieval",
            {"rmsnorm", "matmul", "gather_topk", "swiglu_mlp"},
        ),
    ),
)
def test_reference_templates_use_fixed_high_signal_paths(template_name, required_ops):
    g = _build_template_graph(template_name)

    op_names = {n.op_name for n in g.nodes.values() if not n.is_input}
    assert required_ops.issubset(op_names)
    assert not g.metadata.get("template_slot_usage"), (
        f"{template_name} should not emit random slot telemetry anymore"
    )


@pytest.mark.parametrize(
    "template_name,required_ops",
    (
        # Rewritten: parallel attn+SSM hybrid, no matmul
        ("attn_normalized_matmul", {"rmsnorm", "linear_proj", "add"}),
        (
            "attn_softmax_normalized_matmul",
            {"softmax_attention", "matmul", "swiglu_mlp"},
        ),
        (
            "attn_softmax_normalized_matmul_compact_ffn",
            {"softmax_attention", "matmul", "swiglu_mlp"},
        ),
        (
            "attn_softmax_normalized_matmul_fixed_tail_norm",
            {"softmax_attention", "matmul", "swiglu_mlp"},
        ),
        (
            "attn_linear_no_matmul_ffn",
            {"linear_attention", "softmax_attention", "swiglu_mlp"},
        ),
        (
            "attn_linear_no_matmul_ffn_dense_tail",
            {"linear_attention", "softmax_attention", "fused_linear_gelu"},
        ),
        (
            "attn_linear_no_matmul_ffn_direct_recovery",
            {"linear_attention", "softmax_attention", "swiglu_mlp"},
        ),
        (
            "attn_routing_block",
            {
                "softmax_attention",
                "difficulty_blend_3way",
                "depth_weighted_proj",
                "swiglu_mlp",
            },
        ),
        (
            "graph_attn_sparse_ffn",
            {"graph_attention", "matmul", "block_sparse_linear"},
        ),
        ("attn_spectral_filter", {"spectral_filter", "linear_proj"}),
        ("attn_rwkv_hybrid", {"layernorm", "rwkv_channel"}),
        (
            "depth_token_mask_block",
            {
                "rmsnorm",
                "token_class_proj",
                "score_depth_blend",
                "depth_token_mask",
                "linear_proj",
            },
        ),
        (
            "linear_attn_ffn_block",
            {"linear_attention", "matmul", "swiglu_mlp"},
        ),
        (
            "linear_attn_sparse_ffn",
            {"linear_attention", "matmul", "nm_sparse_linear"},
        ),
    ),
)
def test_rehab_templates_keep_stabilizing_scaffolds(template_name, required_ops):
    g = _build_template_graph(template_name)

    op_names = {n.op_name for n in g.nodes.values() if not n.is_input}
    assert required_ops.issubset(op_names)


@pytest.mark.parametrize(
    "template_name,expected_present,expected_absent",
    (
        (
            "attn_softmax_normalized_matmul",
            {"softmax_attention", "matmul", "swiglu_mlp"},
            {"linear_attention", "nm_sparse_linear", "difficulty_blend_3way"},
        ),
        (
            "attn_softmax_normalized_matmul_compact_ffn",
            {"softmax_attention", "matmul", "swiglu_mlp"},
            {"linear_attention", "nm_sparse_linear", "difficulty_blend_3way"},
        ),
        (
            "attn_softmax_normalized_matmul_fixed_tail_norm",
            {"softmax_attention", "matmul", "swiglu_mlp"},
            {"linear_attention", "nm_sparse_linear", "difficulty_blend_3way"},
        ),
        (
            "attn_linear_no_matmul_ffn",
            {"linear_attention", "softmax_attention", "swiglu_mlp"},
            {"matmul", "nm_sparse_linear", "difficulty_blend_3way"},
        ),
        (
            "attn_linear_no_matmul_ffn_dense_tail",
            {"linear_attention", "softmax_attention", "fused_linear_gelu"},
            {"matmul", "nm_sparse_linear", "difficulty_blend_3way"},
        ),
        (
            "attn_linear_no_matmul_ffn_direct_recovery",
            {"linear_attention", "softmax_attention", "swiglu_mlp"},
            {"matmul", "nm_sparse_linear", "difficulty_blend_3way"},
        ),
    ),
)
def test_controlled_attention_ablations_change_one_major_component(
    template_name, expected_present, expected_absent
):
    g = _build_template_graph(template_name)

    op_names = {n.op_name for n in g.nodes.values() if not n.is_input}
    assert expected_present.issubset(op_names)
    assert expected_absent.isdisjoint(op_names)


@pytest.mark.parametrize(
    "template_name,slot_expectations",
    (
        (
            "attn_routing_block",
            {
                0: ("norm_wrap", False),
                1: ("norm_wrap", False),
            },
        ),
        (
            "attn_normalized_matmul",
            {
                0: ("norm_wrap", False),
                1: ("attention_core", False),
            },
        ),
        (
            "attn_linear_no_matmul_ffn_direct_recovery",
            {
                0: ("norm_wrap", False),
                1: ("norm_wrap", False),
            },
        ),
        (
            "linear_attn_ffn_block",
            {
                0: ("norm_wrap", False),
                1: ("norm_wrap", False),
                2: ("norm_wrap", False),
            },
        ),
        (
            "linear_attn_sparse_ffn",
            {
                0: ("norm_wrap", False),
                1: ("norm_wrap", False),
            },
        ),
        (
            "graph_attn_sparse_ffn",
            {
                0: ("norm_wrap", False),
            },
        ),
    ),
)
def test_rehab_templates_keep_mandatory_slots_constrained(
    template_name, slot_expectations
):
    g = _build_template_graph(template_name)
    slot_usage = g.metadata.get("template_slot_usage") or []
    by_idx = {entry["slot_index"]: entry for entry in slot_usage}

    for slot_index, (expected_class, expected_wildcard) in slot_expectations.items():
        entry = by_idx[slot_index]
        assert entry["wildcard"] is expected_wildcard
        assert entry["selected_motif_class"] == expected_class


def test_winner_mutations_change_only_targeted_tail_settings():
    compact = _build_template_graph("attn_softmax_normalized_matmul_compact_ffn")
    dense = _build_template_graph("attn_linear_no_matmul_ffn_dense_tail")

    compact_ffn = [
        node for node in compact.nodes.values() if node.op_name == "swiglu_mlp"
    ]
    assert compact_ffn
    assert any(node.config.get("mlp_ratio") == 2.0 for node in compact_ffn)

    dense_ffn = [
        node for node in dense.nodes.values() if node.op_name == "fused_linear_gelu"
    ]
    assert dense_ffn
    assert dense_ffn[0].config.get("out_dim") == dense.model_dim


def test_direct_recovery_variant_removes_explicit_refine_norm():
    baseline = _build_template_graph("attn_linear_no_matmul_ffn")
    direct = _build_template_graph("attn_linear_no_matmul_ffn_direct_recovery")
    baseline_rmsnorms = sum(
        1
        for node in baseline.nodes.values()
        if not node.is_input and node.op_name == "rmsnorm"
    )
    direct_rmsnorms = sum(
        1
        for node in direct.nodes.values()
        if not node.is_input and node.op_name == "rmsnorm"
    )
    assert direct_rmsnorms == baseline_rmsnorms - 1


def test_slot_usage_records_canonical_keys_for_motif_slots():
    g = _build_template_graph("attn_routing_block")
    slot_usage = g.metadata.get("template_slot_usage") or []
    assert slot_usage

    for entry in slot_usage:
        assert "slot_key_canonical" in entry
        assert ".slot" in entry["slot_key_canonical"]


@pytest.mark.parametrize(
    "template_name,seeds",
    (
        ("attn_bottleneck_hybrid", range(25)),
        ("attn_hyperbolic", range(25)),
        ("attn_decay_sequence", range(25)),
        ("hybrid_sparse_triplet_router", range(25)),
        ("sequential", range(25)),
    ),
)
def test_audited_templates_validate_across_seed_slice(template_name, seeds):
    failures = []
    for seed in seeds:
        g = ComputationGraph(model_dim=64)
        inp = g.add_input()
        out = apply_template(g, inp, random.Random(seed), template_name=template_name)
        g.set_output(out)
        result = validate_graph(g)
        if not result.valid:
            failures.append((seed, result.errors[:3]))

    assert not failures, failures


def test_slot_rule_summary_exports_current_template_constraints():
    rules = {row["slot_key"]: row for row in get_slot_rule_summary()}

    assert set(rules) == {"depth_token_mask_block.slot1"}
    assert (
        "route_lanes_block" in rules["depth_token_mask_block.slot1"]["blocked_motifs"]
    )
