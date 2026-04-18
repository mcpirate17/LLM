"""Regression tests for targeted template backfill generation."""

import pytest

from research.synthesis.grammar import GrammarConfig, batch_generate
from research.synthesis.templates import TEMPLATES
from research.synthesis.validator import validate_graph
from research.tools.backfill_templates import _phase_settings
from research.scientist.notebook.notebook_misc import _MiscMixin


_UNIFORM_CATEGORY_WEIGHTS = {
    "elementwise_unary": 1.0,
    "elementwise_binary": 1.0,
    "reduction": 1.0,
    "linear_algebra": 1.0,
    "structural": 1.0,
    "parameterized": 1.0,
    "mixing": 1.0,
    "sequence": 1.0,
    "frequency": 1.0,
    "math_space": 1.0,
    "functional": 1.0,
}

_NON_ROUTING_TEMPLATES = {
    "attn_normalized_matmul",
    "attn_softmax_normalized_matmul",
    "attn_softmax_normalized_matmul_v2",
    "attn_softmax_normalized_matmul_compact_ffn",
    "attn_softmax_normalized_matmul_fixed_tail_norm",
    "attn_linear_no_matmul_ffn",
    "attn_linear_no_matmul_ffn_v2",
    "attn_linear_no_matmul_ffn_dense_tail",
    "attn_linear_no_matmul_ffn_direct_recovery",
    "diff_attn_ffn_block",
    "diff_attn_conv_hybrid",
    "local_attn_ssm_hybrid",
    "graph_attn_ffn_block",
    "graph_attn_sparse_ffn",
    "attn_spectral_filter",
    "linear_attn_ffn_block",
    "linear_attn_sparse_ffn",
    "difficulty_routed_attention_block",
    "strided_attention_block",
    "gated_progressive_attention_block",
    "gated_linear_attention_block",
    "long_conv_hyena_block",
    "associative_memory_block",
    "mixture_of_recursions_block",
    "codex_ssm_retention_block",
    "codex_ssm_delta_memory_block",
    "codex_ssm_mla_gated_block",
    "codex_ssm_local_recall_block",
    "typed_slot_memory_block",
    "sparse_relation_graph_block",
    "token_program_interpreter_block",
}


def _template_weights(template_name: str) -> dict[str, float]:
    weights = {name: 0.0 for name in TEMPLATES}
    weights[template_name] = 1.0
    return weights


def _backfill_like_config(template_name: str) -> GrammarConfig:
    return GrammarConfig(
        template_weights=_template_weights(template_name),
        category_weights=dict(_UNIFORM_CATEGORY_WEIGHTS),
        composition_depth=1,
        routing_mandatory=template_name not in _NON_ROUTING_TEMPLATES,
    )


def _assert_targeted_generation(template_name: str, seed: int) -> None:
    result = batch_generate(1, _backfill_like_config(template_name), base_seed=seed)
    assert len(result.graphs) == 1, (
        f"{template_name} failed targeted backfill generation "
        f"(attempted={result.n_attempted}, rejected={result.n_rejected_grammar})"
    )
    templates_used = result.graphs[0].metadata.get("templates_used", [])
    assert templates_used, f"{template_name} graph missing templates_used metadata"
    assert templates_used == [template_name]


def test_targeted_backfill_generates_requested_templates():
    for template_name in (
        "hybrid_sparse_triplet_router",
        "multiscale_difficulty_router",
        "multiscale_rich_lane_router",
        "intelligent_multilane_router",
        "attn_routing_block",
        "attn_normalized_matmul",
        "attn_softmax_normalized_matmul",
        "attn_softmax_normalized_matmul_v2",
        "attn_softmax_normalized_matmul_compact_ffn",
        "attn_softmax_normalized_matmul_fixed_tail_norm",
        "attn_linear_no_matmul_ffn",
        "attn_linear_no_matmul_ffn_v2",
        "attn_linear_no_matmul_ffn_dense_tail",
        "attn_linear_no_matmul_ffn_direct_recovery",
        "diff_attn_ffn_block",
        "diff_attn_conv_hybrid",
        "diff_attn_routing",
        "local_attn_routing",
        "local_attn_moe",
        "local_attn_ssm_hybrid",
        "graph_attn_ffn_block",
        "graph_attn_sparse_ffn",
        "diff_attn_moe",
        "graph_attn_moe",
        "attn_sparse_moe",
        "attn_spectral_filter",
        "linear_attn_ffn_block",
        "linear_attn_sparse_ffn",
        "difficulty_routed_attention_block",
        "strided_attention_block",
        "gated_progressive_attention_block",
        "gated_linear_attention_block",
        "long_conv_hyena_block",
        "associative_memory_block",
        "mixture_of_recursions_block",
        "codex_ssm_retention_block",
        "codex_ssm_delta_memory_block",
        "codex_ssm_mla_gated_block",
        "codex_ssm_local_recall_block",
    ):
        _assert_targeted_generation(template_name, seed=42)


@pytest.mark.parametrize(
    "template_name,seed",
    (
        ("hybrid_sparse_triplet_router", 123),
        ("multiscale_difficulty_router", 123),
        ("multiscale_rich_lane_router", 123),
        ("intelligent_multilane_router", 123),
        ("attn_routing_block", 2027348667),
        ("attn_normalized_matmul", 2135553863),
        ("attn_softmax_normalized_matmul", 2135553863),
        ("attn_softmax_normalized_matmul_v2", 2135553863),
        ("attn_softmax_normalized_matmul_compact_ffn", 2135553863),
        ("attn_softmax_normalized_matmul_fixed_tail_norm", 2135553863),
        ("attn_linear_no_matmul_ffn", 99796358),
        ("attn_linear_no_matmul_ffn_v2", 99796358),
        ("attn_linear_no_matmul_ffn_dense_tail", 99796358),
        ("attn_linear_no_matmul_ffn_direct_recovery", 99796358),
        ("linear_attn_ffn_block", 99796358),
        ("linear_attn_sparse_ffn", 1973232567),
        ("graph_attn_sparse_ffn", 2118196712),
        ("difficulty_routed_attention_block", 42),
        ("strided_attention_block", 42),
        ("gated_progressive_attention_block", 42),
        ("gated_linear_attention_block", 42),
        ("long_conv_hyena_block", 42),
        ("associative_memory_block", 42),
        ("mixture_of_recursions_block", 42),
        ("codex_ssm_retention_block", 42),
        ("codex_ssm_delta_memory_block", 42),
        ("codex_ssm_mla_gated_block", 42),
        ("codex_ssm_local_recall_block", 42),
    ),
)
def test_targeted_backfill_graphs_fit_screening_validator(template_name, seed):
    result = batch_generate(2, _backfill_like_config(template_name), base_seed=seed)
    assert len(result.graphs) == 2, (
        f"{template_name} expected 2 graphs from live backfill seed {seed} "
        f"(attempted={result.n_attempted}, rejected={result.n_rejected_grammar})"
    )
    for graph in result.graphs:
        validation = validate_graph(graph, max_ops=24, max_depth=18)
        assert validation.valid, (
            f"{template_name} produced invalid screening graph at seed {seed}: "
            f"{validation.errors}"
        )


@pytest.mark.parametrize(
    "template_name,expected_min_slots",
    (
        ("codex_ssm_retention_block", 4),
        ("codex_ssm_delta_memory_block", 2),
        ("codex_ssm_mla_gated_block", 4),
        ("codex_ssm_local_recall_block", 3),
        ("typed_slot_memory_block", 7),
        ("sparse_relation_graph_block", 6),
        ("token_program_interpreter_block", 7),
    ),
)
def test_codex_fast_attention_templates_emit_slot_usage(template_name, expected_min_slots):
    result = batch_generate(1, _backfill_like_config(template_name), base_seed=42)
    assert len(result.graphs) == 1
    slot_usage = result.graphs[0].metadata.get("template_slot_usage", [])
    assert len(slot_usage) >= expected_min_slots
    assert _MiscMixin._infer_template_slot_counts()[template_name] >= expected_min_slots


def test_stack_phase_uses_depth_safe_override_for_hybrid_sparse_triplet_router():
    assert (
        _phase_settings("stack", "hybrid_sparse_triplet_router")["composition_depth"]
        == 1
    )
    assert (
        _phase_settings("stack", "multiscale_difficulty_router")["composition_depth"]
        == 1
    )
    assert (
        _phase_settings("stack", "multiscale_rich_lane_router")["composition_depth"]
        == 1
    )
    assert (
        _phase_settings("stack", "intelligent_multilane_router")["composition_depth"]
        == 1
    )


def test_hybrid_sparse_triplet_router_tracks_named_slots_and_disables_wildcard_norm_slot():
    weights = _template_weights("hybrid_sparse_triplet_router")
    config = GrammarConfig(
        template_weights=weights,
        category_weights=dict(_UNIFORM_CATEGORY_WEIGHTS),
        composition_depth=1,
        routing_mandatory=True,
        wildcard_slot_prob=1.0,
    )
    result = batch_generate(1, config, base_seed=42)
    assert len(result.graphs) == 1
    slot_usage = result.graphs[0].metadata.get("template_slot_usage", [])
    assert slot_usage
    slot0 = slot_usage[0]
    assert slot0["selected_motif_class"] == "norm_wrap"
    assert slot0["wildcard"] is False
    slot_keys = {entry["slot_key"] for entry in slot_usage}
    assert "hybrid_sparse_triplet_router[0].default_path" in slot_keys
    assert "hybrid_sparse_triplet_router[0].sparse_spans" in slot_keys
    assert "hybrid_sparse_triplet_router[0].routed_lane" in slot_keys
    assert _MiscMixin._infer_template_slot_counts()["hybrid_sparse_triplet_router"] == 4


def test_multiscale_difficulty_router_tracks_multiscale_and_hard_path_slots():
    weights = _template_weights("multiscale_difficulty_router")
    config = GrammarConfig(
        template_weights=weights,
        category_weights=dict(_UNIFORM_CATEGORY_WEIGHTS),
        composition_depth=1,
        routing_mandatory=True,
        wildcard_slot_prob=1.0,
    )
    result = batch_generate(1, config, base_seed=42)
    assert len(result.graphs) == 1
    slot_usage = result.graphs[0].metadata.get("template_slot_usage", [])
    assert slot_usage
    slot0 = slot_usage[0]
    assert slot0["selected_motif_class"] == "norm_wrap"
    assert slot0["wildcard"] is False
    slot_keys = {entry["slot_key"] for entry in slot_usage}
    assert "multiscale_difficulty_router[0].default_path" in slot_keys
    assert "multiscale_difficulty_router[0].pair_spans" in slot_keys
    assert "multiscale_difficulty_router[0].triplet_spans" in slot_keys
    assert "multiscale_difficulty_router[0].quartet_spans" in slot_keys
    assert "multiscale_difficulty_router[0].pair_router" in slot_keys
    assert "multiscale_difficulty_router[0].triplet_router" in slot_keys
    assert "multiscale_difficulty_router[0].quartet_router" in slot_keys
    assert "multiscale_difficulty_router[0].hard_router" in slot_keys
    assert _MiscMixin._infer_template_slot_counts()["multiscale_difficulty_router"] == 9


def test_multiscale_rich_lane_router_tracks_medium_and_hard_lane_choices():
    weights = _template_weights("multiscale_rich_lane_router")
    config = GrammarConfig(
        template_weights=weights,
        category_weights=dict(_UNIFORM_CATEGORY_WEIGHTS),
        composition_depth=1,
        routing_mandatory=True,
        wildcard_slot_prob=1.0,
    )
    result = batch_generate(1, config, base_seed=42)
    assert len(result.graphs) == 1
    slot_usage = result.graphs[0].metadata.get("template_slot_usage", [])
    assert slot_usage
    slot_keys = {entry["slot_key"] for entry in slot_usage}
    assert "multiscale_rich_lane_router[0].default_path" in slot_keys
    assert "multiscale_rich_lane_router[0].pair_spans" in slot_keys
    assert "multiscale_rich_lane_router[0].triplet_spans" in slot_keys
    assert "multiscale_rich_lane_router[0].quartet_spans" in slot_keys
    assert "multiscale_rich_lane_router[0].medium_router" in slot_keys
    assert "multiscale_rich_lane_router[0].hard_router" in slot_keys
    selected = {entry["slot_key"]: entry["selected_motif"] for entry in slot_usage}
    assert selected["multiscale_rich_lane_router[0].medium_router"] in {
        "route_lanes",
        "adaptive_lane_mixer",
        "semi_structured_2_4_linear",
        "block_sparse_linear",
        "rwkv_time_mixing",
        "nm_sparse_linear",
        "default_path",
        "cheap_verify_blend",
        "conv1d_seq",
        "conv_only",
    }
    assert selected["multiscale_rich_lane_router[0].hard_router"] in {
        "compression_mixture_experts",
        "routing_conditioned_compression",
        "dual_compression_blend",
        "route_recursion",
        "adaptive_recursion",
        "mixed_recursion_gate",
        "moe_topk",
        "moe_2expert",
        "n_way_sparse_router",
        "state_space",
    }
    assert _MiscMixin._infer_template_slot_counts()["multiscale_rich_lane_router"] == 7


def test_intelligent_multilane_router_tracks_staged_lane_slots():
    weights = _template_weights("intelligent_multilane_router")
    config = GrammarConfig(
        template_weights=weights,
        category_weights=dict(_UNIFORM_CATEGORY_WEIGHTS),
        composition_depth=1,
        routing_mandatory=True,
        wildcard_slot_prob=1.0,
    )
    result = batch_generate(1, config, base_seed=42)
    assert len(result.graphs) == 1
    slot_usage = result.graphs[0].metadata.get("template_slot_usage", [])
    assert slot_usage
    slot_keys = {entry["slot_key"] for entry in slot_usage}
    assert "intelligent_multilane_router[0].pre_router" in slot_keys
    assert "intelligent_multilane_router[0].easy_router" in slot_keys
    assert "intelligent_multilane_router[0].pair_spans" in slot_keys
    assert "intelligent_multilane_router[0].triplet_spans" in slot_keys
    assert "intelligent_multilane_router[0].quartet_spans" in slot_keys
    assert "intelligent_multilane_router[0].medium_router" in slot_keys
    assert "intelligent_multilane_router[0].difficulty_signal" in slot_keys
    assert "intelligent_multilane_router[0].hard_router" in slot_keys
    assert "intelligent_multilane_router[0].token_merge" in slot_keys
    assert "intelligent_multilane_router[0].post_merge" in slot_keys
    selected = {entry["slot_key"]: entry["selected_motif"] for entry in slot_usage}
    assert selected["intelligent_multilane_router[0].easy_router"] in {
        "cheap_verify_blend",
        "conv_only",
        "conv1d_seq",
        "linear_proj",
        "nm_sparse_linear",
        "default_path",
    }
    assert selected["intelligent_multilane_router[0].medium_router"] in {
        "route_lanes",
        "adaptive_lane_mixer",
        "semi_structured_2_4_linear",
        "block_sparse_linear",
        "linear_proj",
        "nm_sparse_linear",
    }
    assert selected["intelligent_multilane_router[0].hard_router"] in {
        "adaptive_recursion",
        "route_recursion",
        "moe_topk",
        "moe_2expert",
        "state_space",
        "linear_proj",
    }
    assert selected["intelligent_multilane_router[0].token_merge"] == "linear_proj"
    assert (
        _MiscMixin._infer_template_slot_counts()["intelligent_multilane_router"] == 11
    )


def test_targeted_backfill_generation_reduces_intelligent_multilane_grammar_failures():
    result = batch_generate(
        100,
        _backfill_like_config("intelligent_multilane_router"),
        base_seed=559240519,
    )
    assert len(result.graphs) == 100
    assert result.n_rejected_grammar == 0
    assert result.n_attempted <= 140


def test_targeted_backfill_generation_widens_hybrid_triplet_search_space():
    result = batch_generate(
        100,
        _backfill_like_config("hybrid_sparse_triplet_router"),
        base_seed=559240519,
    )
    assert len(result.graphs) == 100
    assert result.n_rejected_dedup <= 80
    assert result.n_attempted <= 180
