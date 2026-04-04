"""Regression tests for targeted template backfill generation."""

from research.synthesis.grammar import GrammarConfig, batch_generate
from research.synthesis.templates import DEFAULT_TEMPLATE_WEIGHTS


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
    "diff_attn_ffn_block",
    "diff_attn_conv_hybrid",
    "local_attn_ssm_hybrid",
    "graph_attn_ffn_block",
    "attn_spectral_filter",
}


def _template_weights(template_name: str) -> dict[str, float]:
    weights = {name: 0.01 for name in DEFAULT_TEMPLATE_WEIGHTS}
    weights[template_name] = 100.0
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
        "diff_attn_ffn_block",
        "diff_attn_conv_hybrid",
        "diff_attn_routing",
        "local_attn_routing",
        "local_attn_moe",
        "local_attn_ssm_hybrid",
        "graph_attn_ffn_block",
        "diff_attn_moe",
        "graph_attn_moe",
        "attn_sparse_moe",
        "attn_spectral_filter",
    ):
        _assert_targeted_generation(template_name, seed=42)
