from __future__ import annotations

import random

import pytest
import torch

from research.synthesis.compiler import compile_model
from research.synthesis.graph import ComputationGraph
from research.synthesis.op_roles import OpRole, get_role
from research.synthesis.primitives import PRIMITIVE_REGISTRY, estimate_op_params
from research.synthesis.templates import TEMPLATES, apply_template
from research.synthesis.validator import validate_graph
from research.tools.backfill_templates import (
    NOVEL_MIXER_BACKFILL_TEMPLATES,
    _NON_ROUTING_TEMPLATES,
)


pytestmark = [pytest.mark.unit]


NEW_MIXERS = ("reciprocal_rank_attention", "phase_lock_attention")
NEW_TEMPLATES = (
    "reciprocal_rank_attention_block",
    "phase_lock_attention_block",
    "stdp_reciprocal_memory_block",
)
ACTIVE_NOVEL_TEMPLATES = (
    "clifford_geometric_mixer_block",
    "tropical_maxplus_mixer_block",
    "ultrametric_hierarchical_ensemble_block",
)
EXPECTED_TEMPLATE_COMPONENTS = {
    "clifford_geometric_mixer_block": {
        "clifford_attention",
        "geometric_product",
        "rotor_transform",
        "versor_apply",
    },
    "tropical_maxplus_mixer_block": {
        "tropical_attention",
        "tropical_gate",
        "tropical_center",
        "associative_memory",
    },
    "ultrametric_hierarchical_ensemble_block": {
        "ultrametric_attention",
        "padic_expand",
        "state_space",
    },
    "reciprocal_rank_attention_block": {
        "reciprocal_rank_attention",
        "state_space",
    },
    "phase_lock_attention_block": {
        "phase_lock_attention",
        "state_space",
    },
    "stdp_reciprocal_memory_block": {
        "spike_rate_code",
        "stdp_attention",
        "reciprocal_rank_attention",
        "sparsemax_attention",
    },
}


@pytest.mark.parametrize("op_name", NEW_MIXERS)
def test_new_content_addressed_mixers_are_full_range(op_name: str) -> None:
    op = PRIMITIVE_REGISTRY[op_name]
    assert op.binding_range_class == "full"
    assert get_role(op_name) is OpRole.MIX
    assert estimate_op_params(op, 32) == 32 * 32 * 4 + 1


def test_backfill_template_prompt_set_contains_all_six_novel_templates() -> None:
    assert set(NOVEL_MIXER_BACKFILL_TEMPLATES) == set(EXPECTED_TEMPLATE_COMPONENTS)
    assert set(NOVEL_MIXER_BACKFILL_TEMPLATES) <= set(TEMPLATES)
    assert set(NOVEL_MIXER_BACKFILL_TEMPLATES) <= _NON_ROUTING_TEMPLATES


@pytest.mark.parametrize("template_name", NEW_TEMPLATES)
def test_new_retrieval_templates_build_valid_graphs(template_name: str) -> None:
    assert template_name in TEMPLATES
    for seed in (0, 3, 7):
        graph = ComputationGraph(model_dim=128)
        inp = graph.add_input()
        out = apply_template(
            graph, inp, random.Random(seed), template_name=template_name
        )
        graph.set_output(out)
        result = validate_graph(graph)
        ops = {node.op_name for node in graph.nodes.values()}
        assert result.valid, f"{template_name} seed={seed}: {result.errors}"
        assert any(op in ops for op in NEW_MIXERS)
        assert any(
            PRIMITIVE_REGISTRY[op].binding_range_class == "full"
            for op in ops
            if op in PRIMITIVE_REGISTRY
        )


@pytest.mark.parametrize("template_name", NOVEL_MIXER_BACKFILL_TEMPLATES)
def test_all_novel_backfill_templates_emit_expected_components(
    template_name: str,
) -> None:
    for seed in (0, 3, 7):
        graph = ComputationGraph(model_dim=128)
        inp = graph.add_input()
        out = apply_template(
            graph, inp, random.Random(seed), template_name=template_name
        )
        graph.set_output(out)
        result = validate_graph(graph)
        ops = {node.op_name for node in graph.nodes.values()}
        expected = EXPECTED_TEMPLATE_COMPONENTS[template_name]
        full_range_expected = {
            op
            for op in expected
            if op in PRIMITIVE_REGISTRY
            and PRIMITIVE_REGISTRY[op].binding_range_class == "full"
        }
        assert result.valid, f"{template_name} seed={seed}: {result.errors}"
        assert expected <= ops
        assert full_range_expected, f"{template_name} has no expected full-range mixer"
        assert full_range_expected <= ops


@pytest.mark.parametrize("template_name", ACTIVE_NOVEL_TEMPLATES)
def test_active_novel_manifest_templates_validate(template_name: str) -> None:
    """Novel-manifest templates stay structurally valid when wired into TEMPLATES."""
    assert template_name in TEMPLATES
    for seed in (0, 3, 7):
        graph = ComputationGraph(model_dim=128)
        inp = graph.add_input()
        out = apply_template(
            graph, inp, random.Random(seed), template_name=template_name
        )
        graph.set_output(out)
        result = validate_graph(graph)
        ops = {node.op_name for node in graph.nodes.values()}
        assert result.valid, f"{template_name} seed={seed}: {result.errors}"
        assert any(
            PRIMITIVE_REGISTRY[op].binding_range_class == "full"
            for op in ops
            if op in PRIMITIVE_REGISTRY
        )


@pytest.mark.parametrize("op_name", NEW_MIXERS)
def test_new_attention_mixers_compile_forward_backward(op_name: str) -> None:
    graph = ComputationGraph(model_dim=64)
    inp = graph.add_input()
    out = graph.add_op(op_name, [inp])
    graph.set_output(out)

    model = compile_model([graph], vocab_size=128, max_seq_len=16)
    tokens = torch.randint(0, 128, (2, 16))
    logits = model(tokens)
    loss = logits.float().mean()
    loss.backward()

    assert logits.shape == (2, 16, 128)
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
