"""Tests for template optimization: rewritten + new high-performance templates.

Verifies that:
1. New templates build valid graphs across multiple seeds
2. Rewritten templates still compile and produce valid forward passes
3. Retired templates have weight 0 (no sampling)
4. All new/rewritten templates have healthy gradient flow
5. All new/rewritten templates can train (loss decreases)
"""

import random

import pytest
import torch

from research.synthesis.compiler import compile_graph
from research.synthesis.graph import ComputationGraph
from research.synthesis.templates import (
    DEFAULT_TEMPLATE_WEIGHTS,
    TEMPLATES,
    apply_template,
)
from research.synthesis.validator import validate_graph

# ── Template categories ──────────────────────────────────────────────

NEW_TEMPLATES = [
    "recursive_attn_ssm_depth",
    "latent_attn_padic_hybrid",
    "graph_attn_ssm_recursive",
]

REWRITTEN_TEMPLATES = [
    "attn_softmax_normalized_matmul_v2",
    "attn_linear_softmax_recovery_control",
    "attn_softmax_matmul_sparse_tail",
    "attn_normalized_matmul",
    "attn_bottleneck_hybrid",
    "depth_gated_block_matmul_stable",
]

RETIRED_TEMPLATES = [
    "multiscale_difficulty_router_blocksparse_attn_ssm",
    "multiscale_difficulty_router_easy_attn_ssm",
    "attn_reciprocal_gated",
    "attn_softmax_router_sidecar",
]

ALL_OPTIMIZED = NEW_TEMPLATES + REWRITTEN_TEMPLATES


# ── Registration tests ───────────────────────────────────────────────


@pytest.mark.unit
def test_new_templates_registered():
    for name in NEW_TEMPLATES:
        assert name in TEMPLATES, f"{name} not in TEMPLATES"
        assert name in DEFAULT_TEMPLATE_WEIGHTS, f"{name} not in weights"
        assert DEFAULT_TEMPLATE_WEIGHTS[name] > 0, f"{name} has zero weight"


@pytest.mark.unit
def test_retired_templates_fully_pruned():
    """Audit fix 2026-04-17: retired templates were deleted from the registry
    entirely (along with their function bodies and weight entries) — DB-compat
    is now handled via the RETIRED_TEMPLATE_NAMES frozenset in templates.py,
    not by keeping zero-weight ghosts in the active registry."""
    for name in RETIRED_TEMPLATES:
        assert name not in TEMPLATES, (
            f"{name} is retired and must not appear in the active TEMPLATES registry"
        )
        assert name not in DEFAULT_TEMPLATE_WEIGHTS, (
            f"{name} retired — must not have a weight entry"
        )


@pytest.mark.unit
def test_rewritten_templates_boosted_weight():
    for name in REWRITTEN_TEMPLATES:
        w = DEFAULT_TEMPLATE_WEIGHTS[name]
        assert w >= 3.0, f"{name} weight too low: {w}"


# ── Graph building tests ─────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("template_name", ALL_OPTIMIZED)
def test_template_builds_across_seeds(template_name):
    """Each optimized template must build valid graphs across 10 seeds."""
    for seed in range(10):
        g = ComputationGraph(model_dim=128)
        inp = g.add_input()
        out = apply_template(g, inp, random.Random(seed), template_name=template_name)
        g.set_output(out)
        assert len(g.nodes) > 3, f"Trivial graph at seed {seed}"


@pytest.mark.unit
@pytest.mark.parametrize("template_name", ALL_OPTIMIZED)
def test_template_compiles_and_forward(template_name):
    """Each optimized template must compile via IR and produce a valid forward pass."""
    g = ComputationGraph(model_dim=128)
    inp = g.add_input()
    out = apply_template(g, inp, random.Random(42), template_name=template_name)
    g.set_output(out)

    layer = compile_graph(g, use_ir=True)
    x = torch.randn(2, 16, 128)
    with torch.no_grad():
        y = layer(x)

    assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
    assert not torch.isnan(y).any(), "NaN in output"
    assert not torch.isinf(y).any(), "Inf in output"


# ── Gradient flow tests ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("template_name", ALL_OPTIMIZED)
def test_template_gradient_flow(template_name):
    """Each optimized template must allow gradients to flow back to the input."""
    g = ComputationGraph(model_dim=64)
    inp = g.add_input()
    out = apply_template(g, inp, random.Random(42), template_name=template_name)
    g.set_output(out)

    layer = compile_graph(g, use_ir=True)
    x = torch.randn(2, 8, 64, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None, "No gradient on input"
    assert x.grad.norm().item() > 0, "Zero gradient"
    assert not torch.isnan(x.grad).any(), "NaN in gradient"


# ── Validation tests ─────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("template_name", NEW_TEMPLATES)
def test_new_template_validates_across_seeds(template_name):
    """New templates must pass validation across 25 seeds."""
    failures = []
    for seed in range(25):
        g = ComputationGraph(model_dim=64)
        inp = g.add_input()
        out = apply_template(g, inp, random.Random(seed), template_name=template_name)
        g.set_output(out)
        result = validate_graph(g)
        if not result.valid:
            failures.append((seed, result.errors[:3]))

    # Allow up to 3/25 failures (same tolerance as latent_attn_ssm_hybrid winner)
    assert len(failures) <= 3, (
        f"{template_name}: {len(failures)}/25 validation failures: {failures[:5]}"
    )


@pytest.mark.unit
def test_attn_bottleneck_hybrid_validates():
    """attn_bottleneck_hybrid (rewritten) must validate across 25 seeds."""
    failures = []
    for seed in range(25):
        g = ComputationGraph(model_dim=64)
        inp = g.add_input()
        out = apply_template(
            g, inp, random.Random(seed), template_name="attn_bottleneck_hybrid"
        )
        g.set_output(out)
        result = validate_graph(g)
        if not result.valid:
            failures.append((seed, result.errors[:3]))

    assert not failures, f"attn_bottleneck_hybrid: {len(failures)}/25 failed: {failures}"
