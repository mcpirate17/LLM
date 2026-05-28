"""Tests for the PQ-bottleneck MoE block + template.

Verifies the op registers correctly (MIXING category, MIX role, ``identity``
shape rule), the screening template builds a valid graph, and the resulting
model compiles + back-propagates. Also seed-validates the template (≤3/25
invalid-graph rate) to match the workspace convention enforced by
``test_template_optimization``.
"""

from __future__ import annotations

import random

import pytest
import torch

pytestmark = pytest.mark.unit


def test_op_registered_with_mixing_category_and_mix_role():
    """``pq_embedding_moe_block`` is registered as MIXING / MIX / identity."""
    from research.synthesis.op_roles import OpRole, get_role
    from research.synthesis.primitives import PRIMITIVE_REGISTRY, OpCategory

    op = PRIMITIVE_REGISTRY.get("pq_embedding_moe_block")
    assert op is not None, "pq_embedding_moe_block must be in PRIMITIVE_REGISTRY"
    assert op.category is OpCategory.MIXING, (
        f"expected MIXING (sibling MoE convention), got {op.category}"
    )
    assert op.shape_rule == "identity"
    assert op.has_params
    assert get_role("pq_embedding_moe_block") is OpRole.MIX, (
        "missing OpRole.MIX entry — motif/slot wiring will mis-classify"
    )


def test_screening_template_instantiates_and_compiles():
    """The trimmed PQ→MoE→residual template builds + compiles + backprops."""
    from research.synthesis.compiler import compile_model
    from research.synthesis.graph import ComputationGraph
    from research.synthesis.templates import apply_template

    g = ComputationGraph(model_dim=128)
    inp = g.add_input()
    g.set_output(
        apply_template(g, inp, random.Random(0), template_name="pq_embedding_moe_block")
    )
    model = compile_model([g] * 2, vocab_size=256, max_seq_len=32)
    y = model(torch.randint(0, 256, (2, 16)))
    assert tuple(y.shape) == (2, 16, 256)
    y.float().pow(2).mean().backward()


def test_template_validates_across_seeds():
    """Same ≤3/25 tolerance test_template_optimization applies to new templates."""
    from research.synthesis.graph import ComputationGraph
    from research.synthesis.templates import apply_template
    from research.synthesis.validator import validate_graph

    failures = []
    for seed in range(25):
        g = ComputationGraph(model_dim=64)
        out = apply_template(
            g,
            g.add_input(),
            random.Random(seed),
            template_name="pq_embedding_moe_block",
        )
        g.set_output(out)
        result = validate_graph(g)
        if not result.valid:
            failures.append((seed, result.errors[:3]))
    assert len(failures) <= 3, (
        f"pq_embedding_moe_block: {len(failures)}/25 validation failures: "
        f"{failures[:5]}"
    )


def test_pq_block_forward_uses_codebook_params():
    """``arch_builder.PQEmbeddingMoEBlock`` exposes a trainable codebook."""
    from research.arch_builder import PQEmbeddingMoEBlock

    block = PQEmbeddingMoEBlock(dim=128, n_experts=4, topk=2, M=4, K=16)
    assert hasattr(block, "codebooks"), "PQ codebook param missing"
    assert block.codebooks.shape == (4, 16, 32), (
        f"expected (M=4, K=16, sub_dim=32), got {tuple(block.codebooks.shape)}"
    )
    x = torch.randn(2, 16, 128)
    y = block(x)
    assert y.shape == x.shape
    y.float().pow(2).mean().backward()
    assert block.codebooks.grad is not None
    assert torch.isfinite(block.codebooks.grad).all()
