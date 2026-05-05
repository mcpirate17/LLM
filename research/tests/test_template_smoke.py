"""Smoke tests for mined-pattern templates (Bucket D, 2026-05-04).

Covers Phase 3.1 chains (`tropical_attn_conv1d_seq_block`,
`rwkv_channel_conv1d_seq_block`, `matmul_conv1d_seq_block`) plus the Phase 5
V2 sparse-2:4 mine (`sparse_24_linear_block`).

Each new template must:
  1. Be present in TEMPLATES + DEFAULT_TEMPLATE_WEIGHTS.
  2. Build a graph that passes validate_graph() across multiple RNG seeds.
  3. Produce graphs with > 3 nodes (non-trivial).
"""

from __future__ import annotations

import random

import pytest

from research.synthesis.graph import ComputationGraph
from research.synthesis.templates import (
    DEFAULT_TEMPLATE_WEIGHTS,
    TEMPLATES,
    apply_template,
)
from research.synthesis.validator import validate_graph


pytestmark = [pytest.mark.unit]


PHASE_3_1_NEW_TEMPLATES = (
    "tropical_attn_conv1d_seq_block",
    "rwkv_channel_conv1d_seq_block",
    "matmul_conv1d_seq_block",
    # Phase 5 V2 (2026-05-04) — mined sparse-2:4 linear pattern
    "sparse_24_linear_block",
)


@pytest.mark.parametrize("template_name", PHASE_3_1_NEW_TEMPLATES)
def test_new_template_registered(template_name: str) -> None:
    assert template_name in TEMPLATES, f"{template_name} missing from TEMPLATES"
    weight = DEFAULT_TEMPLATE_WEIGHTS.get(template_name)
    assert weight is not None, f"{template_name} missing from DEFAULT_TEMPLATE_WEIGHTS"
    assert 0.5 <= weight <= 8.0, f"{template_name} weight {weight} outside clamp"


@pytest.mark.parametrize("template_name", PHASE_3_1_NEW_TEMPLATES)
def test_new_template_builds_across_seeds(template_name: str) -> None:
    """Each new template must build valid graphs across 10 seeds."""
    for seed in range(10):
        g = ComputationGraph(model_dim=128)
        inp = g.add_input()
        out = apply_template(g, inp, random.Random(seed), template_name=template_name)
        g.set_output(out)
        assert len(g.nodes) > 3, (
            f"{template_name} seed={seed}: trivial graph (n_nodes={len(g.nodes)})"
        )


@pytest.mark.parametrize("template_name", PHASE_3_1_NEW_TEMPLATES)
def test_new_template_validates(template_name: str) -> None:
    """Built graphs must pass validate_graph() (no errors)."""
    for seed in (0, 7, 13):
        g = ComputationGraph(model_dim=128)
        inp = g.add_input()
        out = apply_template(g, inp, random.Random(seed), template_name=template_name)
        g.set_output(out)
        result = validate_graph(g)
        assert result.valid, f"{template_name} seed={seed}: errors={result.errors}"
