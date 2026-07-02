# pyright: reportPrivateImportUsage=false
"""Reachability + gate-safety pins for the two validated p-adic RDR gates.

`padic_gated_mixer` and `padic_depth_route` are the collapse-proof p-adic gates
validated 2026-06-29 (they underpin the RDR scale result). They were registered
and dispatchable, but reachable by NO template — so the grammar could never
propose two of the project's few validated novel mechanisms. This wires them into
the compaction channel pool (GLM handoff, O3). These pins keep them reachable and
gate-safe so a future pool refactor cannot silently drop them again:

- both are proposable (in the compaction channel pool + registered/dispatchable),
- `padic_depth_route` is routing-eligible (in ROUTING_COMPRESSION_MOE_OPS, hence
  gate 5 via the union) while the per-token `padic_gated_mixer` is not,
- both are pointwise → must NOT satisfy gate 6 (SEQUENCE_MIXING_OPS) on their own,
- every built graph that contains a p-adic op also contains a sequence mixer
  (the block's forced sequence secondary), so the block is gate-6 safe,
- a p-adic-primary graph compiles and trains end-to-end (finite, grad reaches
  params) — the collapse-proof gate does not blow up.
"""

from __future__ import annotations

import random

import pytest
import torch

from research.scientist.runner.execution_screening import _EFFICIENCY_OPS
from research.scientist.runner.execution_screening_graphs import SEQUENCE_MIXING_OPS
from research.synthesis._templates_compaction import (
    _CHANNEL_COMPACTION_OPS,
    _SEQUENCE_COMPACTION_OPS,
    COMPACTION_OPS,
)
from research.synthesis.compiler import compile_model
from research.synthesis.compiler_registry import OP_DISPATCH
from research.synthesis.graph import ComputationGraph
from research.synthesis.grammar_support import ROUTING_COMPRESSION_MOE_OPS
from research.synthesis.primitives import PRIMITIVE_REGISTRY
from research.synthesis.templates import TEMPLATES

PADIC_OPS = ("padic_gated_mixer", "padic_depth_route")
_SEQ = set(_SEQUENCE_COMPACTION_OPS)


@pytest.mark.parametrize("op", PADIC_OPS)
def test_padic_registered_and_dispatchable(op: str) -> None:
    prim = PRIMITIVE_REGISTRY.get(op)
    assert prim is not None and prim.has_params, f"{op} not a proposable primitive"
    assert op in OP_DISPATCH, f"{op} has no compiler handler"


@pytest.mark.parametrize("op", PADIC_OPS)
def test_padic_in_channel_pool(op: str) -> None:
    """Proposable via the compaction channel pool (and thus COMPACTION_OPS)."""
    assert op in _CHANNEL_COMPACTION_OPS
    assert op in COMPACTION_OPS


def test_padic_depth_route_is_routing_eligible() -> None:
    """The router satisfies gate 5 (via the ROUTING_COMPRESSION_MOE_OPS union);
    the per-token gate is deliberately not a router."""
    assert "padic_depth_route" in ROUTING_COMPRESSION_MOE_OPS
    assert "padic_depth_route" in _EFFICIENCY_OPS  # gate 5, derived by union
    assert "padic_gated_mixer" not in ROUTING_COMPRESSION_MOE_OPS


@pytest.mark.parametrize("op", PADIC_OPS)
def test_padic_pointwise_not_in_gate6(op: str) -> None:
    """Pointwise ops must not satisfy gate 6 alone — gate 6 demands genuine
    token mixing, which the forced sequence secondary provides instead."""
    assert op not in SEQUENCE_MIXING_OPS


def _build(template_name: str, weights: dict[str, float] | None, seed: int):
    graph = ComputationGraph(model_dim=64)
    if weights is not None:
        graph.metadata["_op_weights"] = weights
    inp = graph.add_input()
    out = TEMPLATES[template_name](graph, inp, random.Random(seed), None)
    graph.set_output(out)
    return graph


@pytest.mark.parametrize("op", PADIC_OPS)
def test_padic_reachable_and_gate6_safe(op: str) -> None:
    """Weighting toward a p-adic op makes it appear in built graphs, and every
    graph that contains it also contains a sequence mixer (gate-6 safety)."""
    weights = {o: 0.25 for o in COMPACTION_OPS}
    weights[op] = 4.5
    appeared = 0
    for seed in range(200):
        ops = {
            n.op_name
            for n in _build("compaction_mixer_block", weights, seed).nodes.values()
        }
        if op in ops:
            appeared += 1
            assert ops & _SEQ, f"seed {seed}: {op} present with no sequence mixer"
    assert appeared > 40, f"{op} unreachable under favorable weights ({appeared}/200)"


@pytest.mark.parametrize("op", PADIC_OPS)
def test_padic_primary_compiles_and_trains(op: str) -> None:
    """A p-adic-primary block (padic + forced sequence secondary, as the template
    builds it) compiles, forward is finite, and backward reaches parameters — the
    collapse-proof gate does not blow up."""
    graph = ComputationGraph(model_dim=64)
    inp = graph.add_input()
    prim = graph.add_op(op, [inp])
    sec = graph.add_op("token_merge_mix", [prim])  # the forced sequence secondary
    graph.set_output(sec)

    model = compile_model([graph], vocab_size=128, max_seq_len=16)
    logits = model(torch.randint(0, 128, (2, 16)))
    assert logits.shape == (2, 16, 128)
    assert torch.isfinite(logits).all(), f"{op} produced non-finite logits"
    logits.float().mean().backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
