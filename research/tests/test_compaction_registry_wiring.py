# pyright: reportPrivateImportUsage=false
"""NM-C compaction-mixer registry wiring — contract pins.

Until this wiring landed, all 11 shipped Tier-D mixers were invisible to the
discovery loop. What must hold for every op:

- registered in ``PRIMITIVE_REGISTRY`` with ``has_params=True`` (proposable),
- a compiler handler in ``OP_DISPATCH`` (compilable),
- ``estimate_op_params`` returns a positive budget estimate,
- a single-op graph compiles and runs forward+backward end-to-end at a
  friendly dim (64) AND an awkward one (50 — exercises the fail-soft knob
  adaptation: divisor shrink for block/chunk sizes, capacity clamp for
  subspaces/overlap),
- the compiled op's params live on the compiled module (gradient reaches at
  least one parameter).
"""

from __future__ import annotations

import pytest
import torch

from research.synthesis.compiler import compile_model
from research.synthesis.compiler_registry import OP_DISPATCH
from research.synthesis.graph import ComputationGraph
from research.synthesis.primitives import (
    PRIMITIVE_REGISTRY,
    estimate_op_params,
)

NM_C_OPS = [
    "monarch_mix",
    "butterfly_mix",
    "recurrent_depth_refine",
    "weight_dictionary_mix",
    "hypernet_layer_mix",
    "persistent_memory_refine",
    "block_sparse_mix",
    "token_merge_mix",
    "ternary_sign_mix",
    "padic_lowprec_mix",
    "subspace_mixture_mix",
    "lowrank_state_memory",
]


@pytest.mark.parametrize("op_name", NM_C_OPS)
def test_registered_and_dispatchable(op_name: str) -> None:
    op = PRIMITIVE_REGISTRY.get(op_name)
    assert op is not None, f"{op_name} missing from PRIMITIVE_REGISTRY"
    assert op.has_params, f"{op_name} must register has_params=True"
    assert op_name in OP_DISPATCH, f"{op_name} has no compiler handler"
    assert estimate_op_params(op, 64) > 0, f"{op_name} param estimate must be > 0"


def _compile_and_step(op_name: str, model_dim: int) -> None:
    graph = ComputationGraph(model_dim=model_dim)
    inp = graph.add_input()
    out = graph.add_op(op_name, [inp])
    graph.set_output(out)

    model = compile_model([graph], vocab_size=128, max_seq_len=16)
    tokens = torch.randint(0, 128, (2, 16))
    logits = model(tokens)
    loss = logits.float().mean()
    loss.backward()

    assert logits.shape == (2, 16, 128)
    assert torch.isfinite(logits).all(), f"{op_name} produced non-finite logits"
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("op_name", NM_C_OPS)
def test_compiles_forward_backward_dim64(op_name: str) -> None:
    _compile_and_step(op_name, model_dim=64)


@pytest.mark.parametrize("op_name", NM_C_OPS)
def test_compiles_at_awkward_dim50(op_name: str) -> None:
    """dim 50 forces the fail-soft knob adaptation paths (8 does not divide 50,
    default subspace/overlap sizes need clamping)."""
    _compile_and_step(op_name, model_dim=50)


@pytest.mark.parametrize(
    "template_name", ["compaction_mixer_block", "compaction_routing_block"]
)
def test_templates_registered_and_compile(template_name: str) -> None:
    """Proposability: both compaction templates are in the template registry
    with a nonzero default weight, build NM-C-bearing graphs, and those graphs
    compile and train end-to-end."""
    import random

    from research.synthesis.templates import DEFAULT_TEMPLATE_WEIGHTS, TEMPLATES

    assert template_name in TEMPLATES
    assert DEFAULT_TEMPLATE_WEIGHTS.get(template_name, 0.0) > 0.0

    for seed in range(4):
        graph = ComputationGraph(model_dim=64)
        inp = graph.add_input()
        out = TEMPLATES[template_name](graph, inp, random.Random(seed), None)
        graph.set_output(out)
        ops = {n.op_name for n in graph.nodes.values()}
        assert ops & set(NM_C_OPS), f"seed {seed}: no NM-C op in {sorted(ops)}"

        model = compile_model([graph], vocab_size=128, max_seq_len=16)
        logits = model(torch.randint(0, 128, (2, 16)))
        logits.float().mean().backward()
        assert torch.isfinite(logits).all()


def test_routing_variant_always_emits_routing_op() -> None:
    """The routing variant must reliably satisfy the routing-mandatory gate:
    every build contains an op from ROUTING_COMPRESSION_MOE_OPS (this is what
    qualifies it for the slot-0 manifest — see _routing_capable_manifest)."""
    import random

    from research.synthesis.grammar_support import ROUTING_COMPRESSION_MOE_OPS
    from research.synthesis.templates import TEMPLATES

    for seed in range(8):
        graph = ComputationGraph(model_dim=64)
        inp = graph.add_input()
        out = TEMPLATES["compaction_routing_block"](
            graph, inp, random.Random(seed), None
        )
        graph.set_output(out)
        ops = {n.op_name for n in graph.nodes.values()}
        assert ops & ROUTING_COMPRESSION_MOE_OPS, f"seed {seed}: {sorted(ops)}"
