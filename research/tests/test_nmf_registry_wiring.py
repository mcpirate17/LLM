# pyright: reportPrivateImportUsage=false
"""NM-F operator-family registry wiring — contract pins.

Mirror of ``test_compaction_registry_wiring.py`` for the NM-F ops (F2/F3/F4/
F5/F9; F6 joins once its files are committed by the owning lane). What must
hold for every op: registered+proposable (``PRIMITIVE_REGISTRY``), compilable
(``OP_DISPATCH``), positive param estimate, end-to-end compile fwd+bwd at a
friendly dim and an awkward one (exercises knob adaptation: CDMA chips
divisor, port-Hamiltonian band clamp, oblique rank clamp), and template-level
proposability incl. the routing-capable variant.
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

NMF_OPS = [
    "idempotent_oblique_memory",
    "nilpotent_lie_scan",
    "integral_control_mixer",
    "port_hamiltonian_mix",
    "cdma_slot_binding",
    "scale_equivariant_wavelet",
]


@pytest.mark.parametrize("op_name", NMF_OPS)
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


@pytest.mark.parametrize("op_name", NMF_OPS)
def test_compiles_forward_backward_dim64(op_name: str) -> None:
    _compile_and_step(op_name, model_dim=64)


@pytest.mark.parametrize("op_name", NMF_OPS)
def test_compiles_at_awkward_dim50(op_name: str) -> None:
    """dim 50 exercises knob adaptation: CDMA chips shrinks to a divisor of 50,
    port-Hamiltonian band and oblique rank clamp below the dim."""
    _compile_and_step(op_name, model_dim=50)


@pytest.mark.parametrize("template_name", ["nmf_mixer_block", "nmf_routing_block"])
def test_templates_registered_and_compile(template_name: str) -> None:
    """Proposability: both NM-F templates are in the registry with nonzero
    weight, build NM-F-bearing graphs, and those graphs compile end-to-end."""
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
        assert ops & set(NMF_OPS), f"seed {seed}: no NM-F op in {sorted(ops)}"

        model = compile_model([graph], vocab_size=128, max_seq_len=16)
        logits = model(torch.randint(0, 128, (2, 16)))
        logits.float().mean().backward()
        assert torch.isfinite(logits).all()


def test_routing_variant_always_emits_routing_op() -> None:
    """The routing variant must reliably satisfy the routing-mandatory gate:
    every build contains an op from ROUTING_COMPRESSION_MOE_OPS (what
    qualifies it for the slot-0 manifest)."""
    import random

    from research.synthesis.grammar_support import ROUTING_COMPRESSION_MOE_OPS
    from research.synthesis.templates import TEMPLATES

    for seed in range(8):
        graph = ComputationGraph(model_dim=64)
        inp = graph.add_input()
        out = TEMPLATES["nmf_routing_block"](graph, inp, random.Random(seed), None)
        graph.set_output(out)
        ops = {n.op_name for n in graph.nodes.values()}
        assert ops & ROUTING_COMPRESSION_MOE_OPS, f"seed {seed}: {sorted(ops)}"
