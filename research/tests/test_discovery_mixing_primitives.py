import random

import pytest
import torch

from research.synthesis.compiled_op import CompiledOp
from research.synthesis.graph import ComputationGraph, ShapeInfo
from research.synthesis.primitives import get_primitive
from research.synthesis.templates import apply_template

DISCOVERY_MIXERS = (
    "sparsemax_attention",
    "entmax_attention",
    "dplr_gated_delta",
    "token_hodge_mixer",
    "wavelet_packet_mix",
    "retention_mix",
    "product_key_memory",
)

DISCOVERY_TEMPLATE_OPS = {
    "sparsemax_attention_block": "sparsemax_attention",
    "entmax_attention_block": "entmax_attention",
    "dplr_gated_delta_block": "dplr_gated_delta",
    "token_hodge_mixer_block": "token_hodge_mixer",
    "wavelet_packet_mix_block": "wavelet_packet_mix",
    "retention_mix_block": "retention_mix",
    "product_key_memory_block": "product_key_memory",
}


def _compiled(op_name: str, dim: int = 16) -> CompiledOp:
    config = {}
    if op_name == "product_key_memory":
        config = {"num_keys": 8, "top_k": 2}
    if op_name == "wavelet_packet_mix":
        config = {"levels": 2}
    if op_name == "entmax_attention":
        config = {"alpha": 1.5}
    return CompiledOp(op_name, config, ShapeInfo(dim=dim), ShapeInfo(dim=dim), dim)


@pytest.mark.parametrize("op_name", DISCOVERY_MIXERS)
def test_discovery_mixer_registry_and_forward_backward(op_name):
    primitive = get_primitive(op_name)
    assert primitive.has_params
    assert primitive.binding_range_class in {"full", "none"}

    torch.manual_seed(7)
    op = _compiled(op_name)
    x = torch.randn(2, 9, 16, requires_grad=True)
    y = op(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("op_name", DISCOVERY_MIXERS)
def test_discovery_mixers_do_not_leak_future_tokens(op_name):
    torch.manual_seed(11)
    op = _compiled(op_name)
    x = torch.randn(1, 8, 16)
    perturbed = x.clone()
    perturbed[:, 5:] = perturbed[:, 5:] + torch.randn_like(perturbed[:, 5:]) * 10

    y = op(x)
    y_perturbed = op(perturbed)

    assert torch.allclose(y[:, :5], y_perturbed[:, :5], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("template_name,target_op", DISCOVERY_TEMPLATE_OPS.items())
def test_discovery_templates_record_primary_core_slot(template_name, target_op):
    graph = ComputationGraph(model_dim=16)
    inp = graph.add_input()
    out = apply_template(
        graph,
        inp,
        random.Random(19),
        template_name=template_name,
    )
    graph.set_output(out)

    op_names = [node.op_name for node in graph.nodes.values() if not node.is_input]
    assert target_op in op_names

    slots = graph.metadata.get("template_slot_usage", [])
    assert any(
        slot.get("selected_motif") == target_op
        and slot.get("slot_classes") == ["primary_core"]
        for slot in slots
    )
