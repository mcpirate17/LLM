"""Tests for NM-1 synthesis math-family lattice operators."""

from __future__ import annotations

import pytest
import torch

from research.synthesis.compiled_op import CompiledOp
from research.synthesis.compiler_registry import OP_DISPATCH
from research.synthesis.graph import ShapeInfo
from research.synthesis.primitives import estimate_op_params, get_primitive


NM1_FAMILY_OPS = {
    "calculus": (
        "causal_gradient_mix",
        "causal_laplacian_mix",
        "lie_derivative_flow_mix",
    ),
    "spectral_graph": (
        "chebyshev_spectral_mix",
        "dct_spectral_mix",
        "graph_eigbasis_mix",
        "legendre_basis_mix",
    ),
    "tensor_decomp": (
        "cp_tensor_mix",
        "tensor_train_mix",
        "tensor_ring_mix",
        "block_term_tensor_mix",
    ),
    "information_geometry": (
        "alpha_divergence_mix",
        "renyi_attention_mix",
        "natural_gradient_mixer",
    ),
    "multiscale": (
        "wavelet_packet_mix",
        "dyadic_diff_mix",
        "laplacian_pyramid_mix",
    ),
}

NEW_NM1_OPS = tuple(
    op
    for family_ops in NM1_FAMILY_OPS.values()
    for op in family_ops
    if op not in {"chebyshev_spectral_mix", "wavelet_packet_mix"}
)

EXPECTED_PARAM_COUNTS_D8 = {
    "causal_gradient_mix": 8 * 8 + 8,
    "causal_laplacian_mix": 8 * 8 + 8,
    "lie_derivative_flow_mix": 8 * 8 * 2 + 8,
    "dct_spectral_mix": 8 * 8 + 4 * 8,
    "graph_eigbasis_mix": 8 * 8 + 4 * 8 + 1,
    "legendre_basis_mix": 8 * 8 + 4 * 8,
    "cp_tensor_mix": 8 * 8 + 12 * 8,
    "tensor_train_mix": 9 * 8 + 16,
    "tensor_ring_mix": 8 * 8 + 2 * 8,
    "block_term_tensor_mix": 8 * 8 + 4 * 8,
    "alpha_divergence_mix": 8 * 8 + 8 + 1,
    "renyi_attention_mix": 8 * 8 * 4 + 1,
    "natural_gradient_mixer": 8 * 8 * 2 + 1,
    "dyadic_diff_mix": 8 * 8 + 3 * 8,
    "laplacian_pyramid_mix": 8 * 8 + 3 * 8,
}


def test_nm1_family_lattices_have_multiple_registered_ops() -> None:
    for family, op_names in NM1_FAMILY_OPS.items():
        assert len(op_names) >= 2, family
        for op_name in op_names:
            op = get_primitive(op_name)
            assert op.name == op_name
            assert op.has_params
            assert op_name in OP_DISPATCH


def test_nm1_param_formulas_match_rank_constants() -> None:
    for op_name, expected in EXPECTED_PARAM_COUNTS_D8.items():
        assert estimate_op_params(get_primitive(op_name), 8) == expected


@pytest.mark.parametrize("op_name", NEW_NM1_OPS)
def test_nm1_new_ops_compile_forward_backward_finite(op_name: str) -> None:
    torch.manual_seed(100 + len(op_name))
    dim = 8
    op = CompiledOp(op_name, {}, ShapeInfo(dim=dim), ShapeInfo(dim=dim), dim)
    x = torch.randn(2, 10, dim, requires_grad=True)

    out = op(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, torch.zeros_like(out))

    loss = out.square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    params = list(op.named_parameters())
    assert params, op_name
    for name, param in params:
        assert param.grad is not None, f"{op_name}:{name}"
        assert torch.isfinite(param.grad).all(), f"{op_name}:{name}"
