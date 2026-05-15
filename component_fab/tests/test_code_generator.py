"""Tests for component_fab.generator.code_generator dispatch."""

from __future__ import annotations

import torch
from torch import nn

from component_fab.generator.code_generator import (
    generate_module,
    generate_module_from_spec,
)
from component_fab.generator.primitive_templates import (
    CalculusAugmentedLane,
    FiniteDifferenceCalculusLane,
    FourierBasisLane,
    GraphDiffusionAdapterLane,
    GraphDiffusionLane,
    LowRankAdapterLane,
    LowRankFactorizedLane,
    MultiscaleWaveletAdapterLane,
    MultiscaleWaveletLane,
    RandomFeatureKernelAdapterLane,
    RandomFeatureKernelLane,
    SparseBandedAdapterLane,
    SparseBandedMatrixLane,
    TopKLinear,
    TropicalAttention,
    TropicalStateSpace,
)
from component_fab.proposer.property_miner import AxisLift, CandidateTuple
from component_fab.proposer.spec_generator import spec_from_candidate


def test_dispatch_tropical_no_state_to_attention() -> None:
    m = generate_module(
        {"op_algebraic_space": "tropical", "op_dynamical_has_state": 0}, dim=16
    )
    assert isinstance(m, TropicalAttention)


def test_dispatch_tropical_with_state_to_state_space() -> None:
    m = generate_module(
        {"op_algebraic_space": "tropical", "op_dynamical_has_state": 1}, dim=16
    )
    assert isinstance(m, TropicalStateSpace)


def test_dispatch_topk_sparsity() -> None:
    m = generate_module({"op_activation_sparsity_pattern": "top_k"}, dim=16)
    assert isinstance(m, TopKLinear)


def test_dispatch_fourier_basis() -> None:
    m = generate_module({"op_spectral_preferred_basis": "fourier"}, dim=16)
    assert isinstance(m, FourierBasisLane)


def test_dispatch_calculus_math_knob() -> None:
    m = generate_module(
        {
            "op_math_family": "calculus",
            "op_calculus_operator": "causal_finite_difference_integral",
        },
        dim=16,
    )
    assert isinstance(m, FiniteDifferenceCalculusLane)


def test_dispatch_linear_algebra_math_knob() -> None:
    m = generate_module(
        {
            "op_math_family": "linear_algebra",
            "op_linear_algebra_structure": "low_rank_factorized",
        },
        dim=16,
    )
    assert isinstance(m, LowRankFactorizedLane)


def test_dispatch_sparse_matrix_math_knob() -> None:
    m = generate_module(
        {
            "op_math_family": "sparse_matrix",
            "op_sparse_matrix_pattern": "causal_banded",
        },
        dim=16,
    )
    assert isinstance(m, SparseBandedMatrixLane)


def test_dispatch_kernel_math_knob() -> None:
    m = generate_module(
        {
            "op_math_family": "kernel_methods",
            "op_kernel_feature_map": "positive_random_features",
        },
        dim=16,
    )
    assert isinstance(m, RandomFeatureKernelLane)


def test_dispatch_multiscale_math_knob() -> None:
    m = generate_module(
        {
            "op_math_family": "multiscale",
            "op_multiscale_transform": "causal_haar",
        },
        dim=16,
    )
    assert isinstance(m, MultiscaleWaveletLane)


def test_dispatch_graph_diffusion_math_knob() -> None:
    m = generate_module(
        {
            "op_math_family": "graph_diffusion",
            "op_graph_topology": "causal_path_laplacian",
        },
        dim=16,
    )
    assert isinstance(m, GraphDiffusionLane)


def test_dispatch_composes_math_knobs_over_base_lane() -> None:
    m = generate_module(
        {
            "op_algebraic_space": "tropical",
            "op_math_knobs": (
                "calculus_finite_difference",
                "linear_algebra_low_rank",
                "sparse_matrix_banded",
            ),
        },
        dim=16,
    )
    assert isinstance(m, SparseBandedAdapterLane)
    assert isinstance(m.base, LowRankAdapterLane)
    assert isinstance(m.base.base, CalculusAugmentedLane)
    assert isinstance(m.base.base.base, TropicalAttention)
    x = torch.randn(2, 8, 16)
    assert m(x).shape == x.shape


def test_dispatch_composes_new_math_knobs_over_base_lane() -> None:
    m = generate_module(
        {
            "op_algebraic_space": "tropical",
            "op_math_knobs": (
                "kernel_random_features",
                "multiscale_wavelet",
                "graph_laplacian_diffusion",
            ),
        },
        dim=16,
    )
    assert isinstance(m, GraphDiffusionAdapterLane)
    assert isinstance(m.base, MultiscaleWaveletAdapterLane)
    assert isinstance(m.base.base, RandomFeatureKernelAdapterLane)
    assert isinstance(m.base.base.base, TropicalAttention)
    x = torch.randn(2, 8, 16)
    assert m(x).shape == x.shape


def test_dispatch_default_fallback_linear() -> None:
    m = generate_module({"op_algebraic_space": "euclidean"}, dim=16)
    assert isinstance(m, nn.Linear)


def test_from_spec_round_trip() -> None:
    axes = {
        "op_algebraic_space": "tropical",
        "op_dynamical_has_state": 1,
        "op_dynamical_memory_length_class": "O(L)",
    }
    lifts = tuple(
        AxisLift(
            axis=k,
            value=v,
            n_ops=1,
            total_evals=1,
            total_s1_pass=0,
            pass_rate=0.5,
            representative_ops=(),
        )
        for k, v in axes.items()
    )
    candidate = CandidateTuple(
        tuple_values=tuple(axes.items()),
        predicted_lift=0.5,
        per_axis_lift=lifts,
        witness_ops=("tropical_attention",),
    )
    spec = spec_from_candidate(candidate)
    module = generate_module_from_spec(spec, dim=16)
    assert isinstance(module, TropicalStateSpace)
    x = torch.randn(1, 8, 16)
    y = module(x)
    assert y.shape == x.shape
