"""Tests for component_fab.generator.code_generator dispatch."""

from __future__ import annotations

import pytest
import torch

from component_fab.generator.code_generator import (
    NativeParityEvidenceError,
    UndispatchableSpecError,
    generate_module,
    generate_module_from_spec,
)
from component_fab.generator.memory_primitives import MultiHeadSlotTableMemoryLane
from component_fab.generator.native_surprise_memory import (
    NativeSemiringRopeSurpriseMemoryLane,
    NativeSemiringSurpriseMemoryLane,
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
from research.synthesis.parametric_ops import ParametricMix


def test_dispatch_tropical_no_state_to_attention() -> None:
    m = generate_module(
        {"op_algebraic_space": "tropical", "op_dynamical_has_state": 0}, dim=16
    )
    assert isinstance(m, TropicalAttention)


def test_dispatch_slot_table_memory_to_improved_multi_head_lane() -> None:
    module = generate_module({"op_invention_mechanism": "slot_table_memory"}, dim=16)
    assert isinstance(module, MultiHeadSlotTableMemoryLane)
    assert module.use_null_write
    assert module.use_composer
    assert not module.use_delta_update
    assert module.normalize_read
    assert module.route_from_input
    assert module.memory_dim == 12


@pytest.mark.parametrize("mechanism", ("legendre_ssm", "power_semiring_memory"))
def test_unimplemented_invention_stubs_fail_loud(mechanism: str) -> None:
    with pytest.raises(NotImplementedError, match="unimplemented stub"):
        generate_module({"op_invention_mechanism": mechanism}, dim=16)


def test_native_equivalent_surprise_dispatches_validated_native() -> None:
    # The legacy mechanism names carry a checked-in validated native lane in the
    # registry, so they route straight to the native C++ lane — no per-spec
    # evidence axes required, and the drifty Python lane is never selected.
    module = generate_module(
        {"op_invention_mechanism": "semiring_surprise_memory"}, dim=16
    )
    assert isinstance(module, NativeSemiringSurpriseMemoryLane)

    rope = generate_module(
        {"op_invention_mechanism": "semiring_surprise_memory_rope"}, dim=16
    )
    assert isinstance(rope, NativeSemiringRopeSurpriseMemoryLane)


def test_native_equivalent_without_validation_artifact_fails_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A native mapping with a missing proof is a wiring bug: refuse to dispatch
    # rather than silently fall back to the Python path or an unvalidated native.
    import component_fab.generator.code_generator as cg

    monkeypatch.setitem(
        cg._NATIVE_EQUIVALENT_MECHANISMS,
        "semiring_surprise_memory",
        ("native_semiring_surprise_memory", ""),
    )
    with pytest.raises(NativeParityEvidenceError, match="validation artifact"):
        generate_module({"op_invention_mechanism": "semiring_surprise_memory"}, dim=16)


def test_dispatch_physics_atom_program_from_axes() -> None:
    module = generate_module(
        {
            "op_search_track": "physics_atom",
            "op_physics_atom_kinds": "scan+basis",
            "op_physics_basis_axis": "token",
            "op_physics_address_family": "reciprocal",
            "op_physics_score_norm_family": "sharpen",
            "op_physics_aggregate_family": "semiring",
            "op_physics_knob_scale": 1.5,
            "op_physics_seed": 7,
        },
        dim=16,
    )
    assert isinstance(module[-1], ParametricMix)
    x = torch.randn(2, 8, 16)
    y = module(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


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


def test_undispatchable_spec_raises() -> None:
    # An un-dispatchable spec must fail loud, not silently become nn.Linear
    # (a linear stand-in looks gradeable but measures nothing).
    with pytest.raises(UndispatchableSpecError):
        generate_module({"op_algebraic_space": "euclidean"}, dim=16)


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
