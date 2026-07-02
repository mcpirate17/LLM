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
from component_fab.generator.block_templates import LossMonsterPairedBlock
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
    HyperbolicAdapterLane,
    CliffordAdapterLane,
    CliffordAttention,
    CliffordRotorAdapterLane,
    CliffordRotorSandwichLane,
    LambdaFunctionalAdapterLane,
    LambdaFunctionalLane,
    LowRankAdapterLane,
    LowRankFactorizedLane,
    MultiscaleWaveletAdapterLane,
    MultiscaleWaveletLane,
    PadicAdapterLane,
    PadicProjection,
    PoincareAttention,
    RandomFeatureKernelAdapterLane,
    RandomFeatureKernelLane,
    SparseBandedAdapterLane,
    SparseBandedMatrixLane,
    TopKLinear,
    TropicalAdapterLane,
    TropicalAttention,
    TropicalStateSpace,
)
from component_fab.generator.routing_primitives import RecursionSite, SiteRecursionStack
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


def test_native_equivalent_surprise_dispatch_requires_parity_evidence() -> None:
    # A legacy mechanism with a native equivalent refuses to dispatch either path
    # until the spec carries per-spec parity evidence — no silent fall back to the
    # drifty Python lane or an unvalidated native lane.
    with pytest.raises(NativeParityEvidenceError, match="op_native_parity_passed"):
        generate_module({"op_invention_mechanism": "semiring_surprise_memory"}, dim=16)


def test_native_equivalent_surprise_dispatch_routes_after_parity_evidence() -> None:
    module = generate_module(
        {
            "op_invention_mechanism": "semiring_surprise_memory",
            "op_native_parity_passed": True,
            "op_native_parity_evidence": "component_fab/tests/test_native_surprise_memory.py",
        },
        dim=16,
    )
    assert isinstance(module, NativeSemiringSurpriseMemoryLane)

    rope = generate_module(
        {
            "op_invention_mechanism": "semiring_surprise_memory_rope",
            "op_native_parity_passed": "passed",
            "op_native_parity_evidence": "component_fab/tests/test_native_surprise_memory.py",
        },
        dim=16,
    )
    assert isinstance(rope, NativeSemiringRopeSurpriseMemoryLane)


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


@pytest.mark.parametrize(
    ("axes", "expected_type"),
    (
        (
            {
                "op_math_family": "calculus",
                "op_calculus_operator": "causal_laplacian",
            },
            FiniteDifferenceCalculusLane,
        ),
        (
            {
                "op_math_family": "linear_algebra",
                "op_linear_algebra_structure": "block_low_rank_factorized",
            },
            LowRankFactorizedLane,
        ),
        (
            {
                "op_math_family": "sparse_matrix",
                "op_sparse_matrix_pattern": "causal_dilated_banded",
            },
            SparseBandedMatrixLane,
        ),
        (
            {
                "op_math_family": "kernel_methods",
                "op_kernel_feature_map": "nystrom_landmarks",
            },
            RandomFeatureKernelLane,
        ),
        (
            {
                "op_math_family": "multiscale",
                "op_multiscale_transform": "laplacian_pyramid",
            },
            MultiscaleWaveletLane,
        ),
        (
            {
                "op_math_family": "graph_diffusion",
                "op_graph_topology": "learned_path_laplacian",
            },
            GraphDiffusionLane,
        ),
    ),
)
def test_dispatch_auto_deepened_math_knob_axis_aliases(
    axes: dict[str, str], expected_type: type[torch.nn.Module]
) -> None:
    module = generate_module(axes, dim=16)
    assert isinstance(module, expected_type)


def test_dispatch_lambda_functional_math_knob() -> None:
    m = generate_module(
        {
            "op_math_family": "lambda_functional",
            "op_lambda_transform": "learned_functional_blend",
            "op_lambda_gate": "state",
            "op_lambda_basis": "phase",
        },
        dim=16,
    )
    assert isinstance(m, LambdaFunctionalLane)
    x = torch.randn(2, 8, 16)
    y = m(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert float(torch.sigmoid(m.gate_logit).mean().detach()) < 0.05


def test_dispatch_lambda_functional_token_basis() -> None:
    m = generate_module(
        {
            "op_math_family": "lambda_functional",
            "op_lambda_transform": "learned_functional_blend",
            "op_lambda_gate": "content",
            "op_lambda_basis": "token",
        },
        dim=16,
    )
    assert isinstance(m, LambdaFunctionalLane)
    assert m.lambda_basis == "token"
    x = torch.randn(2, 8, 16)
    y = m(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_lambda_functional_token_basis_is_causal() -> None:
    m = LambdaFunctionalLane(16, gate="content", basis="token").eval()
    x = torch.randn(1, 8, 16)
    with torch.no_grad():
        base = m(x)
        perturbed = x.clone()
        perturbed[:, -1] += 5.0
        after = m(perturbed)
    # A causal token-domain basis must not let a future token change the prefix.
    assert torch.allclose(base[:, :-1], after[:, :-1], atol=1e-6)


def test_lambda_functional_gate_changes_output() -> None:
    m = LambdaFunctionalLane(16, gate="content", basis="dct")
    x = torch.randn(2, 8, 16)
    closed = m(x)
    with torch.no_grad():
        m.gate_logit.fill_(6.0)
    opened = m(x)
    assert (opened - closed).abs().mean().item() > 1e-4


def test_dispatch_exotic_algebra_math_knob_families() -> None:
    cases = [
        (
            {"op_math_family": "tropical", "op_tropical_adapter": "maxplus_read"},
            TropicalAttention,
        ),
        (
            {"op_math_family": "padic", "op_padic_adapter": "ultrametric_projection"},
            PadicProjection,
        ),
        (
            {
                "op_math_family": "clifford",
                "op_clifford_adapter": "geometric_product",
            },
            CliffordAttention,
        ),
        (
            {
                "op_math_family": "clifford",
                "op_clifford_adapter": "rotor_sandwich",
            },
            CliffordRotorSandwichLane,
        ),
        (
            {
                "op_math_family": "hyperbolic",
                "op_hyperbolic_adapter": "poincare_projection",
            },
            PoincareAttention,
        ),
    ]
    x = torch.randn(2, 8, 16)
    for axes, expected_type in cases:
        module = generate_module(axes, dim=16)
        assert isinstance(module, expected_type)
        y = module(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()


def test_dispatch_clifford_rotor_sandwich_is_identity_at_init() -> None:
    module = generate_module(
        {
            "op_math_family": "clifford",
            "op_clifford_adapter": "rotor_sandwich",
        },
        dim=16,
    )
    assert isinstance(module, CliffordRotorSandwichLane)
    x = torch.randn(2, 8, 16)

    closed = module(x)
    with torch.no_grad():
        module.rotor_angle.fill_(0.4)
    opened = module(x)

    assert torch.allclose(closed, x, atol=1e-6)
    assert not torch.allclose(opened, x)
    assert torch.isfinite(opened).all()


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


def test_dispatch_composes_auto_deepened_math_knobs_over_base_lane() -> None:
    module = generate_module(
        {
            "op_algebraic_space": "tropical",
            "op_math_knobs": (
                "calculus_laplacian",
                "linear_algebra_block_low_rank",
                "sparse_matrix_dilated_banded",
                "kernel_fourier_features",
                "multiscale_laplacian_pyramid",
                "graph_diffusion_learned_path",
                "lambda_functional_token_basis",
            ),
        },
        dim=16,
    )

    assert isinstance(module, LambdaFunctionalAdapterLane)
    assert isinstance(module.base, GraphDiffusionAdapterLane)
    assert isinstance(module.base.base, MultiscaleWaveletAdapterLane)
    assert isinstance(module.base.base.base, RandomFeatureKernelAdapterLane)
    assert isinstance(module.base.base.base.base, SparseBandedAdapterLane)
    assert isinstance(module.base.base.base.base.base, LowRankAdapterLane)
    assert isinstance(module.base.base.base.base.base.base, CalculusAugmentedLane)
    assert isinstance(module.base.base.base.base.base.base.base, TropicalAttention)
    x = torch.randn(2, 8, 16)
    assert module(x).shape == x.shape


def test_dispatch_composes_lambda_math_knob_over_base_lane() -> None:
    m = generate_module(
        {
            "op_algebraic_space": "tropical",
            "op_math_knobs": ("lambda_functional_blend",),
            "op_lambda_gate": "content",
            "op_lambda_basis": "valuation",
        },
        dim=16,
    )
    assert isinstance(m, LambdaFunctionalAdapterLane)
    assert isinstance(m.base, TropicalAttention)
    x = torch.randn(2, 8, 16)
    assert m(x).shape == x.shape


def test_dispatch_composes_exotic_algebra_knobs_over_hyperbolic_anchor() -> None:
    axes = {
        "op_algebraic_space": "hyperbolic",
        "op_math_knobs": ("tropical_knob", "padic_knob"),
    }
    stacked = generate_module(axes, dim=16)
    tropical_only = generate_module(
        {"op_algebraic_space": "hyperbolic", "op_math_knobs": ("tropical_knob",)},
        dim=16,
    )
    padic_only = generate_module(
        {"op_algebraic_space": "hyperbolic", "op_math_knobs": ("padic_knob",)},
        dim=16,
    )

    assert isinstance(stacked, PadicAdapterLane)
    assert isinstance(stacked.base, TropicalAdapterLane)
    assert isinstance(stacked.base.base, PoincareAttention)
    x = torch.randn(2, 8, 16)
    y_stack = stacked(x)
    y_tropical = tropical_only(x)
    y_padic = padic_only(x)
    assert y_stack.shape == x.shape
    assert torch.isfinite(y_stack).all()
    assert not torch.allclose(y_stack, y_tropical)
    assert not torch.allclose(y_stack, y_padic)


def test_dispatch_composes_clifford_and_hyperbolic_knobs() -> None:
    module = generate_module(
        {
            "op_algebraic_space": "tropical",
            "op_math_knobs": ("clifford_knob", "hyperbolic_knob"),
        },
        dim=16,
    )
    assert isinstance(module, HyperbolicAdapterLane)
    assert isinstance(module.base, CliffordAdapterLane)
    assert isinstance(module.base.base, TropicalAttention)
    x = torch.randn(2, 8, 16)
    assert torch.isfinite(module(x)).all()


def test_dispatch_composes_clifford_rotor_knob_over_base_lane() -> None:
    module = generate_module(
        {
            "op_algebraic_space": "tropical",
            "op_math_knobs": ("clifford_rotor_sandwich",),
        },
        dim=16,
    )
    assert isinstance(module, CliffordRotorAdapterLane)
    assert isinstance(module.base, TropicalAttention)
    x = torch.randn(2, 8, 16)
    y = module(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_dispatch_site_recursion_wraps_anchor_mixer() -> None:
    module = generate_module(
        {
            "op_algebraic_space": "tropical",
            "op_dynamical_has_state": 0,
            "op_routing_kind": "site_recursion",
            "op_recursion_sites": "mixer",
            "op_max_depth": 3,
        },
        dim=16,
    )
    assert isinstance(module, SiteRecursionStack)
    assert isinstance(module.stages[0], RecursionSite)
    x = torch.randn(2, 8, 16)
    y = module(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_dispatch_site_recursion_fails_loud_for_unsupported_sites() -> None:
    with pytest.raises(NotImplementedError, match="unsupported=\\['not_a_site'\\]"):
        generate_module(
            {
                "op_algebraic_space": "tropical",
                "op_routing_kind": "site_recursion",
                "op_recursion_sites": ("embedding", "not_a_site"),
            },
            dim=16,
        )


def test_dispatch_loss_monster_pair_builds_carrier_protected_block() -> None:
    module = generate_module(
        {
            "op_algebraic_space": "tropical",
            "op_block_template": "loss_monster_paired",
            "op_partner_kind": "slot_dplr",
            "op_block_slot_loss": "routed_bottleneck",
            "op_partner_floor": 0.5,
        },
        dim=16,
    )
    assert isinstance(module, LossMonsterPairedBlock)
    x = torch.randn(2, 8, 16)
    y = module(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert module.last_partner_frac is not None
    assert module.last_partner_frac >= 0.5


def test_dispatch_loss_monster_pair_rejects_unknown_partner_kind() -> None:
    with pytest.raises(ValueError, match="unknown loss-monster partner kind"):
        generate_module(
            {
                "op_algebraic_space": "tropical",
                "op_block_template": "loss_monster_paired",
                "op_partner_kind": "not_a_partner",
            },
            dim=16,
        )


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
