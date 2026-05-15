"""Dispatcher — ProposalSpec → runnable nn.Module.

Maps the spec's math axes + synthesis_kind onto the right
``primitive_templates`` class. Returns an instantiated module ready to
plug into the standard test harness.

Dispatch order (first match wins): explicit ``op_math_family`` knobs come
first because they name concrete operator mechanisms; algebraic_space comes
before sparsity otherwise because algebra determines the underlying math.
E.g. ``tropical + state + top_k`` should materialize as
``TropicalTopKStateSpace``, not bypass to ``TopKLinear``.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from ..proposer.spec_generator import ProposalSpec
from .primitive_templates import (
    CalculusAugmentedLane,
    CausalFastWeightMemoryLane,
    CausalSlotRouterMemoryLane,
    CliffordAttention,
    FiniteDifferenceCalculusLane,
    FourierBasisLane,
    GraphDiffusionAdapterLane,
    GraphDiffusionLane,
    HierarchicalResidualCompressorLane,
    LinearStateSpaceLane,
    LowRankAdapterLane,
    LowRankFactorizedLane,
    MultiscaleWaveletAdapterLane,
    MultiscaleWaveletLane,
    PadicProjection,
    RandomFeatureKernelAdapterLane,
    RandomFeatureKernelLane,
    SparseBandedAdapterLane,
    SparseBandedMatrixLane,
    SpikingActivationGate,
    SymplecticResidualMixerLane,
    TopKLinear,
    TropicalAttention,
    TropicalStateSpace,
    TropicalTopKStateSpace,
)


def _axis(math_axes: dict[str, Any], key: str) -> str:
    value = math_axes.get(key)
    return "" if value is None else str(value)


def _has_state(math_axes: dict[str, Any]) -> bool:
    return bool(int(math_axes.get("op_dynamical_has_state") or 0))


def _dispatch_tropical(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float
) -> nn.Module | None:
    if _axis(math_axes, "op_algebraic_space") != "tropical":
        return None
    sparsity = _axis(math_axes, "op_activation_sparsity_pattern")
    if _has_state(math_axes) and sparsity == "top_k":
        k = max(1, int(round(dim * top_k_frac)))
        return TropicalTopKStateSpace(dim, k=k)
    if _has_state(math_axes):
        return TropicalStateSpace(dim)
    return TropicalAttention(dim)


def _dispatch_clifford(math_axes: dict[str, Any], *, dim: int) -> nn.Module | None:
    if _axis(math_axes, "op_algebraic_space") != "clifford":
        return None
    if dim % 4 != 0:
        return nn.Linear(dim, dim)
    return CliffordAttention(dim)


def _dispatch_spiking(math_axes: dict[str, Any], *, dim: int) -> nn.Module | None:
    if _axis(math_axes, "op_algebraic_space") != "spiking":
        return None
    return SpikingActivationGate(dim)


def _dispatch_padic(math_axes: dict[str, Any], *, dim: int) -> nn.Module | None:
    if _axis(math_axes, "op_algebraic_space") != "padic":
        return None
    if dim % 8 != 0:
        return nn.Linear(dim, dim)
    return PadicProjection(dim, p=2, n_levels=3)


def _dispatch_state_kernel(math_axes: dict[str, Any], *, dim: int) -> nn.Module | None:
    """Generic state-bearing primitive for non-tropical / non-clifford / non-padic
    proposals declaring ``op_dynamical_has_state=1``. Algebra-specific
    state primitives (TropicalStateSpace, etc.) already fired earlier in
    the dispatch chain, so reaching here means no domain module matched.
    """
    if not _has_state(math_axes):
        return None
    algebra = _axis(math_axes, "op_algebraic_space")
    if algebra in ("tropical", "clifford", "spiking", "padic"):
        return None
    return LinearStateSpaceLane(dim)


def _dispatch_axis_modifier(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float
) -> nn.Module | None:
    sparsity = _axis(math_axes, "op_activation_sparsity_pattern")
    if sparsity == "top_k":
        k = max(1, int(round(dim * top_k_frac)))
        return TopKLinear(dim, dim, k=k)
    basis = _axis(math_axes, "op_spectral_preferred_basis")
    if basis in ("fourier", "frequency"):
        return FourierBasisLane(dim)
    return None


def _dispatch_math_knob(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float
) -> nn.Module | None:
    family = _axis(math_axes, "op_math_family")
    if family == "calculus":
        operator = _axis(math_axes, "op_calculus_operator")
        if operator in ("causal_finite_difference_integral", "finite_difference"):
            return FiniteDifferenceCalculusLane(dim)
    if family == "linear_algebra":
        structure = _axis(math_axes, "op_linear_algebra_structure")
        if structure in ("low_rank_factorized", "low_rank"):
            rank = max(1, int(round(dim * top_k_frac)))
            return LowRankFactorizedLane(dim, rank=rank)
    if family == "sparse_matrix":
        pattern = _axis(math_axes, "op_sparse_matrix_pattern")
        if pattern in ("causal_banded", "banded"):
            bandwidth = max(1, min(dim, int(round(dim * top_k_frac))))
            return SparseBandedMatrixLane(dim, bandwidth=bandwidth)
    if family == "kernel_methods":
        kernel = _axis(math_axes, "op_kernel_feature_map")
        if kernel in ("positive_random_features", "random_features"):
            n_features = max(4, int(round(dim * 0.5)))
            return RandomFeatureKernelLane(dim, n_features=n_features)
    if family == "multiscale":
        transform = _axis(math_axes, "op_multiscale_transform")
        if transform in ("causal_haar", "wavelet"):
            return MultiscaleWaveletLane(dim)
    if family == "graph_diffusion":
        topology = _axis(math_axes, "op_graph_topology")
        if topology in ("causal_path_laplacian", "causal_path"):
            return GraphDiffusionLane(dim)
    return None


def _dispatch_invention_mechanism(
    math_axes: dict[str, Any], *, dim: int
) -> nn.Module | None:
    mechanism = _axis(math_axes, "op_invention_mechanism")
    if mechanism == "causal_fast_weight_memory":
        return CausalFastWeightMemoryLane(dim)
    if mechanism == "causal_slot_router_memory":
        return CausalSlotRouterMemoryLane(dim)
    if mechanism == "hierarchical_residual_compressor":
        return HierarchicalResidualCompressorLane(dim)
    if mechanism == "symplectic_residual_mixer":
        if dim % 2 != 0:
            return nn.Linear(dim, dim)
        return SymplecticResidualMixerLane(dim)
    return None


def _math_knobs(math_axes: dict[str, Any]) -> tuple[str, ...]:
    raw = math_axes.get("op_math_knobs")
    if raw is None:
        family = _axis(math_axes, "op_math_family")
        if family == "calculus":
            return ("calculus_finite_difference",)
        if family == "linear_algebra":
            return ("linear_algebra_low_rank",)
        if family == "sparse_matrix":
            return ("sparse_matrix_banded",)
        if family == "kernel_methods":
            return ("kernel_random_features",)
        if family == "multiscale":
            return ("multiscale_wavelet",)
        if family == "graph_diffusion":
            return ("graph_laplacian_diffusion",)
        return ()
    if isinstance(raw, str):
        return tuple(part.strip() for part in raw.split("+") if part.strip())
    if isinstance(raw, (list, tuple)):
        return tuple(str(part) for part in raw if str(part))
    return ()


def _base_module(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float
) -> nn.Module:
    for dispatcher in (
        lambda: _dispatch_tropical(math_axes, dim=dim, top_k_frac=top_k_frac),
        lambda: _dispatch_clifford(math_axes, dim=dim),
        lambda: _dispatch_spiking(math_axes, dim=dim),
        lambda: _dispatch_padic(math_axes, dim=dim),
        lambda: _dispatch_state_kernel(math_axes, dim=dim),
        lambda: _dispatch_axis_modifier(math_axes, dim=dim, top_k_frac=top_k_frac),
    ):
        result = dispatcher()
        if result is not None:
            return result
    return nn.Linear(dim, dim)


def _apply_math_knobs(
    module: nn.Module,
    math_axes: dict[str, Any],
    *,
    dim: int,
    top_k_frac: float,
) -> nn.Module:
    rank = max(1, int(round(dim * top_k_frac)))
    bandwidth = max(1, min(dim, int(round(dim * top_k_frac))))
    n_features = max(4, int(round(dim * 0.5)))
    for knob in _math_knobs(math_axes):
        if knob == "calculus_finite_difference":
            module = CalculusAugmentedLane(module, dim)
        elif knob == "linear_algebra_low_rank":
            module = LowRankAdapterLane(module, dim, rank=rank)
        elif knob == "sparse_matrix_banded":
            module = SparseBandedAdapterLane(module, dim, bandwidth=bandwidth)
        elif knob == "kernel_random_features":
            module = RandomFeatureKernelAdapterLane(module, dim, n_features=n_features)
        elif knob == "multiscale_wavelet":
            module = MultiscaleWaveletAdapterLane(module, dim)
        elif knob == "graph_laplacian_diffusion":
            module = GraphDiffusionAdapterLane(module, dim)
    return module


def generate_module(
    math_axes: dict[str, Any],
    *,
    dim: int = 32,
    top_k_frac: float = 0.25,
) -> nn.Module:
    """Generate a primitive instance from a math-axis tuple."""
    invention = _dispatch_invention_mechanism(math_axes, dim=dim)
    if invention is not None:
        return invention
    if math_axes.get("op_math_knobs") is not None:
        base = _base_module(math_axes, dim=dim, top_k_frac=top_k_frac)
        return _apply_math_knobs(base, math_axes, dim=dim, top_k_frac=top_k_frac)
    for dispatcher in (
        lambda: _dispatch_math_knob(math_axes, dim=dim, top_k_frac=top_k_frac),
        lambda: _base_module(math_axes, dim=dim, top_k_frac=top_k_frac),
    ):
        result = dispatcher()
        if result is not None:
            return result
    raise RuntimeError("unreachable module dispatch state")


def generate_module_from_spec(
    spec: ProposalSpec, *, dim: int = 32, top_k_frac: float = 0.25
) -> nn.Module:
    return generate_module(spec.math_axes, dim=dim, top_k_frac=top_k_frac)
