"""Block-template slot registry for generator dispatch."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from component_fab.generator.block_templates import (
    GatedParallelBlock,
    ThreeLaneAdaptive,
)
from component_fab.generator.primitive_templates import (
    ChebyshevSpectralLane,
    CliffordAttention,
    FisherAttention,
    FourierBasisLane,
    GraphDiffusionLane,
    LinearStateSpaceLane,
    LowRankFactorizedLane,
    MultiscaleWaveletLane,
    PoincareAttention,
    QuaternionAttention,
    RandomFeatureKernelLane,
    SparsemaxAttention,
    TropicalAttention,
    TuckerDecompLane,
)

LaneFactory = Callable[[int], nn.Module]


class UnknownBlockSlotError(ValueError):
    pass


def _two_lane_ts(dim: int) -> nn.Module:
    return GatedParallelBlock(
        lambda d: TropicalAttention(d),
        lambda d: SparsemaxAttention(d),
        dim,
    )


def _three_lane_tsw(dim: int) -> nn.Module:
    return ThreeLaneAdaptive(
        lambda d: TropicalAttention(d),
        lambda d: SparsemaxAttention(d),
        lambda d: MultiscaleWaveletLane(d),
        dim,
    )


def _local_window(dim: int) -> nn.Module:
    from component_fab.harness.top_ar_block import LocalWindowAttention

    return LocalWindowAttention(dim, window_size=16)


_SLOT_REGISTRY: dict[str, LaneFactory] = {
    "tropical_attention": TropicalAttention,
    "sparsemax_attention": SparsemaxAttention,
    "clifford_attention": lambda dim: (
        CliffordAttention(dim) if dim % 4 == 0 else nn.Linear(dim, dim)
    ),
    "linear_state_space": LinearStateSpaceLane,
    "multiscale_wavelet": MultiscaleWaveletLane,
    "fourier_basis": FourierBasisLane,
    "graph_diffusion": GraphDiffusionLane,
    "fisher_attention": FisherAttention,
    "chebyshev_spectral": ChebyshevSpectralLane,
    "tucker_decomp": TuckerDecompLane,
    "quaternion": lambda dim: (
        QuaternionAttention(dim) if dim % 4 == 0 else nn.Linear(dim, dim)
    ),
    "poincare": PoincareAttention,
    "random_features": RandomFeatureKernelLane,
    "low_rank": LowRankFactorizedLane,
    "tropical_sparsemax_two_lane": _two_lane_ts,
    "tropical_sparsemax_wavelet_three_lane": _three_lane_tsw,
    "local_window_attn": _local_window,
}


def build_block_slot_factory(name: str) -> LaneFactory:
    try:
        ctor = _SLOT_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_SLOT_REGISTRY))
        raise UnknownBlockSlotError(
            f"unknown block slot {name!r}; registered slots: {known}"
        ) from exc

    def factory(dim: int) -> nn.Module:
        return ctor(dim)

    return factory
