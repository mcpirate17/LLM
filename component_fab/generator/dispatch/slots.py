"""Block-template slot registry for generator dispatch."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

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
from component_fab.generator.routing_primitives import RoutedBottleneckLane

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


def _routed_bottleneck(dim: int) -> nn.Module:
    return RoutedBottleneckLane(
        (
            TropicalAttention,
            LinearStateSpaceLane,
            LowRankFactorizedLane,
            RandomFeatureKernelLane,
        ),
        dim,
        top_k=2,
    )


def _slot_table_memory(dim: int) -> nn.Module:
    from component_fab.generator.memory_primitives import MultiHeadSlotTableMemoryLane

    memory_dim = max(4, ((7 * dim) // 32) * 4)
    return MultiHeadSlotTableMemoryLane(
        dim,
        memory_dim=memory_dim,
        use_delta_update=False,
        route_from_input=True,
        normalize_slot_values=True,
    )


def _native_semiring_surprise(dim: int) -> nn.Module:
    from component_fab.generator.native_surprise_memory import (
        NativeSemiringSurpriseMemoryLane,
    )

    return NativeSemiringSurpriseMemoryLane(dim)


def _native_semiring_surprise_rope(dim: int) -> nn.Module:
    from component_fab.generator.native_surprise_memory import (
        NativeSemiringRopeSurpriseMemoryLane,
    )

    return NativeSemiringRopeSurpriseMemoryLane(dim)


def _hyper_mor_bilane(dim: int) -> nn.Module:
    from component_fab.generator.hyper_mor_bilane import (
        HyperbolicMoRSurpriseRefineMLPBiLane,
    )

    return HyperbolicMoRSurpriseRefineMLPBiLane(
        dim,
        memory_dim=max(8, min(dim, 32)),
        max_recursive_steps=4,
    )


PARTNER_KIND_SLOT_ALIASES: Final[dict[str, str]] = {
    "attention": "local_window_attn",
    "hyper_mor": "hyper_mor_bilane",
    "hyper_mor_b": "hyper_mor_bilane",
    "hyper_mor_bilane": "hyper_mor_bilane",
    "hyperbolic_mor": "hyper_mor_bilane",
    "native_semiring": "native_semiring_surprise_memory",
    "native_semiring_rope": "native_semiring_surprise_memory_rope",
    "native_semiring_surprise": "native_semiring_surprise_memory",
    "slot_dplr": "slot_table_memory",
    "slot_table": "slot_table_memory",
    "slot_table_memory": "slot_table_memory",
    "tropical_recall": "tropical_attention",
}


def slot_name_for_partner_kind(kind: str) -> str | None:
    """Return a registered block slot for a named carrier partner kind."""

    return PARTNER_KIND_SLOT_ALIASES.get(kind)


def known_partner_kinds() -> tuple[str, ...]:
    return tuple(sorted(PARTNER_KIND_SLOT_ALIASES))


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
    "routed_bottleneck": _routed_bottleneck,
    "slot_table_memory": _slot_table_memory,
    "native_semiring_surprise_memory": _native_semiring_surprise,
    "native_semiring_surprise_memory_rope": _native_semiring_surprise_rope,
    "hyper_mor_bilane": _hyper_mor_bilane,
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
