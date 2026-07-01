"""Invention-mechanism dispatch registry for generated fab modules."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from torch import nn

from component_fab.generator.memory_primitives import (
    CausalFastWeightMemoryLane,
    CausalSlotRouterMemoryLane,
    DataDependentDecayMemoryLane,
    HierarchicalResidualCompressorLane,
    LegendreSSMLane,
    MultiHeadSlotTableMemoryLane,
    PadicSurpriseMemoryLane,
    PowerSemiringMemoryLane,
    SemiringSurpriseMemoryLane,
    TropicalSurpriseMemoryLane,
)
from component_fab.generator.novel_math_primitives import (
    FractionalIntegralMemoryLane,
    MeraRenormMixerLane,
    SheafDiffusionMixerLane,
)
from component_fab.generator.native_surprise_memory import (
    NativeAdaptiveSemiringBiLaneSurpriseMemoryLane,
    NativeAdaptiveSemiringRopeTitansMACSurpriseMemoryLane,
    NativeAtlasPolySurpriseMemoryLane,
    NativeBalancedSemiringBiLaneSurpriseMemoryLane,
    NativeBalancedSemiringRopeTitansMACSurpriseMemoryLane,
    NativeBalancedSemiringTitansMACSurpriseMemoryLane,
    NativeBalancedSemiringTriLaneSurpriseMemoryLane,
    NativeContextGatedSurpriseMemoryLane,
    NativeReadBeforeWriteSurpriseMemoryLane,
    NativeSemiringRopeSurpriseMemoryLane,
    NativeSemiringRopeTitansMACSurpriseMemoryLane,
    NativeSemiringSurpriseMemoryLane,
    NativeSemiringTitansMACSurpriseMemoryLane,
    NativeTitansMACSurpriseMemoryLane,
)
from component_fab.generator.primitive_templates import SymplecticResidualMixerLane

LaneFactory = Callable[[int], nn.Module]


class NativeParityEvidenceError(ValueError):
    """A spec requested a legacy surprise lane with a native replacement."""


def _axis(math_axes: dict[str, Any], key: str) -> str:
    value = math_axes.get(key)
    return "" if value is None else str(value)


def _slot_table_memory_lane(dim: int) -> nn.Module:
    memory_dim = max(4, ((7 * dim) // 32) * 4)
    return MultiHeadSlotTableMemoryLane(
        dim,
        memory_dim=memory_dim,
        use_delta_update=False,
        route_from_input=True,
        normalize_slot_values=True,
    )


def _symplectic_residual_mixer_lane(dim: int) -> nn.Module:
    if dim % 2 != 0:
        return nn.Linear(dim, dim)
    return SymplecticResidualMixerLane(dim)


_INVENTION_MECHANISMS: dict[str, LaneFactory] = {
    "causal_fast_weight_memory": CausalFastWeightMemoryLane,
    "data_dependent_decay_memory": DataDependentDecayMemoryLane,
    "power_semiring_memory": PowerSemiringMemoryLane,
    "legendre_ssm": LegendreSSMLane,
    "slot_table_memory": _slot_table_memory_lane,
    "causal_slot_router_memory": CausalSlotRouterMemoryLane,
    "hierarchical_residual_compressor": HierarchicalResidualCompressorLane,
    "symplectic_residual_mixer": _symplectic_residual_mixer_lane,
    "tropical_surprise_memory": TropicalSurpriseMemoryLane,
    "semiring_surprise_memory": SemiringSurpriseMemoryLane,
    "semiring_surprise_memory_rope": partial(SemiringSurpriseMemoryLane, use_rope=True),
    "padic_surprise_memory": PadicSurpriseMemoryLane,
    "fractional_integral_memory": FractionalIntegralMemoryLane,
    "sheaf_consistent_slot_mixer": SheafDiffusionMixerLane,
    "mera_block": MeraRenormMixerLane,
    "native_read_before_write_surprise_memory": NativeReadBeforeWriteSurpriseMemoryLane,
    "native_context_gated_surprise_memory": NativeContextGatedSurpriseMemoryLane,
    "native_atlas_poly_surprise_memory": NativeAtlasPolySurpriseMemoryLane,
    "native_titans_mac_surprise_memory": NativeTitansMACSurpriseMemoryLane,
    "native_semiring_surprise_memory": NativeSemiringSurpriseMemoryLane,
    "native_semiring_surprise_memory_rope": NativeSemiringRopeSurpriseMemoryLane,
    "native_semiring_titans_mac_surprise_memory": NativeSemiringTitansMACSurpriseMemoryLane,
    "native_semiring_rope_titans_mac_surprise_memory": NativeSemiringRopeTitansMACSurpriseMemoryLane,
    "native_balanced_semiring_titans_mac_surprise_memory": NativeBalancedSemiringTitansMACSurpriseMemoryLane,
    "native_balanced_semiring_rope_titans_mac_surprise_memory": NativeBalancedSemiringRopeTitansMACSurpriseMemoryLane,
    "native_balanced_semiring_bilane_surprise_memory": NativeBalancedSemiringBiLaneSurpriseMemoryLane,
    "native_balanced_semiring_trilane_surprise_memory": NativeBalancedSemiringTriLaneSurpriseMemoryLane,
    "native_adaptive_semiring_rope_titans_mac_surprise_memory": NativeAdaptiveSemiringRopeTitansMACSurpriseMemoryLane,
    "native_adaptive_semiring_bilane_surprise_memory": NativeAdaptiveSemiringBiLaneSurpriseMemoryLane,
}

_NATIVE_EQUIVALENT_MECHANISMS: dict[str, str] = {
    "semiring_surprise_memory": "native_semiring_surprise_memory",
    "semiring_surprise_memory_rope": "native_semiring_surprise_memory_rope",
}


def _truthy_axis(math_axes: dict[str, Any], key: str) -> bool:
    raw = math_axes.get(key)
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y", "passed", "pass"}
    return bool(raw)


def _has_native_parity_evidence(math_axes: dict[str, Any]) -> bool:
    evidence = str(math_axes.get("op_native_parity_evidence") or "").strip()
    return _truthy_axis(math_axes, "op_native_parity_passed") and bool(evidence)


def dispatch_invention_mechanism(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float = 0.25
) -> nn.Module | None:
    """Build an invention-mechanism module, or ``None`` when no mechanism matches."""

    del top_k_frac
    mechanism = _axis(math_axes, "op_invention_mechanism")
    native_mechanism = _NATIVE_EQUIVALENT_MECHANISMS.get(mechanism)
    if native_mechanism is not None:
        if not _has_native_parity_evidence(math_axes):
            raise NativeParityEvidenceError(
                f"{mechanism!r} has native equivalent {native_mechanism!r}; "
                "refusing to dispatch either path without "
                "op_native_parity_passed=True and op_native_parity_evidence"
            )
        mechanism = native_mechanism
    factory = _INVENTION_MECHANISMS.get(mechanism)
    return factory(dim) if factory is not None else None
