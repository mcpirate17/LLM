"""Parametric lane builders for scaling_blimp_study.

Contains the regex-dispatched MoR/bilane builders and the native/non-native
surprise-memory family builders.  Imported by _scaling_lanes; not a public API.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from torch import nn


# ---------------------------------------------------------------------------
# Native surprise-memory family
# ---------------------------------------------------------------------------

_NATIVE_SURPRISE_NAMES = frozenset(
    {
        "native_read_before_write_surprise_memory",
        "native_context_gated_surprise_memory",
        "native_atlas_poly_surprise_memory",
        "native_titans_mac_surprise_memory",
        "native_semiring_surprise_memory",
        "native_semiring_surprise_memory_rope",
        "native_semiring_titans_mac_surprise_memory",
        "native_semiring_rope_titans_mac_surprise_memory",
        "native_balanced_semiring_titans_mac_surprise_memory",
        "native_balanced_semiring_rope_titans_mac_surprise_memory",
        "native_balanced_semiring_bilane_surprise_memory",
        "native_balanced_semiring_trilane_surprise_memory",
        "native_adaptive_semiring_rope_titans_mac_surprise_memory",
        "native_adaptive_semiring_bilane_surprise_memory",
    }
)

_NON_NATIVE_SURPRISE_NAMES = frozenset(
    {
        "tropical_surprise_memory",
        "semiring_surprise_memory",
        "semiring_surprise_memory_rope",
    }
)

ALL_SURPRISE_NAMES = _NATIVE_SURPRISE_NAMES | _NON_NATIVE_SURPRISE_NAMES


def build_native_surprise(name: str, top_k_frac: float) -> Callable[[int], nn.Module]:
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

    _map: dict[str, Callable[[int], nn.Module]] = {
        "native_read_before_write_surprise_memory": lambda d: (
            NativeReadBeforeWriteSurpriseMemoryLane(d)
        ),
        "native_context_gated_surprise_memory": lambda d: (
            NativeContextGatedSurpriseMemoryLane(d)
        ),
        "native_atlas_poly_surprise_memory": lambda d: (
            NativeAtlasPolySurpriseMemoryLane(d)
        ),
        "native_semiring_surprise_memory": lambda d: NativeSemiringSurpriseMemoryLane(
            d
        ),
        "native_semiring_surprise_memory_rope": lambda d: (
            NativeSemiringRopeSurpriseMemoryLane(d)
        ),
        "native_semiring_titans_mac_surprise_memory": lambda d: (
            NativeSemiringTitansMACSurpriseMemoryLane(d)
        ),
        "native_semiring_rope_titans_mac_surprise_memory": lambda d: (
            NativeSemiringRopeTitansMACSurpriseMemoryLane(d)
        ),
        "native_balanced_semiring_titans_mac_surprise_memory": lambda d: (
            NativeBalancedSemiringTitansMACSurpriseMemoryLane(d)
        ),
        "native_balanced_semiring_rope_titans_mac_surprise_memory": lambda d: (
            NativeBalancedSemiringRopeTitansMACSurpriseMemoryLane(d)
        ),
        "native_balanced_semiring_bilane_surprise_memory": lambda d: (
            NativeBalancedSemiringBiLaneSurpriseMemoryLane(d)
        ),
        "native_balanced_semiring_trilane_surprise_memory": lambda d: (
            NativeBalancedSemiringTriLaneSurpriseMemoryLane(d)
        ),
        "native_adaptive_semiring_rope_titans_mac_surprise_memory": lambda d: (
            NativeAdaptiveSemiringRopeTitansMACSurpriseMemoryLane(d)
        ),
        "native_adaptive_semiring_bilane_surprise_memory": lambda d: (
            NativeAdaptiveSemiringBiLaneSurpriseMemoryLane(d)
        ),
    }
    if name in _map:
        return _map[name]
    # Default: native_titans_mac_surprise_memory
    return lambda d: NativeTitansMACSurpriseMemoryLane(d)


def build_non_native_surprise(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    from component_fab.generator.memory_primitives import (
        SemiringSurpriseMemoryLane,
        TropicalSurpriseMemoryLane,
    )

    # compile_step is INTENTIONALLY OFF: per-step torch.compile of _delta_step
    # is 2.8x in an isolated fixed-shape benchmark, but in the live 12-block
    # model it re-traces forever (GPU idle, 1 core pegged, hung at step 480
    # for 45min) because the batch shape differs between train (b16) and the
    # gMQAR/BLiMP eval batches. Eager (~3.5k tok/s) is the reliable path until
    # a chunkwise-parallel kernel exists.
    if name == "tropical_surprise_memory":
        return lambda d: TropicalSurpriseMemoryLane(d)
    rope = name.endswith("_rope")
    return lambda d: SemiringSurpriseMemoryLane(d, use_rope=rope)


# ---------------------------------------------------------------------------
# Regex-pattern builders (matched against name before the dict lookup)
# ---------------------------------------------------------------------------


def _mor_lane_kwargs(g: re.Match) -> dict[str, Any]:
    """Shared kwargs from a MoR-composite regex match (groups 2-10)."""
    if g.group(8) is None:
        low_t, high_t = float(g.group(6)) / 100.0, float(g.group(7)) / 100.0
    else:
        low_t, high_t = float(g.group(8)) / 10000.0, float(g.group(9)) / 10000.0
    return {
        "memory_dim": int(g.group(2)),
        "gate_bias": float(g.group(3)),
        "semiring_temp_init": float(g.group(4)),
        "recursive_balance_init": float(g.group(5)),
        "low_threshold": low_t,
        "high_threshold": high_t,
        "max_recursive_steps": int(g.group(10)),
    }


def _build_mor_mlp_subclass(
    m: re.Match, base_cls: type[nn.Module], name_prefix: str
) -> Callable[[int], nn.Module]:
    router_hidden = int(m.group(1))
    # ROUTER_HIDDEN is read in _make_lane_a during __init__, so bake it into a
    # per-width subclass rather than setting an instance attr (too late).
    cls = type(
        f"{name_prefix}{router_hidden}Bilane",
        (base_cls,),
        {"ROUTER_HIDDEN": router_hidden},
    )
    kwargs = _mor_lane_kwargs(m)
    return lambda d: cls(d, **kwargs)


def build_mor_surprise_composite(
    m: re.Match, top_k_frac: float
) -> Callable[[int], nn.Module]:
    from component_fab.generator.mor_bilane import (
        MoRSurpriseRefineMLPAdaptiveSemiringBiLaneSurpriseMemoryLane,
    )

    return _build_mor_mlp_subclass(
        m,
        MoRSurpriseRefineMLPAdaptiveSemiringBiLaneSurpriseMemoryLane,
        "MoRSurpriseRefineMLP",
    )


def build_mor_mlp_composite(
    m: re.Match, top_k_frac: float
) -> Callable[[int], nn.Module]:
    from component_fab.generator.mor_bilane import (
        MoRRefineMLPAdaptiveSemiringBiLaneSurpriseMemoryLane,
    )

    return _build_mor_mlp_subclass(
        m, MoRRefineMLPAdaptiveSemiringBiLaneSurpriseMemoryLane, "MoRRefineMLP"
    )


def build_mor_adaptive_composite(
    m: re.Match, top_k_frac: float
) -> Callable[[int], nn.Module]:
    from component_fab.generator.mor_bilane import (
        MoRAdaptiveSemiringBiLaneSurpriseMemoryLane,
        MoRRefineAdaptiveSemiringBiLaneSurpriseMemoryLane,
    )

    cls = (
        MoRRefineAdaptiveSemiringBiLaneSurpriseMemoryLane
        if m.group(1) == "mor_refine"  # group 1 is the mor/mor_refine prefix
        else MoRAdaptiveSemiringBiLaneSurpriseMemoryLane
    )
    kwargs = _mor_lane_kwargs(m)
    return lambda d: cls(d, **kwargs)


def _build_adaptive_semiring_lane(
    m: re.Match, lane_cls: type[nn.Module]
) -> Callable[[int], nn.Module]:
    memory_dim = int(m.group(1))
    gate_bias = float(m.group(2))
    semiring_temp_init = float(m.group(3))
    recursive_balance_init = float(m.group(4))
    if m.group(7) is None:
        low_threshold = float(m.group(5)) / 100.0
        high_threshold = float(m.group(6)) / 100.0
    else:
        low_threshold = float(m.group(7)) / 10000.0
        high_threshold = float(m.group(8)) / 10000.0
    max_recursive_steps = int(m.group(9))
    return lambda d: lane_cls(
        d,
        memory_dim=memory_dim,
        gate_bias=gate_bias,
        semiring_temp_init=semiring_temp_init,
        recursive_balance_init=recursive_balance_init,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        max_recursive_steps=max_recursive_steps,
    )


def build_tuned_adaptive_composite(
    m: re.Match, top_k_frac: float
) -> Callable[[int], nn.Module]:
    from component_fab.generator.native_surprise_memory import (
        NativeAdaptiveSemiringBiLaneSurpriseMemoryLane,
    )

    return _build_adaptive_semiring_lane(
        m, NativeAdaptiveSemiringBiLaneSurpriseMemoryLane
    )


def build_tuned_adaptive_semiring_mac(
    m: re.Match, top_k_frac: float
) -> Callable[[int], nn.Module]:
    from component_fab.generator.native_surprise_memory import (
        NativeAdaptiveSemiringRopeTitansMACSurpriseMemoryLane,
    )

    return _build_adaptive_semiring_lane(
        m, NativeAdaptiveSemiringRopeTitansMACSurpriseMemoryLane
    )


def build_tuned_balanced_composite(
    m: re.Match, top_k_frac: float
) -> Callable[[int], nn.Module]:
    from component_fab.generator.native_surprise_memory import (
        NativeBalancedSemiringBiLaneSurpriseMemoryLane,
        NativeBalancedSemiringTriLaneSurpriseMemoryLane,
    )

    lane_cls = (
        NativeBalancedSemiringBiLaneSurpriseMemoryLane
        if m.group(1) == "bi"
        else NativeBalancedSemiringTriLaneSurpriseMemoryLane
    )
    memory_dim = int(m.group(2))
    gate_bias = float(m.group(3))
    semiring_temp_init = float(m.group(4))
    recursive_balance_init = float(m.group(5))
    return lambda d: lane_cls(
        d,
        memory_dim=memory_dim,
        gate_bias=gate_bias,
        semiring_temp_init=semiring_temp_init,
        recursive_balance_init=recursive_balance_init,
    )


def build_tuned_semiring_mac(
    m: re.Match, top_k_frac: float
) -> Callable[[int], nn.Module]:
    from component_fab.generator.native_surprise_memory import (
        NativeBalancedSemiringRopeTitansMACSurpriseMemoryLane,
        NativeBalancedSemiringTitansMACSurpriseMemoryLane,
        NativeSemiringRopeTitansMACSurpriseMemoryLane,
        NativeSemiringTitansMACSurpriseMemoryLane,
    )

    use_rope = m.group(1) is not None
    balanced = m.group(2) is not None
    qk_norm = m.group(3) is not None
    memory_dim = int(m.group(4))
    gate_bias = float(m.group(5))
    semiring_temp_init = float(m.group(6))
    recursive_balance_init = float(m.group(7) or 1)
    if balanced and use_rope:
        lane_cls = NativeBalancedSemiringRopeTitansMACSurpriseMemoryLane
    elif balanced:
        lane_cls = NativeBalancedSemiringTitansMACSurpriseMemoryLane
    elif use_rope:
        lane_cls = NativeSemiringRopeTitansMACSurpriseMemoryLane
    else:
        lane_cls = NativeSemiringTitansMACSurpriseMemoryLane
    return lambda d: lane_cls(
        d,
        memory_dim=memory_dim,
        gate_bias=gate_bias,
        semiring_temp_init=semiring_temp_init,
        qk_norm=qk_norm,
        **({"recursive_balance_init": recursive_balance_init} if balanced else {}),
    )


# ---------------------------------------------------------------------------
# Compiled regex table consumed by _scaling_lanes._build_lane_factory
# ---------------------------------------------------------------------------

REGEX_BUILDERS: list[
    tuple[re.Pattern, Callable[[re.Match, float], Callable[[int], nn.Module]]]
] = [
    (
        re.compile(
            r"mor_surprise_refine_mlp(\d+)_native_semiring_adapt_bilane_m(\d+)_g(-?\d+)_t(\d+)_b(\d+)_(?:l(\d+)_h(\d+)|l(\d+)bp_h(\d+)bp)_r(\d+)_surprise_memory"
        ),
        build_mor_surprise_composite,
    ),
    (
        re.compile(
            r"mor_refine_mlp(\d+)_native_semiring_adapt_bilane_m(\d+)_g(-?\d+)_t(\d+)_b(\d+)_(?:l(\d+)_h(\d+)|l(\d+)bp_h(\d+)bp)_r(\d+)_surprise_memory"
        ),
        build_mor_mlp_composite,
    ),
    (
        re.compile(
            r"(mor|mor_refine)_native_semiring_adapt_bilane_m(\d+)_g(-?\d+)_t(\d+)_b(\d+)_(?:l(\d+)_h(\d+)|l(\d+)bp_h(\d+)bp)_r(\d+)_surprise_memory"
        ),
        build_mor_adaptive_composite,
    ),
    (
        re.compile(
            r"native_semiring_adapt_bilane_m(\d+)_g(-?\d+)_t(\d+)_b(\d+)_(?:l(\d+)_h(\d+)|l(\d+)bp_h(\d+)bp)_r(\d+)_surprise_memory"
        ),
        build_tuned_adaptive_composite,
    ),
    (
        re.compile(
            r"native_semiring_rope_titans_mac_adapt_m(\d+)_g(-?\d+)_t(\d+)_b(\d+)_(?:l(\d+)_h(\d+)|l(\d+)bp_h(\d+)bp)_r(\d+)_surprise_memory"
        ),
        build_tuned_adaptive_semiring_mac,
    ),
    (
        re.compile(
            r"native_semiring_bal_(bi|tri)lane_m(\d+)_g(-?\d+)_t(\d+)_b(\d+)_surprise_memory"
        ),
        build_tuned_balanced_composite,
    ),
    (
        re.compile(
            r"native_semiring_(rope_)?titans_mac_(bal_)?(qkn_)?m(\d+)_g(-?\d+)_t(\d+)(?:_b(\d+))?_surprise_memory"
        ),
        build_tuned_semiring_mac,
    ),
]
