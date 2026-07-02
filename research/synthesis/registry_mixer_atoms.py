"""Registry mixers as OpenDiscovery mixer stages.

The open-discovery archive saturates at ~8/243 physics niches on the
parametric-atom space (measured 2026-07-02: 6K and 20K iterations found the
same 8 niches) — coverage growth needs new MECHANISM families, not more
iterations. This module exposes the registry-wired novel cross-token mixers
(NM-C/NM-F) as drop-in mixer stages for ``ProgramSpec``.

Construction reuses the compiled-op param-init mixins verbatim via a throwaway
carrier, so the dim-adaptive knob logic (CDMA chips divisor, port-Hamiltonian
band clamp, oblique rank clamp, ...) lives in exactly one place. Only
CROSS-TOKEN ops are exposed: a channel-only op as the mixer stage cannot mix
tokens, floors the capability probe, and wastes archive slots.

The discovery knob randomizer (``_randomize_knobs``, markers incl. "scale"/
"gate"/"decay") opens each mixer's ReZero/gate parameters, so identity-at-init
mechanisms are probed in their ACTIVE regime.
"""

from __future__ import annotations

from torch import nn

from .compiled_op_params_compaction import CompactionOpParamInit
from .compiled_op_params_nmf import NMFOpParamInit

# Cross-token registry ops only (see module docstring).
REGISTRY_STAGE_OPS: tuple[str, ...] = (
    "token_merge_mix",
    "recurrent_depth_refine",
    "persistent_memory_refine",
    "lowrank_state_memory",
    "idempotent_oblique_memory",
    "nilpotent_lie_scan",
    "integral_control_mixer",
    "port_hamiltonian_mix",
    "cdma_slot_binding",
    "scale_equivariant_wavelet",
)


class _Carrier(CompactionOpParamInit, NMFOpParamInit, nn.Module):
    """Throwaway host for the param-init mixins: each ``_init_<op>`` attaches
    exactly one configured submodule, which we hand back."""

    def __init__(self) -> None:
        nn.Module.__init__(self)


def build_registry_mixer(op_name: str, dim: int) -> nn.Module:
    """Instantiate a registry mixer at ``dim`` with the seam's knob adaptation."""
    if op_name not in REGISTRY_STAGE_OPS:
        raise ValueError(
            f"unknown registry stage op {op_name!r}; known: {REGISTRY_STAGE_OPS}"
        )
    carrier = _Carrier()
    getattr(carrier, f"_init_{op_name}")({}, dim)
    children = list(carrier.children())
    if len(children) != 1:
        raise RuntimeError(
            f"_init_{op_name} attached {len(children)} submodules, expected 1"
        )
    return children[0]
