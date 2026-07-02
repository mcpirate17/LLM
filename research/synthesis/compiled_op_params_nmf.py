"""Param-init mixin for the NM-F operator families.

Each ``_init_*`` instantiates the full self-contained NM-F class as a compiled
submodule (the NM-C / ``pq_embedding_moe_block`` convention). Knob defaults
carrying a divisibility/capacity constraint are adapted to ``d_in`` fail-soft
(the op must initialise at any model dim the search proposes); everything else
fails loud. Forward handlers live in ``compiler_ops_nmf.py``.

Consumed by ``CompiledOpParamInitMixin`` via inheritance.
"""

from __future__ import annotations

from typing import Dict


class NMFOpParamInit:
    @staticmethod
    def _nmf_largest_divisor_at_most(d_in: int, cap: int) -> int:
        for candidate in range(min(cap, d_in), 0, -1):
            if d_in % candidate == 0:
                return candidate
        return 1

    def _init_idempotent_oblique_memory(self, config: Dict, d_in: int) -> None:
        from .idempotent_oblique_memory import IdempotentObliqueMemory

        cfg = config or {}
        rank = min(int(cfg.get("rank", 4)), d_in)
        self.oblique_memory_block = IdempotentObliqueMemory(d_in, rank=max(1, rank))

    def _init_nilpotent_lie_scan(self, config: Dict, d_in: int) -> None:
        from .nilpotent_lie_scan import NilpotentLieScan

        cfg = config or {}
        self.lie_scan_block = NilpotentLieScan(
            d_in, lift_dim=int(cfg.get("lift_dim", 16))
        )

    def _init_integral_control_mixer(self, config: Dict, d_in: int) -> None:
        from .integral_control_gate import IntegralControlMixer

        cfg = config or {}
        self.integral_control_block = IntegralControlMixer(
            d_in,
            anti_windup=bool(cfg.get("anti_windup", True)),
            s_max=float(cfg.get("s_max", 10.0)),
        )

    def _init_port_hamiltonian_mix(self, config: Dict, d_in: int) -> None:
        from .port_hamiltonian_mix import PortHamiltonianMixer

        cfg = config or {}
        # PortHamiltonianMixer requires 1 <= band < dim.
        band = max(1, min(int(cfg.get("band", 4)), d_in - 1))
        self.port_hamiltonian_block = PortHamiltonianMixer(
            d_in, band=band, tau=float(cfg.get("tau", 0.5))
        )

    def _init_scale_equivariant_wavelet(self, config: Dict, d_in: int) -> None:
        from .scale_equivariant_wavelet import ScaleEquivariantWaveletStack

        cfg = config or {}
        self.wavelet_block = ScaleEquivariantWaveletStack(
            d_in,
            kernel_size=max(2, int(cfg.get("kernel_size", 8))),
            n_scales=max(1, int(cfg.get("n_scales", 5))),
        )

    def _init_cdma_slot_binding(self, config: Dict, d_in: int) -> None:
        from .cdma_slot_binding import CDMASlotBinding

        cfg = config or {}
        # chips must divide dim (d_v = dim // chips >= 1).
        chips = self._nmf_largest_divisor_at_most(d_in, int(cfg.get("chips", 32)))
        self.cdma_binding_block = CDMASlotBinding(
            d_in,
            n_slots=int(cfg.get("n_slots", 8)),
            chips=chips,
            code_family=str(cfg.get("code_family", "gold")),
        )
