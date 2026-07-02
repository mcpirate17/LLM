"""Param-init mixin for the NM-C compaction mixers (Tier D program).

Each ``_init_*`` instantiates the full self-contained mixer class as a
submodule (the ``pq_embedding_moe_block`` convention) so search/synthesis and
final model generation share exact semantics. Knob defaults that carry a
divisibility/capacity constraint are adapted to ``d_in`` the same fail-soft way
``_init_pq_embedding`` shrinks ``M`` — the op must initialise at any model dim
the search proposes. Forward handlers live in ``compiler_ops_compaction.py``.

Split out of ``compiled_op_params.py`` (god-file guard); consumed by
``CompiledOpParamInitMixin`` via inheritance.
"""

from __future__ import annotations

from typing import Dict


class CompactionOpParamInit:
    @staticmethod
    def _largest_divisor_at_most(d_in: int, cap: int) -> int:
        for candidate in range(min(cap, d_in), 0, -1):
            if d_in % candidate == 0:
                return candidate
        return 1

    def _init_monarch_mix(self, config: Dict, d_in: int) -> None:
        from .monarch_mix import MonarchMix

        cfg = config or {}
        block_size = cfg.get("block_size")
        self.monarch_block = MonarchMix(
            d_in, block_size=int(block_size) if block_size else None
        )

    def _init_butterfly_mix(self, config: Dict, d_in: int) -> None:
        from .butterfly_mix import ButterflyMix

        cfg = config or {}
        self.butterfly_block = ButterflyMix(d_in, n_passes=int(cfg.get("n_passes", 2)))

    def _init_recurrent_depth_refine(self, config: Dict, d_in: int) -> None:
        from .recurrent_depth_refine import RecurrentDepthRefine

        cfg = config or {}
        self.recurrent_depth_block = RecurrentDepthRefine(
            d_in, max_depth=int(cfg.get("max_depth", 3)), p=int(cfg.get("p", 2))
        )

    def _init_weight_dictionary_mix(self, config: Dict, d_in: int) -> None:
        from .weight_dictionary_mix import WeightDictionaryMix

        cfg = config or {}
        self.weight_dictionary_block = WeightDictionaryMix(
            d_in,
            n_layers=int(cfg.get("n_layers", 4)),
            n_basis=int(cfg.get("n_basis", 2)),
        )

    def _init_hypernet_layer_mix(self, config: Dict, d_in: int) -> None:
        from .hypernet_layer_mix import HyperLayerMix

        cfg = config or {}
        n_chunks = self._largest_divisor_at_most(d_in, int(cfg.get("n_chunks", 8)))
        self.hypernet_block = HyperLayerMix(
            d_in,
            n_layers=int(cfg.get("n_layers", 4)),
            n_roles=int(cfg.get("n_roles", 2)),
            n_chunks=n_chunks,
        )

    def _init_persistent_memory_refine(self, config: Dict, d_in: int) -> None:
        from .persistent_memory_refine import PersistentMemoryRefine

        cfg = config or {}
        n_slots = int(cfg.get("n_slots", 16))
        self.persistent_memory_block = PersistentMemoryRefine(
            d_in,
            n_slots=n_slots,
            top_k=min(int(cfg.get("top_k", 4)), n_slots),
            p=int(cfg.get("p", 2)),
        )

    def _init_block_sparse_mix(self, config: Dict, d_in: int) -> None:
        from .block_sparse_mix import BlockSparseMix

        cfg = config or {}
        block_size = self._largest_divisor_at_most(d_in, int(cfg.get("block_size", 8)))
        self.block_sparse_block = BlockSparseMix(
            d_in, block_size=block_size, n_blocks=int(cfg.get("n_blocks", 8))
        )

    def _init_token_merge_mix(self, config: Dict, d_in: int) -> None:
        from .token_merge_mix import TokenMergeMix

        cfg = config or {}
        # The sheaf overlap must be a PROPER restriction (overlap_dim < d_in).
        overlap_dim = min(int(cfg.get("overlap_dim", 8)), max(1, d_in // 4))
        self.token_merge_block = TokenMergeMix(
            d_in,
            overlap_dim=max(1, min(overlap_dim, d_in - 1)),
            max_cluster=int(cfg.get("max_cluster", 8)),
        )

    def _init_ternary_sign_mix(self, config: Dict, d_in: int) -> None:
        from .ternary_sign_mix import TernarySignMix

        cfg = config or {}
        rank = cfg.get("rank")
        self.ternary_sign_block = TernarySignMix(d_in, rank=int(rank) if rank else None)

    def _init_padic_lowprec_mix(self, config: Dict, d_in: int) -> None:
        from .padic_lowprec_mix import PadicLowPrecMixer

        cfg = config or {}
        self.padic_lowprec_block = PadicLowPrecMixer(
            d_in, n_digits=int(cfg.get("n_digits", 4)), p=int(cfg.get("p", 2))
        )

    def _init_lowrank_state_memory(self, config: Dict, d_in: int) -> None:
        from .lowrank_state_memory import LowRankStateMemory

        cfg = config or {}
        # rank must be < dim (the low-rank state claim).
        rank = max(1, min(int(cfg.get("rank", 8)), d_in - 1))
        self.lowrank_state_block = LowRankStateMemory(d_in, rank=rank)

    def _init_subspace_mixture_mix(self, config: Dict, d_in: int) -> None:
        from .subspace_mixture_mix import SubspaceMixtureMix

        cfg = config or {}
        # Total subspace width m·s must stay < d_in (the compaction claim).
        subspace_dim = max(1, min(int(cfg.get("subspace_dim", d_in // 8)), d_in - 1))
        n_subspaces = int(cfg.get("n_subspaces", 4))
        while n_subspaces > 1 and n_subspaces * subspace_dim >= d_in:
            n_subspaces -= 1
        while subspace_dim > 1 and n_subspaces * subspace_dim >= d_in:
            subspace_dim -= 1
        self.subspace_mixture_block = SubspaceMixtureMix(
            d_in, n_subspaces=n_subspaces, subspace_dim=subspace_dim
        )
