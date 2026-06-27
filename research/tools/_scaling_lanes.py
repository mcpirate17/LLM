"""Lane helper classes and builder registry for scaling_blimp_study.

Exports
-------
_SimplifiedMambaLane       -- selective-SSM lane class
AdjacentTokenMergeLane     -- adjacent-token-merge lane class
WINNER_LANE_FINGERPRINTS   -- nano BLiMP winner fingerprint dict
_build_lane_factory        -- main public entry point
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from component_fab.generator.block_templates import (
    GatedParallelBlock,
    HeteroMoEBlock,
    RecursiveDepthRouterBlock,
    ThreeLaneAdaptive,
)
from component_fab.generator.primitive_templates import (
    AnisotropicSemiringReciprocalAttention,
    FixedRankReciprocalAttention,
    HeteroSemiringReciprocalAttention,
    LinearStateSpaceLane,
    MultiscaleWaveletLane,
    PhaseLockAttention,
    PoincareAttention,
    ReciprocalPrimaryRefine,
    ReciprocalRankAttention,
    SemiringReciprocalAttention,
    SparseReciprocalAttention,
    SparsemaxAttention,
    TemperedTropicalAttention,
    TropicalAttention,
)
from component_fab.harness.tiny_lm import lane_factory_for_baseline
from research.tools._scaling_lanes_parametric import (
    ALL_SURPRISE_NAMES as _ALL_SURPRISE_NAMES,
    REGEX_BUILDERS as _REGEX_BUILDERS,
    build_native_surprise as _build_native_surprise,
    build_non_native_surprise as _build_non_native_surprise,
)


# ---------------------------------------------------------------------------
# Lane classes
# ---------------------------------------------------------------------------


class _SimplifiedMambaLane(nn.Module):
    """Selective state-space lane (simplified Mamba/S6).

    Each token computes its OWN dt (time step), A (state-transition), B (input
    projection) — i.e., selectivity is content-dependent. Unlike
    ``LinearStateSpaceLane`` which has fixed-per-channel transition matrices,
    this lane lets each token gate state updates based on its own embedding.
    Critically: the dt-gating is the load-bearing part of why Mamba beats
    linear SSMs on BLiMP-like tasks.

    Implementation note: this is a simplified scalar-state version (not the
    full Mamba complex state). Sufficient for binding/induction comparisons
    but not a faithful Mamba reproduction.
    """

    def __init__(self, dim: int, state_dim: int | None = None) -> None:
        super().__init__()
        state_dim = state_dim or dim
        self.dim = dim
        self.state_dim = state_dim
        # Content-dependent dt projection — the selectivity mechanism.
        self.dt_proj = nn.Linear(dim, state_dim)
        # Content-dependent B (input → state) projection.
        self.B_proj = nn.Linear(dim, state_dim)
        # Static A (per-state-dim decay rate, like Mamba's per-channel a).
        self.a_log = nn.Parameter(torch.zeros(state_dim))
        # Output projection.
        self.C_proj = nn.Linear(dim, state_dim)
        self.out_proj = nn.Linear(state_dim, dim, bias=False)
        # Gate (Mamba's SiLU gate over the SSM output).
        self.gate_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Per-token dt (positive via softplus).
        dt = torch.nn.functional.softplus(self.dt_proj(x))  # [B, L, state]
        # Per-token B (input gate).
        B = self.B_proj(x)  # [B, L, state]
        # Per-token C (output projection).
        C = self.C_proj(x)  # [B, L, state]
        # Discrete-time recurrence with content-dependent transitions.
        a = -torch.exp(self.a_log)  # negative for stability (real Mamba uses neg eig)
        # Parallel associative (Kogge-Stone) scan over h[t] = exp(dt*a) * h[t-1] + dt*B.
        # log-domain: log_a_bar[t] = dt[t] * a; b[t] = dt[t] * B[t].
        # Existing scan expects [..., L] with last dim = sequence, so transpose state/L axes.
        from research.synthesis.compiler_ops_sequence import _parallel_associative_scan

        log_a_bar = (dt * a).transpose(-1, -2).contiguous()  # [B, state, L]
        b_bar = (dt * B).transpose(-1, -2).contiguous()  # [B, state, L]
        h_t = _parallel_associative_scan(log_a_bar, b_bar)  # [B, state, L]
        h = h_t.transpose(-1, -2)  # [B, L, state]
        y_scalar = (C * h).sum(dim=-1, keepdim=True)  # [B, L, 1]
        y_seq = y_scalar.expand(-1, -1, self.dim)  # [B, L, dim]
        ssm_out = self.out_proj(y_seq)
        gate = torch.nn.functional.silu(self.gate_proj(x))
        return ssm_out * gate


class AdjacentTokenMergeLane(nn.Module):
    """Binding-specialist lane built on the ``adjacent_token_merge`` primitive.

    The primitive (``compiler_ops_routing._op_adjacent_token_merge``) is a
    parameter-free, strictly-causal token compressor: it merges even-stride
    tokens into their predecessor and then restores the original seq_len via a
    nearest-kept mapping (``(B,S,D) -> (B,S,D)``). On its own it carries no
    learnable capacity, so we surround it with learnable in/out projections and
    a sigmoid content gate to make it a trainable mixer.

    As a bare mixer it relies on the outer ``_LaneBlock`` for the
    norm/residual/FFN structure (the merge is information-destructive and
    ``REQUIRES_RESIDUAL_BYPASS`` — the outer ``x + lane(norm1(x))`` supplies it).

    The merge op has data-dependent control flow (``searchsorted``/dynamic
    ``arange``) and mutates a ``routing_telemetry`` dict on ``self`` every call,
    both of which poison the dynamo recompile cache, so it is isolated via
    ``@torch._dynamo.disable`` (see [[feedback_dynamo_disable_synthesis_forwards]]).
    It is also forced to fp32 to avoid the bf16 ``scatter_add`` dtype mismatch
    under AMP.
    """

    def __init__(self, dim: int, keep_frac: float = 0.5) -> None:
        super().__init__()
        self.dim = dim
        self.keep_frac = float(keep_frac)
        self.in_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    @torch._dynamo.disable
    def _merge(self, h: torch.Tensor) -> torch.Tensor:
        from research.synthesis.compiler_ops_routing import _op_adjacent_token_merge

        seq_len = h.shape[1]
        n_keep = max(1, int(round(seq_len * self.keep_frac)))
        with torch.autocast(device_type=h.device.type, enabled=False):
            return _op_adjacent_token_merge(self, [h.float()], {"n_keep": n_keep})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        merged = self._merge(self.in_proj(x)).to(x.dtype)
        gate = torch.sigmoid(self.gate(x))
        return self.out_proj(merged) * gate


class NativeAdaptiveReciprocalSlotDeltaLane(nn.Module):
    """Native-adaptive trunk with reciprocal addressing and delta-slot sidecars.

    This native-format hybrid reuses the frontier components directly:
    native adaptive surprise memory supplies the broad-capability trunk,
    reciprocal attention supplies mutual-match read addressing, and the slot
    table supplies explicit AR/state-tracking memory with delta updates enabled.
    """

    GATE_FLOOR = 0.25

    def __init__(self, dim: int) -> None:
        super().__init__()
        from component_fab.generator.memory_primitives import (
            MultiHeadSlotTableMemoryLane,
        )
        from component_fab.generator.native_surprise_memory import (
            NativeAdaptiveSemiringBiLaneSurpriseMemoryLane,
        )

        slot_memory_dim = max(4, ((7 * dim) // 32) * 4)
        self.native = NativeAdaptiveSemiringBiLaneSurpriseMemoryLane(
            dim,
            memory_dim=32,
            gate_bias=0.0,
            semiring_temp_init=1.0,
            recursive_balance_init=1.0,
            low_threshold=0.0,
            high_threshold=0.02,
            max_recursive_steps=4,
        )
        self.reciprocal = ReciprocalRankAttention(dim, use_rope=True)
        self.slot = MultiHeadSlotTableMemoryLane(
            dim,
            memory_dim=slot_memory_dim,
            n_heads=max(4, dim // 64),
            n_slots=8,
            use_delta_update=True,
            route_from_input=True,
            normalize_slot_values=True,
            refine_write_route=True,
            consolidate_slots=True,
        )
        self.gate = nn.Linear(dim, 3)
        self.native_gate_floor = float(self.GATE_FLOOR)
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.bias.copy_(torch.tensor([2.0, -0.5, -0.5]))

    @staticmethod
    def _rms_normalize_branch(x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_weights = torch.softmax(self.gate(x), dim=-1)
        native_floor = float(getattr(self, "native_gate_floor", self.GATE_FLOOR))
        native_floor = min(max(native_floor, 0.0), 1.0)
        weights = raw_weights.clone()
        weights[..., 0] = native_floor + (1.0 - native_floor) * raw_weights[..., 0]
        weights[..., 1:] = (1.0 - native_floor) * raw_weights[..., 1:]
        if x.is_cuda:
            with torch.autocast(device_type=x.device.type, enabled=False):
                native = self.native(x.float()).to(x.dtype)
        else:
            native = self.native(x)
        reciprocal = self.reciprocal(x)
        slot = self.slot(x)
        native = self._rms_normalize_branch(native)
        reciprocal = self._rms_normalize_branch(reciprocal)
        slot = self._rms_normalize_branch(slot)
        weighted_native = weights[..., 0:1] * native
        weighted_reciprocal = weights[..., 1:2] * reciprocal
        weighted_slot = weights[..., 2:3] * slot
        with torch.no_grad():
            entropy = -(raw_weights * (raw_weights + 1e-8).log()).sum(dim=-1)
            self.last_gate_metrics = {
                "raw_gate_mean": raw_weights.float().mean(dim=(0, 1)).detach().cpu(),
                "effective_gate_mean": weights.float().mean(dim=(0, 1)).detach().cpu(),
                "gate_entropy": float(entropy.float().mean().item()),
                "weighted_branch_rms": torch.stack(
                    [
                        weighted_native.float().pow(2).mean().sqrt(),
                        weighted_reciprocal.float().pow(2).mean().sqrt(),
                        weighted_slot.float().pow(2).mean().sqrt(),
                    ]
                )
                .detach()
                .cpu(),
            }
        return weighted_native + weighted_reciprocal + weighted_slot


# 2026-05-28: nano-scale BLiMP winners (seq=512 sweet-spot study) — lane name →
# (graphs-table fingerprint, description). Single source of truth shared by the
# lane factory below and `tools/apply_mixer_fingerprint_pretrain.py`.
WINNER_LANE_FINGERPRINTS: dict[str, tuple[str, str]] = {
    "pq_rope_winner": (
        "9be78a43c07948c4",  # pragma: allowlist secret
        "pq_embedding_moe_block_rope (BLiMP 0.5543 @ 10k seq512)",
    ),
    "semiring_winner": (
        "a2f747a20982907a",  # pragma: allowlist secret
        "learnable_semiring_attention_block (BLiMP 0.5651 @ 10k seq512)",
    ),
}


# ---------------------------------------------------------------------------
# Per-lane builder functions
# ---------------------------------------------------------------------------
# Each builder takes (name, top_k_frac) and returns a Callable[[int], nn.Module].
# Regex-dispatched builders take a pre-matched re.Match in place of name.


def _build_softmax(name: str, top_k_frac: float) -> Callable[[int], nn.Module]:
    return lane_factory_for_baseline("softmax_attention")


def _build_tropical(name: str, top_k_frac: float) -> Callable[[int], nn.Module]:
    return TropicalAttention


def _build_sparsemax(name: str, top_k_frac: float) -> Callable[[int], nn.Module]:
    return SparsemaxAttention


def _build_simplified_mamba(name: str, top_k_frac: float) -> Callable[[int], nn.Module]:
    return _SimplifiedMambaLane


def _build_multiscale_wavelet(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    return MultiscaleWaveletLane


def _build_linear_ssm(name: str, top_k_frac: float) -> Callable[[int], nn.Module]:
    return LinearStateSpaceLane


def _build_causal_conv(name: str, top_k_frac: float) -> Callable[[int], nn.Module]:
    return lane_factory_for_baseline("causal_conv")


def _build_adjacent_token_merge(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    return AdjacentTokenMergeLane


def _build_gemini_master(name: str, top_k_frac: float) -> Callable[[int], nn.Module]:
    from component_fab.generator.memory_primitives import UniversalMasterLane

    return lambda d: UniversalMasterLane(d)


def _build_slot_table_mh(name: str, top_k_frac: float) -> Callable[[int], nn.Module]:
    # Locked content-binding lane (validated at nano: solves binding_validity 0.99@3200,
    # robust to 32 pairs / seq512, where softmax_4h is pinned ~0.22). Production config:
    # composer + normalized read + input-route + RMSNorm + null-write, joint router.
    # n_heads scales with dim (like attention); memory_dim via the dispatcher formula.
    from component_fab.generator.memory_primitives import MultiHeadSlotTableMemoryLane

    return lambda d: MultiHeadSlotTableMemoryLane(
        d,
        memory_dim=max(4, ((7 * d) // 32) * 4),
        n_heads=max(4, d // 64),
        n_slots=8,
        use_delta_update=False,
        route_from_input=True,
        normalize_slot_values=True,
    )


def _build_slot_table_mh_dplr(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    # Run-1 compositional upgrade of slot_table_mh: attacks the multislot wall
    # (all_slots=0.0, held_class=1.0 = right content/wrong slot). On top of the
    # delta write: content-aware per-slot eviction decoupled from the route
    # (last-write-wins instead of blending), a DPLR low-rank value correction, and
    # LEARNT slot identity/capacity (prototypes + per-slot bias + route temp).
    # Isolated single-lane build so the nano/40M grade attributes any all_slots
    # movement to the slot mechanism alone.
    from component_fab.generator.memory_primitives import MultiHeadSlotTableMemoryLane

    return lambda d: MultiHeadSlotTableMemoryLane(
        d,
        memory_dim=max(4, ((7 * d) // 32) * 4),
        n_heads=max(4, d // 64),
        n_slots=8,
        use_delta_update=True,
        route_from_input=True,
        normalize_slot_values=True,
        refine_write_route=True,
        consolidate_slots=True,
        content_forget=True,
        dplr_value_rank=16,
        learnable_slots=True,
    )


def _build_native_adaptive_reciprocal_slot_delta(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    return NativeAdaptiveReciprocalSlotDeltaLane


def _build_reciprocal_rank(name: str, top_k_frac: float) -> Callable[[int], nn.Module]:
    return lambda d: ReciprocalRankAttention(d, use_rope=True)


def _build_hyperbolic(name: str, top_k_frac: float) -> Callable[[int], nn.Module]:
    # Non-Euclidean Poincare-ball addressing geometry. Keep the public lane
    # name stable so existing scale reports continue to identify this family.
    return PoincareAttention


def _build_phase_lock(name: str, top_k_frac: float) -> Callable[[int], nn.Module]:
    return lambda d: PhaseLockAttention(d, use_rope=True)


def _build_reciprocal_phase_two_lane(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    # Both induction-strong novel lanes only (no tropical/sparsemax), to test
    # whether a 2-lane preserves the standout nano_induction_nearest that the
    # 3-lane dilutes (0.44 single -> 0.29 in reciprocal_phase_tropical).
    def factory(dim: int) -> nn.Module:
        return GatedParallelBlock(
            lambda d: ReciprocalRankAttention(d, use_rope=True),
            lambda d: PhaseLockAttention(d, use_rope=True),
            dim,
        )

    return factory


def _build_sparse_reciprocal(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    return lambda d: SparseReciprocalAttention(d, use_rope=True)


def _build_semiring_reciprocal(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    return lambda d: SemiringReciprocalAttention(d, use_rope=True)


def _build_hetero_semiring_reciprocal(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    # Heterogeneous-algebra multi-head: per-head reciprocity β_h + signed per-head
    # semiring γ_h (soft-min/mean/soft-max), head split + output proj.
    return lambda d: HeteroSemiringReciprocalAttention(d, use_rope=True)


def _build_anisotropic_semiring_reciprocal(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    # Anisotropic per-channel semiring: keeps the full-width single head and makes
    # γ a per-channel vector γ_d so each value feature pools under its own algebra.
    return lambda d: AnisotropicSemiringReciprocalAttention(d, use_rope=True)


def _build_fixed_rank_reciprocal(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    # Fixed-rank reciprocity: ONE attention pattern with the q·k score computed in
    # a width-invariant rank-96 subspace, mixing full-width values.
    return lambda d: FixedRankReciprocalAttention(d, rank=96, use_rope=True)


def _build_tempered_tropical(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    # Track B: learnable per-head Boltzmann temperature interpolating hard max-plus
    # ↔ soft log-mean-exp pooling.
    return lambda d: TemperedTropicalAttention(d, use_rope=True)


def _build_reciprocal_primary_phase_refine(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    return lambda d: ReciprocalPrimaryRefine(d, side="phase", use_rope=True)


def _build_reciprocal_primary_tropical_refine(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    return lambda d: ReciprocalPrimaryRefine(d, side="tropical", use_rope=True)


def _build_reciprocal_phase_tropical_three_lane(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    def factory(dim: int) -> nn.Module:
        return ThreeLaneAdaptive(
            lambda d: ReciprocalRankAttention(d, use_rope=True),
            lambda d: PhaseLockAttention(d, use_rope=True),
            lambda d: TropicalAttention(d),
            dim,
        )

    return factory


def _build_reciprocal_sparsemax_wavelet_three_lane(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    def factory(dim: int) -> nn.Module:
        return ThreeLaneAdaptive(
            lambda d: ReciprocalRankAttention(d, use_rope=True),
            lambda d: SparsemaxAttention(d),
            lambda d: MultiscaleWaveletLane(d),
            dim,
        )

    return factory


def _build_tropical_sparsemax_wavelet_three_lane(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    def factory(dim: int) -> nn.Module:
        return ThreeLaneAdaptive(
            lambda d: TropicalAttention(d),
            lambda d: SparsemaxAttention(d),
            lambda d: MultiscaleWaveletLane(d),
            dim,
        )

    return factory


def _build_tropical_sparsemax_two_lane(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    # 2026-05-19: 3-lane sublane ablation showed wavelet alone broken
    # on every structural probe; tropical+sparsemax recover the full
    # hybrid capability. This composite drops wavelet to test whether
    # the 2-lane is simpler-and-better.
    def factory(dim: int) -> nn.Module:
        def lane_a(d: int) -> nn.Module:
            return TropicalAttention(d)

        def lane_b(d: int) -> nn.Module:
            return SparsemaxAttention(d)

        return GatedParallelBlock(lane_a, lane_b, dim)

    return factory


def _build_top_ar_block(name: str, top_k_frac: float) -> Callable[[int], nn.Module]:
    # 2026-05-19: faithful reproduction of fp 7fb0412ec57a1213 (the
    # leaderboard-best AR-curriculum scorer at 0.9046, ~13M params, 1000
    # wikitext steps). Dual mixer with conv1d_seq + swiglu between the
    # two attentions, 3-way residual to original input.
    from component_fab.harness.top_ar_block import LocalWindowAttention, TopArchBlock

    def factory(dim: int) -> nn.Module:
        def mixer_a(d: int) -> nn.Module:
            return TropicalAttention(d)

        def mixer_b(d: int) -> nn.Module:
            return LocalWindowAttention(d, window_size=16)

        return TopArchBlock(dim, mixer_a, mixer_b)

    return factory


def _build_top_ar_block_with_two_lane(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    # 2026-05-19: same scaffold as top_ar_block but mixer_a (the tropical_attention
    # slot) is replaced with our 2-lane composite (GatedParallelBlock of tropical +
    # sparsemax). Tests whether the 2-lane is a productive substitution inside the
    # AR-friendly scaffold.
    from component_fab.harness.top_ar_block import LocalWindowAttention, TopArchBlock

    def factory(dim: int) -> nn.Module:
        def two_lane(d: int) -> nn.Module:
            def la(dd: int) -> nn.Module:
                return TropicalAttention(dd)

            def lb(dd: int) -> nn.Module:
                return SparsemaxAttention(dd)

            return GatedParallelBlock(la, lb, d)

        def mixer_b(d: int) -> nn.Module:
            return LocalWindowAttention(d, window_size=16)

        return TopArchBlock(dim, two_lane, mixer_b)

    return factory


def _build_block_gated_parallel(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    def factory(dim: int) -> nn.Module:
        def anchor(d: int) -> nn.Module:
            return TropicalAttention(d)

        def wavelet(d: int) -> nn.Module:
            return MultiscaleWaveletLane(d)

        return GatedParallelBlock(anchor, wavelet, dim)

    return factory


def _build_recursive_depth_router(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    def factory(dim: int) -> nn.Module:
        def anchor(d: int) -> nn.Module:
            return TropicalAttention(d)

        return RecursiveDepthRouterBlock(anchor, dim, max_depth=4)

    return factory


def _build_hetero_moe_block(name: str, top_k_frac: float) -> Callable[[int], nn.Module]:
    def factory(dim: int) -> nn.Module:
        def anchor(d: int) -> nn.Module:
            return TropicalAttention(d)

        return HeteroMoEBlock(
            anchor,
            (LinearStateSpaceLane, MultiscaleWaveletLane, TropicalAttention),
            dim,
        )

    return factory


def _build_ensemble_top_ar(name: str, top_k_frac: float) -> Callable[[int], nn.Module]:
    # 2026-05-19: parallel-sum ensemble lanes built from the top AR-curriculum
    # graphs in runs.db. Lazy-import the factory builder to avoid a
    # module-import-time dependency on sqlite/synthesis paths. Names:
    #   ensemble_top_ar_Nway (N in 1..4) — equal-weight parallel-sum of top-N
    #   ensemble_top_ar_plus_three_lane — top-4 graphs sum + ThreeLaneAsBlock
    # All ensemble lanes have internal norm/FFN/residual via the underlying
    # graphs — pair with mixer_fingerprint via `--no-ffn` if the TinyLM
    # outer FFN must be skipped.
    from research.tools.ensemble_screening import (
        _load_top_graphs,
        _make_ensemble_lane_factory,
        _make_ensemble_plus_three_lane_factory,
    )

    if name == "ensemble_top_ar_plus_three_lane":
        specs = _load_top_graphs(4)
        return _make_ensemble_plus_three_lane_factory(specs)
    suffix = name[len("ensemble_top_ar_") :]
    if not suffix.endswith("way"):
        raise ValueError(f"expected '_Nway' suffix in {name!r}")
    n = int(suffix[: -len("way")])
    specs = _load_top_graphs(n)
    return _make_ensemble_lane_factory(specs)


def _build_local_ssm_diff(name: str, top_k_frac: float) -> Callable[[int], nn.Module]:
    # 2026-05-21 Phase-1 cross-bias mining: the only rows with both
    # binding_intermediate > 0.7, ar_curriculum > 0.4, and induction=1.0
    # were local_attn_ssm_hybrid variants with local_window_attn +
    # conv1d_seq + selective_scan + diff_attention.
    from research.tools.ensemble_screening import (
        _load_graphs_by_fingerprint,
        _make_ensemble_lane_factory,
    )

    specs = _load_graphs_by_fingerprint(
        (
            (
                "bb0b8d5856da1f29",  # pragma: allowlist secret
                "local_window + conv + selective_scan + diff_attention",
                0.7975,
            ),
            (
                "5c5013c79d1f0a51",  # pragma: allowlist secret
                "local_window + conv + selective_scan + diff_attention alt",
                0.4069,
            ),
        )
    )
    return _make_ensemble_lane_factory(specs)


def _build_routed_compress(name: str, top_k_frac: float) -> Callable[[int], nn.Module]:
    # 2026-05-21 Phase-1 cross-bias mining: cluster 2 — graphs that clear
    # binding_intermediate > 0.7 AND induction_intermediate > 0.5 AND
    # ar_curriculum > 0.3, sharing the `latent_compress + difficulty_routed
    # + routed_bottleneck` templates.
    from research.tools.ensemble_screening import (
        ROUTED_COMPRESS_FPS,
        _load_graphs_by_fingerprint,
        _make_ensemble_lane_factory,
    )

    return _make_ensemble_lane_factory(_load_graphs_by_fingerprint(ROUTED_COMPRESS_FPS))


def _build_local_ssm_diff_rope(
    name: str, top_k_frac: float
) -> Callable[[int], nn.Module]:
    # 2026-05-21: controlled ablation of `local_ssm_diff` with `rope_rotate`
    # nodes injected before every `local_window_attn` / `diff_attention`
    # node in the cluster-1 graphs.
    from research.tools.ensemble_screening import (
        CROSS_BIAS_FPS,
        _inject_rope_before_ops,
        _load_graphs_by_fingerprint,
        _make_ensemble_lane_factory,
    )

    specs = _load_graphs_by_fingerprint(CROSS_BIAS_FPS)
    roped = [
        (fp, desc, db_auc, _inject_rope_before_ops(g)) for fp, desc, db_auc, g in specs
    ]
    return _make_ensemble_lane_factory(roped)


def _build_winner_lane(name: str, top_k_frac: float) -> Callable[[int], nn.Module]:
    # 2026-05-28: nano-scale BLiMP winners (seq=512 sweet-spot study). Each
    # is a single synthesized block reconstructed from its graph_json in the
    # `graphs` dedup table.  Wrapped as a 1-branch ensemble = the block
    # itself (its own norm/residual built in → pair with mixer_fingerprint
    # `--no-ffn`). Powers the from-scratch Chinchilla pretrain.
    from research.tools.ensemble_screening import (
        _load_graphs_from_graphs_table,
        _make_ensemble_lane_factory,
    )

    fp, desc = WINNER_LANE_FINGERPRINTS[name]
    specs = _load_graphs_from_graphs_table(((fp, desc, 0.0),))
    return _make_ensemble_lane_factory(specs)


# ---------------------------------------------------------------------------
# Exact-match dispatch table
# ---------------------------------------------------------------------------
# _ALL_SURPRISE_NAMES, _build_native_surprise, _build_non_native_surprise,
# and _REGEX_BUILDERS are imported from _scaling_lanes_parametric above.

LANE_BUILDERS: dict[str, Callable[..., Callable[[int], nn.Module]]] = {
    "softmax_ffn": _build_softmax,
    "softmax_attention": _build_softmax,
    "tropical_attention": _build_tropical,
    "sparsemax_attention": _build_sparsemax,
    "simplified_mamba": _build_simplified_mamba,
    "multiscale_wavelet": _build_multiscale_wavelet,
    "linear_ssm": _build_linear_ssm,
    "causal_conv": _build_causal_conv,
    "adjacent_token_merge_lane": _build_adjacent_token_merge,
    "gemini_master": _build_gemini_master,
    "slot_table_mh": _build_slot_table_mh,
    "slot_table_mh_dplr": _build_slot_table_mh_dplr,
    "native_adaptive_reciprocal_slot_delta": _build_native_adaptive_reciprocal_slot_delta,
    "reciprocal_rank_attention": _build_reciprocal_rank,
    "hyperbolic_attention": _build_hyperbolic,
    "phase_lock_attention": _build_phase_lock,
    "reciprocal_phase_two_lane": _build_reciprocal_phase_two_lane,
    "sparse_reciprocal_attention": _build_sparse_reciprocal,
    "semiring_reciprocal_attention": _build_semiring_reciprocal,
    "hetero_semiring_reciprocal": _build_hetero_semiring_reciprocal,
    "anisotropic_semiring_reciprocal": _build_anisotropic_semiring_reciprocal,
    "fixed_rank_reciprocal": _build_fixed_rank_reciprocal,
    "tempered_tropical": _build_tempered_tropical,
    "reciprocal_primary_phase_refine": _build_reciprocal_primary_phase_refine,
    "reciprocal_primary_tropical_refine": _build_reciprocal_primary_tropical_refine,
    "reciprocal_phase_tropical_three_lane": _build_reciprocal_phase_tropical_three_lane,
    "reciprocal_sparsemax_wavelet_three_lane": _build_reciprocal_sparsemax_wavelet_three_lane,
    "tropical_sparsemax_wavelet_three_lane": _build_tropical_sparsemax_wavelet_three_lane,
    "tropical_sparsemax_two_lane": _build_tropical_sparsemax_two_lane,
    "top_ar_block": _build_top_ar_block,
    "top_ar_block_with_two_lane": _build_top_ar_block_with_two_lane,
    "block_gated_parallel": _build_block_gated_parallel,
    "recursive_depth_router": _build_recursive_depth_router,
    "hetero_moe_block": _build_hetero_moe_block,
    "local_ssm_diff": _build_local_ssm_diff,
    "routed_compress": _build_routed_compress,
    "local_ssm_diff_rope": _build_local_ssm_diff_rope,
    "pq_rope_winner": _build_winner_lane,
    "semiring_winner": _build_winner_lane,
    **{
        n: _build_native_surprise
        for n in _ALL_SURPRISE_NAMES
        if n.startswith("native_")
    },
    **{
        n: _build_non_native_surprise
        for n in _ALL_SURPRISE_NAMES
        if not n.startswith("native_")
    },
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _build_lane_factory(
    name: str, top_k_frac: float = 0.25
) -> Callable[[int], nn.Module]:
    """Resolve a lane-name to a lane factory for scaling tests."""
    # Exact-match lookup first (O(1)).
    builder = LANE_BUILDERS.get(name)
    if builder is not None:
        return builder(name, top_k_frac)

    # ensemble_top_ar_* prefix (also handles ensemble_top_ar_plus_three_lane).
    if name.startswith("ensemble_top_ar_"):
        return _build_ensemble_top_ar(name, top_k_frac)

    # Regex-dispatched parametric lanes.
    for pattern, regex_builder in _REGEX_BUILDERS:
        m = pattern.fullmatch(name)
        if m is not None:
            return regex_builder(m, top_k_frac)

    raise ValueError(f"unknown lane name: {name!r}")
