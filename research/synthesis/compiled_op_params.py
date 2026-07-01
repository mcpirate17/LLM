from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .graph import ShapeInfo
from .primitives import PrimitiveOp


class CompiledOpParamInitMixin:
    def _make_param(self, shape: Tuple[int, ...], std: float = 0.02) -> nn.Parameter:
        return nn.Parameter(
            torch.empty(shape, dtype=torch.float32).normal_(mean=0.0, std=std)
        )

    def _init_params(self, op: PrimitiveOp, config: Dict, input_shape: ShapeInfo):
        # guardrail: allow-god-function
        d_in = max(1, input_shape.dim)
        linear_ops = {
            "linear_proj",
            "linear_proj_down",
            "linear_proj_up",
            "fused_linear_gelu",
        }
        if d_in < 4 and op.name not in linear_ops:
            d_in = self.model_dim
        d_out = max(1, config.get("out_dim", d_in))
        std = 1.0 / math.sqrt(d_in) if d_in > 0 else 0.02
        # Optional residual-aware init dampening: templates that emit a
        # `linear_proj` whose output joins a residual stream after N merged
        # paths can pass `init_scale=1/sqrt(N+1)` to shrink the projection's
        # initial variance contribution. Bounds Jacobian spectral norm at
        # the initial step so the investigation eligibility gate doesn't
        # reject before training has a chance to stabilize.
        proj_std = 0.02 * float(config.get("init_scale", 1.0))

        dispatch = {
            "linear_proj": lambda: setattr(
                self, "weight", self._make_param((d_out, d_in), std=proj_std)
            ),
            "linear_proj_down": lambda: setattr(
                self, "weight", self._make_param((d_out, d_in), std=proj_std)
            ),
            "linear_proj_up": lambda: setattr(
                self, "weight", self._make_param((d_out, d_in), std=proj_std)
            ),
            "fused_linear_gelu": lambda: (
                setattr(self, "weight", self._make_param((d_out, d_in), std=proj_std)),
                setattr(self, "bias", nn.Parameter(torch.zeros(d_out))),
            ),
            "learnable_scale": lambda: setattr(
                self, "scale", nn.Parameter(torch.ones(d_in))
            ),
            "learnable_bias": lambda: setattr(
                self, "bias", nn.Parameter(torch.zeros(d_in))
            ),
            "calibrated_branch_merge": lambda: self._init_calibrated_branch_merge(
                config, d_in
            ),
            "selective_scan": lambda: (
                setattr(self, "A_log", self._make_param((d_in,), std=0.1)),
                setattr(self, "dt_proj", self._make_param((d_in,), std=0.1)),
                setattr(self, "B_proj", nn.Linear(d_in, d_in, bias=False)),
                setattr(self, "C_proj", nn.Linear(d_in, d_in, bias=False)),
                self.B_proj.weight.data.normal_(std=0.02),
                self.C_proj.weight.data.normal_(std=0.02),
            ),
            "conv1d_seq": lambda: setattr(
                self,
                "conv_weight",
                self._make_param((d_in, 1, 3), std=1.0 / math.sqrt(3)),
            ),
            "gated_lane_blend": lambda: self._init_gated_lane_blend(config, d_in),
            "depth_gated_transform": lambda: self._init_depth_gated_transform(
                config, d_in
            ),
            "route_lanes": lambda: self._init_gated_lane_blend(config, d_in),
            "route_recursion": lambda: self._init_depth_gated_transform(config, d_in),
            "topk_gate": lambda: setattr(
                self, "gate_proj", self._make_param((2, d_in), std=0.02)
            ),
            "moe_topk": lambda: self._init_moe_topk(config, d_in),
            "pq_embedding_moe_block": lambda: self._init_pq_embedding_moe_block(
                config, d_in
            ),
            "moe_2expert": lambda: (
                setattr(self, "gate_proj", self._make_param((2, d_in), std=0.02)),
                setattr(
                    self, "expert_0_weight", self._make_param((d_in, d_in), std=0.02)
                ),
                setattr(
                    self, "expert_1_weight", self._make_param((d_in, d_in), std=0.02)
                ),
            ),
            "nm_sparse_linear": lambda: (
                setattr(self, "weight", self._make_param((d_out, d_in), std=0.02)),
                setattr(self, "sparsity_n", int(config.get("n", 2))),
                setattr(self, "sparsity_m", int(config.get("m", 4))),
            ),
            "block_sparse_linear": lambda: (
                setattr(self, "weight", self._make_param((d_out, d_in), std=0.02)),
                setattr(self, "block_size", max(1, int(config.get("block_size", 16)))),
                setattr(
                    self,
                    "block_density",
                    float(max(0.25, min(1.0, config.get("block_density", 0.25)))),
                ),
            ),
            "rmsnorm": lambda: setattr(self, "weight", nn.Parameter(torch.ones(d_in))),
            "qk_norm": lambda: setattr(
                self, "qk_scale", nn.Parameter(torch.ones(d_in))
            ),
            "logit_softcap": lambda: setattr(
                self, "softcap_logit", nn.Parameter(torch.tensor(2.0))
            ),
            "layernorm": lambda: (
                setattr(self, "weight", nn.Parameter(torch.ones(d_in))),
                setattr(self, "bias", nn.Parameter(torch.zeros(d_in))),
            ),
            "gated_linear": lambda: (
                setattr(
                    self, "linear_weight", self._make_param((d_out, d_in), std=0.02)
                ),
                setattr(self, "gate_weight", self._make_param((d_out, d_in), std=0.02)),
                setattr(self, "linear_bias", nn.Parameter(torch.zeros(d_out))),
                setattr(self, "gate_bias", nn.Parameter(torch.zeros(d_out))),
            ),
            "rwkv_time_mixing": lambda: (
                setattr(self, "w_decay", nn.Parameter(torch.ones(d_in) * -0.5)),
                setattr(self, "u_bonus", nn.Parameter(torch.zeros(d_in))),
                setattr(self, "W_k", self._make_param((d_in, d_in), std=0.02)),
                setattr(self, "W_v", self._make_param((d_in, d_in), std=0.02)),
                setattr(self, "W_r", self._make_param((d_in, d_in), std=0.02)),
                setattr(self, "W_o", self._make_param((d_in, d_in), std=0.02)),
                setattr(self, "_rwkv_kernel_ready", True),
            ),
            "embedding_lookup": lambda: (
                setattr(
                    self,
                    "codebook",
                    nn.Parameter(
                        torch.randn(min(int(config.get("vocab_size", 64)), 256), d_in)
                        * 0.02
                    ),
                ),
                setattr(
                    self,
                    "codebook_proj",
                    nn.Parameter(torch.randn(d_in, d_in) * (d_in**-0.5)),
                ),
            ),
            "rope_rotate": lambda: None,
            "cosine_similarity": lambda: None,
            "gather_topk": lambda: None,
            "spectral_filter": lambda: setattr(
                self, "freq_mask", nn.Parameter(torch.ones(d_in // 2 + 1))
            ),
            "semi_structured_2_4_linear": lambda: (
                setattr(self, "weight", self._make_param((d_out, d_in), std=0.02)),
                setattr(
                    self, "sparse_kernel_ready", bool(d_in % 4 == 0 and d_out % 4 == 0)
                ),
            ),
            "basis_expansion": lambda: setattr(
                self, "weight", nn.Parameter(torch.randn(4, d_in) * 0.1)
            ),
            "integral_kernel": lambda: setattr(
                self, "weight", nn.Parameter(torch.randn(d_in, d_in) * 0.02)
            ),
            "fixed_point_iter": lambda: setattr(
                self, "weight", nn.Parameter(torch.randn(d_in + 1, d_in) * 0.02)
            ),
            "kronecker_linear": lambda: self._init_kronecker_linear(d_in),
            "low_rank_proj": lambda: self._init_low_rank_proj(d_in),
            "grouped_linear": lambda: self._init_grouped_linear(d_in),
            "bottleneck_proj": lambda: self._init_bottleneck_proj(d_in),
            "shared_basis_proj": lambda: self._init_shared_basis_proj(d_in),
            "tied_proj": lambda: self._init_tied_proj(d_in),
            "tree_mix": lambda: self._init_tree_mix(config, d_in),
            "mla_attention": lambda: self._init_mla_attention(config, d_in),
            "pq_embedding": lambda: self._init_pq_embedding(config, d_in),
            "mlstm_cell": lambda: self._init_mlstm_cell(config, d_in),
            "swiglu_mlp": lambda: self._init_swiglu_mlp(config, d_in),
            "rwkv_channel": lambda: self._init_rwkv_channel(config, d_in),
            "softmax_attention": lambda: self._init_attention_stack(
                "softmax_attention", d_in
            ),
            "sparsemax_attention": lambda: self._init_attention_stack(
                "sparsemax_attention", d_in
            ),
            "entmax_attention": lambda: self._init_attention_stack(
                "entmax_attention", d_in
            ),
            "learnable_semiring_attention": lambda: self._init_semiring_attention(d_in),
            "reciprocal_rank_attention": lambda: self._init_reciprocal_rank_attention(
                d_in
            ),
            "reciprocal_semiring_attention": (
                lambda: self._init_reciprocal_semiring_attention(d_in)
            ),
            "phase_lock_attention": lambda: self._init_phase_lock_attention(d_in),
            "linear_attention": lambda: self._init_attention_stack(
                "linear_attention", d_in
            ),
            "graph_attention": lambda: self._init_attention_stack(
                "graph_attention", d_in
            ),
            "diff_attention": lambda: self._init_diff_attention(d_in),
            "gated_delta": lambda: self._init_gated_delta(d_in),
            "dplr_gated_delta": lambda: self._init_dplr_gated_delta(d_in),
            "state_space": lambda: self._init_state_space(d_in),
            "conv_only": lambda: self._init_conv_only(d_in),
            "stdp_attention": lambda: setattr(
                self, "log_tau", nn.Parameter(torch.tensor(0.0))
            ),
            "depth_token_mask": lambda: (
                setattr(self, "router_weight", self._make_param((1, d_in), std=0.02)),
            ),
            "difficulty_blend_3way": lambda: self._init_difficulty_blend_3way(d_in),
            "score_depth_blend": lambda: self._init_score_depth_blend(config, d_in),
            "confidence_token_gate": lambda: setattr(
                self, "confidence_proj", self._make_param((1, d_in), std=0.02)
            ),
            "learned_token_gate": lambda: setattr(
                self, "cascade_proj", self._make_param((1, d_in), std=0.02)
            ),
            "cheap_verify_blend": lambda: self._init_cheap_verify_blend(d_in),
            "hybrid_token_gate": lambda: setattr(
                self, "hybrid_gate_proj", self._make_param((1, d_in), std=0.02)
            ),
            "sparse_span_builder": lambda: None,
            "hybrid_sparse_router": lambda: self._init_hybrid_sparse_router(
                config, d_in
            ),
            "lane_conditioned_block": lambda: self._init_lane_conditioned_block(d_in),
            "default_path": lambda: None,
            "depth_weighted_proj": lambda: self._init_depth_weighted_proj(config, d_in),
            "padic_depth_route": lambda: self._init_padic_depth_route(config, d_in),
            "padic_gated_mixer": lambda: self._init_padic_gated_mixer(d_in),
            "sinkhorn_ot_mix": lambda: self._init_sinkhorn_ot_mix(d_in),
            "ultrametric_tree_mix": lambda: self._init_ultrametric_tree_mix(d_in),
            "fno_spectral_mix": lambda: self._init_fno_spectral_mix(d_in),
            "token_class_proj": lambda: self._init_token_class_proj(config, d_in),
            "adaptive_rank_gate": lambda: self._init_adaptive_rank_gate(d_in, d_out),
            "dual_compression_blend": lambda: self._init_dual_compression_blend(
                d_in, d_out
            ),
            "relu_gated_moe": lambda: self._init_relu_gated_moe(config, d_in),
            "relu_gate_routing": lambda: self._init_relu_gated_moe(config, d_in),
            "ternary_projection": lambda: self._init_ternary_projection(
                config, d_in, d_out
            ),
            "latent_attention_compressor": lambda: (
                self._init_latent_attention_compressor(d_in)
            ),
            "signal_conditioned_compression": lambda: (
                self._init_signal_conditioned_compression(d_in)
            ),
            "routing_conditioned_compression": lambda: (
                self._init_signal_conditioned_compression(d_in)
            ),
            "chebyshev_spectral_mix": lambda: self._init_chebyshev_spectral_mix(
                config, d_in
            ),
            "sparse_bottleneck_moe": lambda: self._init_sparse_bottleneck_moe(
                config, d_in
            ),
            "hetero_moe": lambda: self._init_hetero_moe(d_in),
            "arch_router": lambda: self._init_arch_router(d_in),
            "compute_budget_router": lambda: self._init_compute_budget_router(d_in),
            "difficulty_routed_attention": lambda: (
                self._init_difficulty_routed_attention(d_in)
            ),
            "strided_attention": lambda: self._init_strided_attention(d_in),
            "gated_progressive_attention": lambda: (
                self._init_gated_progressive_attention(d_in)
            ),
            "gated_linear_attention": lambda: self._init_gated_linear_attention(d_in),
            "long_conv_hyena": lambda: self._init_long_conv_hyena(d_in),
            "associative_memory": lambda: self._init_associative_memory(d_in),
            "role_slot_attention": lambda: self._init_role_slot_attention(config, d_in),
            "mixture_of_recursions": lambda: self._init_mixture_of_recursions(d_in),
            "token_hodge_mixer": lambda: self._init_token_hodge_mixer(d_in),
            "wavelet_packet_mix": lambda: self._init_wavelet_packet_mix(d_in),
            "retention_mix": lambda: self._init_retention_mix(d_in),
            "product_key_memory": lambda: self._init_product_key_memory(config, d_in),
        }

        handler = dispatch.get(op.name)
        if handler is not None:
            handler()
            return
        if op.category.value == "math_space":
            self._init_math_space(op, config, d_in, d_out)
            return
        if hasattr(op, "init_params"):
            op.init_params(self, d_in)
            return
        self.weight = nn.Parameter(torch.randn(d_in, d_in) * std)

    def _init_attention_stack(self, op_name: str, d_in: int) -> None:
        n_heads = max(1, d_in // 64)
        head_dim = d_in // n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        if op_name in (
            "softmax_attention",
            "graph_attention",
            "sparsemax_attention",
            "entmax_attention",
        ):
            self.attn_scale = head_dim**-0.5
        self.q_proj = nn.Linear(d_in, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_in, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_in, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_in, bias=False)
        self.q_proj.weight.data.normal_(std=0.02)
        self.k_proj.weight.data.normal_(std=0.02)
        self.v_proj.weight.data.normal_(std=0.02)
        self.o_proj.weight.data.normal_(std=0.02)
        if op_name == "graph_attention":
            self.edge_proj = nn.Linear(d_in, d_in, bias=False)
            self.edge_proj.weight.data.normal_(std=0.02)

    def _init_semiring_attention(self, d_in: int) -> None:
        """Attention stack + a learnable per-head value-aggregation exponent β.

        β is spread across the sum↔max/min continuum (linspace) so heads diversify
        from init; |β|>0 keeps β's gradient alive (β=0 hits the softmax-limit
        fallback, which is β-independent). See ``mathspaces.semiring``.
        """
        self._init_attention_stack("learnable_semiring_attention", d_in)
        self.attn_scale = self.head_dim**-0.5
        self.semiring_beta = nn.Parameter(torch.linspace(-0.6, 0.6, self.n_heads))

    def _init_reciprocal_rank_attention(self, d_in: int) -> None:
        """Attention stack with a learned reciprocal-match boost."""
        self._init_attention_stack("reciprocal_rank_attention", d_in)
        self.attn_scale = self.head_dim**-0.5
        self.reciprocal_logit_scale = nn.Parameter(torch.tensor(0.0))

    def _init_reciprocal_semiring_attention(self, d_in: int) -> None:
        """Attention stack with BOTH the reciprocal-match boost (address) and a
        learnable per-head value-aggregation exponent β (pooling) — the
        composition of reciprocal_rank + learnable_semiring. Both init to their
        identity (boost=0, β spread across the sum↔max/min continuum)."""
        self._init_attention_stack("reciprocal_semiring_attention", d_in)
        self.attn_scale = self.head_dim**-0.5
        self.reciprocal_logit_scale = nn.Parameter(torch.tensor(0.0))
        self.semiring_beta = nn.Parameter(torch.linspace(-0.6, 0.6, self.n_heads))

    def _init_phase_lock_attention(self, d_in: int) -> None:
        """Attention stack with a learned phase-synchrony score term."""
        self._init_attention_stack("phase_lock_attention", d_in)
        self.attn_scale = self.head_dim**-0.5
        self.phase_lock_scale = nn.Parameter(torch.tensor(0.0))

    def _init_math_space(
        self, op: PrimitiveOp, config: Dict, d_in: int, d_out: int
    ) -> None:
        if op.has_params:
            self.weight = self._make_param((d_out, d_in), std=0.02)
        if op.name in ("padic_expand", "padic_residual"):
            self.weight = self._make_param((d_in, d_in * 2), std=0.02)
            self.residual_scale = nn.Parameter(torch.zeros(1))
        elif op.name == "rotor_transform":
            self.rotor = nn.Parameter(torch.randn(8) * 0.02)
        elif op.name == "poincare_add":
            self.bias = nn.Parameter(torch.zeros(d_in))
        elif op.name == "hyp_linear":
            self.weight = self._make_param((d_in, d_in), std=0.02)
        elif op.name == "projective_linear":
            self.weight = self._make_param(((d_in + 1) * (d_in + 1),), std=0.02)
        elif op.name == "tropical_router":
            n_exp = int(config.get("n_experts", 8))
            self.centroids = nn.Parameter(torch.randn(n_exp, d_in) * 0.02)

    def _init_moe_topk(self, config: Dict, d_in: int) -> None:
        n_experts = int(config.get("num_experts", 4))
        self.gate_weight = self._make_param((n_experts, d_in), std=0.02)
        hidden = int(d_in * float(config.get("mlp_ratio", 2.0)))
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_in, hidden, bias=False),
                    nn.GELU(),
                    nn.Linear(hidden, d_in, bias=False),
                )
                for _ in range(n_experts)
            ]
        )
        for expert in self.experts:
            expert[0].weight.data.normal_(mean=0.0, std=0.02)
            expert[2].weight.data.normal_(
                mean=0.0, std=1.0 / math.sqrt(hidden if hidden > 0 else 1)
            )

    def _init_pq_embedding_moe_block(self, config: Dict, d_in: int) -> None:
        """Factorized Semantic Bottleneck MoE: PQ denoised routing.

        Instantiates the full block from arch_builder to ensure exact parity
        between search/synthesis and final model generation.
        """
        from ..arch_builder import PQEmbeddingMoEBlock

        self.block = PQEmbeddingMoEBlock(
            dim=d_in,
            n_experts=int(config.get("num_experts", 4)),
            topk=int(config.get("top_k", 2)),
            mlp_ratio=float(config.get("mlp_ratio", 3.0)),
            M=int(config.get("M", 4)),
            K=int(config.get("K", 16)),
        )

    def _init_kronecker_linear(self, d_in: int) -> None:
        p = int(d_in**0.5)
        q = d_in // p
        if p * q != d_in:
            for candidate in range(p, 0, -1):
                if d_in % candidate == 0:
                    p = candidate
                    q = d_in // p
                    break
        self.kron_A = nn.Parameter(torch.randn(p, p) * (p**-0.5))
        self.kron_B = nn.Parameter(torch.randn(q, q) * (q**-0.5))

    def _init_low_rank_proj(self, d_in: int) -> None:
        rank = max(d_in // 4, 1)
        self.U = nn.Parameter(torch.randn(d_in, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(rank, d_in) * 0.02)

    def _init_grouped_linear(self, d_in: int) -> None:
        g = 4
        group_dim = max(d_in // g, 1)
        self.weight = nn.Parameter(torch.randn(g, group_dim, group_dim) * 0.02)
        self.n_groups = g

    def _init_bottleneck_proj(self, d_in: int) -> None:
        rank = max(d_in // 4, 1)
        self.down = nn.Parameter(torch.randn(rank, d_in) * 0.02)
        self.up = nn.Parameter(torch.randn(d_in, rank) * 0.02)

    def _init_shared_basis_proj(self, d_in: int) -> None:
        k = 8
        self.basis = nn.Parameter(torch.randn(k, d_in) * 0.02)
        self.mixing = nn.Parameter(torch.randn(d_in, k) * 0.02)

    def _init_tied_proj(self, d_in: int) -> None:
        rank = max(d_in // 4, 1)
        self.tied_weight = nn.Parameter(torch.randn(rank, d_in) * 0.02)

    def _init_tree_mix(self, config: Dict, d_in: int) -> None:
        """Atomic binary mixer node (research §2.1).

        Single learned (d_in,) gate. Initialized to zeros so sigmoid(0)=0.5
        means each tree_mix node starts as a symmetric average; the gate
        learns asymmetry from gradient.
        """
        self.gate = nn.Parameter(torch.zeros(d_in))

    def _init_mla_attention(self, config: Dict, d_in: int) -> None:
        """Multi-Head Latent Attention params (research §1.1).

        Three projections: shared down (d_in -> d_latent) and two distinct
        ups (d_latent -> d_in, one each for K and V). d_latent defaults to
        d_in // 8 (DeepSeek V2 setting, ~93% KV cache compression).
        """
        from ..mathspaces.mla import _DEFAULT_LATENT_DIV

        if config:
            d_latent = int(config.get("d_latent", max(d_in // _DEFAULT_LATENT_DIV, 1)))
        else:
            d_latent = max(d_in // _DEFAULT_LATENT_DIV, 1)
        # Standard 0.02-std init matches surrounding attention projections.
        self.W_down = nn.Parameter(torch.randn(d_in, d_latent) * 0.02)
        self.W_up_K = nn.Parameter(torch.randn(d_latent, d_in) * 0.02)
        self.W_up_V = nn.Parameter(torch.randn(d_latent, d_in) * 0.02)

    def _init_pq_embedding(self, config: Dict, d_in: int) -> None:
        """Product-Quantized Embedding params (research §2.3).

        Splits d_in into M subspaces of size sub_dim = d_in // M; each
        subspace has K learnable codebook centroids. Falls back the largest
        valid M if d_in isn't divisible by the requested M.
        """
        from ..mathspaces.pq_embedding import _DEFAULT_M, _DEFAULT_K

        if config:
            m = int(config.get("M", _DEFAULT_M))
            k = int(config.get("K", _DEFAULT_K))
        else:
            m = _DEFAULT_M
            k = _DEFAULT_K
        # Fail-soft: shrink M until it divides d_in evenly.
        while m > 1 and d_in % m != 0:
            m -= 1
        sub_dim = max(d_in // m, 1)
        # Std-0.02 init like surrounding compression ops.
        self.codebooks = nn.Parameter(torch.randn(m, k, sub_dim) * 0.02)
        # Temperature knob (also adjustable from config).
        self.pq_tau = float(config.get("tau", 1.0)) if config else 1.0

    def _init_mlstm_cell(self, config: Dict, d_in: int) -> None:
        """mLSTM (matrix-memory LSTM) parameters (research §1.5).

        Three projections (W_q, W_k, W_v) feed the per-token query/key/value;
        W_o is the per-feature output gate; w_i, w_f are scalar input/forget
        gate projections. All projections use 0.02-std init matching the
        surrounding attention/SSM ops; gate biases are initialised so
        forget≈1 and input≈0 — this makes the first few training steps look
        like a pure-passthrough state (zero state, slow leak), which is the
        stable starting point used in the xLSTM paper.
        """
        _ = config  # mLSTM cell has no tuned hyperparams at this granularity.
        self.W_q = nn.Parameter(torch.randn(d_in, d_in) * 0.02)
        self.W_k = nn.Parameter(torch.randn(d_in, d_in) * 0.02)
        self.W_v = nn.Parameter(torch.randn(d_in, d_in) * 0.02)
        self.W_o = nn.Parameter(torch.randn(d_in, d_in) * 0.02)
        self.w_i = nn.Parameter(torch.randn(d_in) * 0.02)
        self.w_f = nn.Parameter(torch.randn(d_in) * 0.02)
        # Bias init: forget≈sigmoid(2)≈0.88, input≈sigmoid(-2)≈0.12, output≈sigmoid(0)=0.5.
        self.b_i = nn.Parameter(torch.full((), -2.0))
        self.b_f = nn.Parameter(torch.full((), 2.0))
        self.b_o = nn.Parameter(torch.zeros(d_in))

    def _init_swiglu_mlp(self, config: Dict, d_in: int) -> None:
        hidden = int(d_in * float(config.get("mlp_ratio", 3.0)))
        self.gate_proj = nn.Linear(d_in, hidden, bias=False)
        self.up_proj = nn.Linear(d_in, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_in, bias=False)
        self.gate_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.up_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.down_proj.weight.data.normal_(
            mean=0.0, std=1.0 / math.sqrt(hidden if hidden > 0 else 1)
        )

    def _init_rwkv_channel(self, config: Dict, d_in: int) -> None:
        hidden = int(d_in * float(config.get("mlp_ratio", 3.0)))
        self.mix_k = nn.Parameter(torch.ones(d_in) * 0.5)
        self.mix_r = nn.Parameter(torch.ones(d_in) * 0.5)
        self.key_proj = nn.Linear(d_in, hidden, bias=False)
        self.receptance_proj = nn.Linear(d_in, d_in, bias=False)
        self.value_proj = nn.Linear(hidden, d_in, bias=False)
        self.key_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.receptance_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.value_proj.weight.data.normal_(
            mean=0.0, std=1.0 / math.sqrt(hidden if hidden > 0 else 1)
        )

    def _init_diff_attention(self, d_in: int) -> None:
        n_heads = max(1, d_in // 64)
        head_dim = d_in // n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_in, n_heads * 2 * head_dim, bias=False)
        self.k_proj = nn.Linear(d_in, n_heads * 2 * head_dim, bias=False)
        self.v_proj = nn.Linear(d_in, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_in, bias=False)
        self.lambda_param = nn.Parameter(torch.tensor(0.5))
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            proj.weight.data.normal_(std=0.02)

    def _init_gated_delta(self, d_in: int) -> None:
        self.q_proj = nn.Linear(d_in, d_in, bias=False)
        self.k_proj = nn.Linear(d_in, d_in, bias=False)
        self.v_proj = nn.Linear(d_in, d_in, bias=False)
        self.o_proj = nn.Linear(d_in, d_in, bias=False)
        self.alpha_proj = nn.Linear(d_in, d_in, bias=False)
        self.beta_proj = nn.Linear(d_in, d_in, bias=False)
        for proj in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.alpha_proj,
            self.beta_proj,
        ):
            proj.weight.data.normal_(std=0.02)

    def _init_dplr_gated_delta(self, d_in: int) -> None:
        rank = max(1, d_in // 8)
        self.q_proj = nn.Linear(d_in, d_in, bias=False)
        self.k_proj = nn.Linear(d_in, d_in, bias=False)
        self.v_proj = nn.Linear(d_in, d_in, bias=False)
        self.o_proj = nn.Linear(d_in, d_in, bias=False)
        self.diag_proj = nn.Linear(d_in, d_in, bias=False)
        self.beta_proj = nn.Linear(d_in, d_in, bias=False)
        self.lr_in = nn.Linear(d_in, rank, bias=False)
        self.lr_out = nn.Linear(rank, d_in, bias=False)
        self.dplr_rank = rank
        for proj in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.diag_proj,
            self.beta_proj,
            self.lr_in,
            self.lr_out,
        ):
            proj.weight.data.normal_(std=0.02)

    def _init_difficulty_routed_attention(self, d_in: int) -> None:
        """Difficulty-routed: difficulty scorer + QKV for hard tokens + cheap path for easy."""
        self.difficulty_proj = nn.Linear(d_in, 1, bias=True)
        self.q_proj = nn.Linear(d_in, d_in, bias=False)
        self.k_proj = nn.Linear(d_in, d_in, bias=False)
        self.v_proj = nn.Linear(d_in, d_in, bias=False)
        self.o_proj = nn.Linear(d_in, d_in, bias=False)
        self.easy_proj = nn.Linear(d_in, d_in, bias=False)
        for p in (self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.easy_proj):
            p.weight.data.normal_(std=0.02)
        nn.init.zeros_(self.difficulty_proj.bias)

    def _init_strided_attention(self, d_in: int) -> None:
        """Strided/dilated multi-head attention with different strides per head."""
        self.q_proj = nn.Linear(d_in, d_in, bias=False)
        self.k_proj = nn.Linear(d_in, d_in, bias=False)
        self.v_proj = nn.Linear(d_in, d_in, bias=False)
        self.o_proj = nn.Linear(d_in, d_in, bias=False)
        for p in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            p.weight.data.normal_(std=0.02)

    def _init_gated_progressive_attention(self, d_in: int) -> None:
        """Progressive attention: learned gate controls attention strength 0->1."""
        self.q_proj = nn.Linear(d_in, d_in, bias=False)
        self.k_proj = nn.Linear(d_in, d_in, bias=False)
        self.v_proj = nn.Linear(d_in, d_in, bias=False)
        self.o_proj = nn.Linear(d_in, d_in, bias=False)
        self.gate_proj = nn.Linear(d_in, d_in, bias=True)
        for p in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            p.weight.data.normal_(std=0.02)
        # Initialize gate bias negative so attention starts OFF and learns to engage
        nn.init.constant_(self.gate_proj.bias, -2.0)

    def _init_gated_linear_attention(self, d_in: int) -> None:
        """GLA: Q, K, V projections + decay gate for adaptive memory."""
        self.q_proj = nn.Linear(d_in, d_in, bias=False)
        self.k_proj = nn.Linear(d_in, d_in, bias=False)
        self.v_proj = nn.Linear(d_in, d_in, bias=False)
        self.o_proj = nn.Linear(d_in, d_in, bias=False)
        self.gate_proj = nn.Linear(d_in, d_in, bias=False)
        for p in (self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.gate_proj):
            p.weight.data.normal_(std=0.02)

    def _init_long_conv_hyena(self, d_in: int) -> None:
        """Hyena long conv: implicit conv kernel (MLP-parameterized) + gating."""
        self.in_proj = nn.Linear(d_in, d_in * 2, bias=False)
        self.out_proj = nn.Linear(d_in, d_in, bias=False)
        # Implicit kernel: small MLP that maps position -> kernel weight
        self.kernel_net = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, d_in)
        )
        self.in_proj.weight.data.normal_(std=0.02)
        self.out_proj.weight.data.normal_(std=0.02)

    def _init_associative_memory(self, d_in: int) -> None:
        """Modern Hopfield: query/key projections for content-addressed retrieval."""
        self.query_proj = nn.Linear(d_in, d_in, bias=False)
        self.memory_proj = nn.Linear(d_in, d_in, bias=False)
        self.value_proj = nn.Linear(d_in, d_in, bias=False)
        self.o_proj = nn.Linear(d_in, d_in, bias=False)
        self.beta = nn.Parameter(torch.tensor(1.0))  # inverse temperature
        for p in (self.query_proj, self.memory_proj, self.value_proj, self.o_proj):
            p.weight.data.normal_(std=0.02)

    def _init_role_slot_attention(self, config: Dict, d_in: int) -> None:
        """Persistent Role-Slot Attention params.

        Learned global latent Slots (Keys/Values) are queried by input tokens.
        """
        num_slots = int(config.get("num_slots", 16))
        self.num_slots = num_slots
        self.q_proj = nn.Linear(d_in, d_in, bias=False)
        self.slot_keys = nn.Parameter(torch.randn(num_slots, d_in) * 0.02)
        self.slot_values = nn.Parameter(torch.randn(num_slots, d_in) * 0.02)
        self.o_proj = nn.Linear(d_in, d_in, bias=False)
        self.q_proj.weight.data.normal_(std=0.02)
        self.o_proj.weight.data.normal_(std=0.02)

    def _init_mixture_of_recursions(self, d_in: int) -> None:
        """MoR: shared transform block + per-token depth router."""
        self.block_norm = nn.LayerNorm(d_in)
        self.block_ffn_up = nn.Linear(d_in, d_in * 2, bias=False)
        self.block_ffn_down = nn.Linear(d_in * 2, d_in, bias=False)
        self.block_gate = nn.Linear(d_in, d_in * 2, bias=False)
        self.depth_router = nn.Linear(d_in, 4, bias=True)  # 4 depth options: 1,2,3,4
        self.block_ffn_up.weight.data.normal_(std=0.02)
        self.block_ffn_down.weight.data.normal_(std=0.02)
        self.block_gate.weight.data.normal_(std=0.02)
        nn.init.zeros_(self.depth_router.bias)

    def _init_token_hodge_mixer(self, d_in: int) -> None:
        self.edge_proj = nn.Linear(d_in, d_in, bias=False)
        self.face_proj = nn.Linear(d_in, d_in, bias=False)
        self.gate_proj = nn.Linear(d_in, d_in, bias=True)
        self.out_proj = nn.Linear(d_in, d_in, bias=False)
        for proj in (self.edge_proj, self.face_proj, self.gate_proj, self.out_proj):
            proj.weight.data.normal_(std=0.02)
        nn.init.constant_(self.gate_proj.bias, -1.0)

    def _init_wavelet_packet_mix(self, d_in: int) -> None:
        self.low_proj = nn.Linear(d_in, d_in, bias=False)
        self.high_proj = nn.Linear(d_in, d_in, bias=False)
        self.gate_proj = nn.Linear(d_in, d_in, bias=False)
        self.out_proj = nn.Linear(d_in, d_in, bias=False)
        self.wavelet_low_scale = nn.Parameter(torch.ones(d_in))
        self.wavelet_high_scale = nn.Parameter(torch.zeros(d_in))
        for proj in (self.low_proj, self.high_proj, self.gate_proj, self.out_proj):
            proj.weight.data.normal_(std=0.02)

    def _init_retention_mix(self, d_in: int) -> None:
        self.q_proj = nn.Linear(d_in, d_in, bias=False)
        self.k_proj = nn.Linear(d_in, d_in, bias=False)
        self.v_proj = nn.Linear(d_in, d_in, bias=False)
        self.o_proj = nn.Linear(d_in, d_in, bias=False)
        self.retention_log_decay = nn.Parameter(torch.linspace(-3.0, -0.1, d_in))
        self.retention_phase = nn.Parameter(torch.zeros(d_in))
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            proj.weight.data.normal_(std=0.02)

    def _init_product_key_memory(self, config: Dict, d_in: int) -> None:
        num_keys = max(2, min(64, int(config.get("num_keys", 32))))
        self.pkm_num_keys = num_keys
        self.pkm_top_k = max(1, min(8, int(config.get("top_k", 4))))
        left_dim = max(1, d_in // 2)
        right_dim = max(1, d_in - left_dim)
        self.pkm_left_dim = left_dim
        self.pkm_right_dim = right_dim
        self.key_left = nn.Parameter(torch.randn(num_keys, left_dim) * 0.02)
        self.key_right = nn.Parameter(torch.randn(num_keys, right_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(num_keys * num_keys, d_in) * 0.02)
        self.o_proj = nn.Linear(d_in, d_in, bias=False)
        self.o_proj.weight.data.normal_(std=0.02)

    def _init_gated_lane_blend(self, config: Dict, d_in: int) -> None:
        n_lanes = int(config.get("n_lanes", 3))
        self.lane_scorer = self._make_param((n_lanes, d_in), std=0.02)
        projs = []
        for _ in range(n_lanes):
            p = nn.Parameter(torch.empty(d_in, d_in))
            p.data.normal_(std=0.02)
            projs.append(p)
        self.lane_projs = nn.ParameterList(projs)

    def _init_depth_gated_transform(self, config: Dict, d_in: int) -> None:
        max_depth = max(1, min(6, int(config.get("max_depth", 3))))
        self.depth_scorer = self._make_param((max_depth, d_in), std=0.02)
        projs = []
        for _ in range(max_depth):
            p = nn.Parameter(torch.empty(d_in, d_in))
            p.data.normal_(std=0.02)
            projs.append(p)
        self.depth_projs = nn.ParameterList(projs)

    def _init_state_space(self, d_in: int) -> None:
        state_dim = 16
        self.ssm_state_dim = state_dim
        a_init = (
            -torch.arange(1, state_dim + 1, dtype=torch.float32)
            .unsqueeze(0)
            .expand(d_in, -1)
        )
        self.ssm_A = nn.Parameter(a_init)
        self.ssm_B = nn.Linear(d_in, d_in * state_dim, bias=False)
        self.ssm_C = nn.Linear(d_in * state_dim, d_in, bias=False)
        self.ssm_D = nn.Parameter(torch.ones(d_in))
        self.ssm_dt = nn.Linear(d_in, d_in)
        self.ssm_B.weight.data.normal_(std=0.02)
        self.ssm_C.weight.data.normal_(std=0.02)
        self.ssm_dt.weight.data.normal_(std=0.02)
        self.ssm_dt.bias.data.fill_(0.0)

    def _init_conv_only(self, d_in: int) -> None:
        self.conv_dw = nn.Conv1d(d_in, d_in, 3, padding=2, groups=d_in)
        self.conv_dw.weight.data.normal_(std=0.01)
        self.conv_proj = nn.Linear(d_in, d_in, bias=False)
        self.conv_proj.weight.data.normal_(std=0.01)

    def _init_difficulty_blend_3way(self, d_in: int) -> None:
        self.gate_proj = self._make_param((3, d_in), std=0.02)
        rank = max(d_in // 4, 1)
        self.U_mid = self._make_param((rank, d_in), std=0.02)
        self.V_mid = self._make_param((d_in, rank), std=0.02)
        hidden = d_in * 2
        self.heavy_mlp = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_in),
        )
        self.heavy_mlp[0].weight.data.normal_(std=0.02)
        self.heavy_mlp[2].weight.data.normal_(std=0.02)

    def _init_score_depth_blend(self, config: Dict, d_in: int) -> None:
        max_depth = int(config.get("max_depth", 3))
        projs = []
        for _ in range(max_depth):
            p = nn.Parameter(torch.empty(d_in, d_in))
            p.data.normal_(std=0.02)
            projs.append(p)
        self.step_projs = nn.ParameterList(projs)

    def _init_cheap_verify_blend(self, d_in: int) -> None:
        self.cheap_proj = self._make_param((d_in, d_in), std=0.02)
        self.verify_gate = self._make_param((1, d_in), std=0.02)

    def _init_hybrid_sparse_router(self, config: Dict, d_in: int) -> None:
        lane_count = max(2, min(int(config.get("lane_count", 3)), 8))
        self.hybrid_gate_proj = self._make_param((1, d_in), std=0.02)
        self.hybrid_lane_proj = self._make_param((lane_count, d_in), std=0.02)
        lane_weights = []
        for _ in range(lane_count):
            lane_weights.append(self._make_param((d_in, d_in), std=0.02))
        self.hybrid_lane_weights = nn.ParameterList(lane_weights)
        self.hybrid_default_proj = self._make_param((d_in, d_in), std=0.02)

    def _init_lane_conditioned_block(self, d_in: int) -> None:
        self.lane_block_weight = self._make_param((d_in, d_in), std=0.02)

    def _init_depth_weighted_proj(self, config: Dict, d_in: int) -> None:
        max_depth = max(1, min(6, int(config.get("max_depth", 3))))
        self.depth_scorer = self._make_param((max_depth, d_in), std=0.02)
        projs = []
        for _ in range(max_depth):
            p = nn.Parameter(torch.empty(d_in, d_in))
            p.data.normal_(std=0.02)
            projs.append(p)
        self.step_projs = nn.ParameterList(projs)

    def _init_padic_depth_route(self, config: Dict, d_in: int) -> None:
        # Same per-step transforms as depth_weighted_proj, but NO learned softmax scorer.
        # Routing is driven by the token's intrinsic p-adic valuation onto learnable depth
        # anchors (standardized space, init spread so depths cover the valuation range) with
        # a learnable reciprocal sharpness — see _op_padic_depth_route.
        max_depth = max(1, min(6, int(config.get("max_depth", 3))))
        projs = []
        for _ in range(max_depth):
            p = nn.Parameter(torch.empty(d_in, d_in))
            p.data.normal_(std=0.02)
            projs.append(p)
        self.step_projs = nn.ParameterList(projs)
        self.depth_anchors = nn.Parameter(torch.linspace(-1.5, 1.5, max_depth))
        self.route_log_sharpness = nn.Parameter(torch.zeros(()))

    def _init_padic_gated_mixer(self, d_in: int) -> None:
        # Learned highway gate informed by p-adic valuation + a learned projection.
        # gate_bias=0 -> sigmoid starts balanced (0.5); see _op_padic_gated_mixer.
        self.gate_x = self._make_param((d_in, d_in), std=0.02)
        self.gate_v = self._make_param((d_in, d_in), std=0.02)
        self.proj_w = self._make_param((d_in, d_in), std=0.02)
        self.gate_bias = nn.Parameter(torch.zeros(d_in))

    def _init_sinkhorn_ot_mix(self, d_in: int) -> None:
        # Optimal-transport mixer params: query/key projections define the transport cost
        # (squared Euclidean on L2-normalized projections), the value is read by the plan, and
        # the output projection merges. sinkhorn_log_eps=0 -> eps = softplus(0)+0.1 ~= 0.79
        # (moderate transport sharpness). See _op_sinkhorn_ot_mix.
        self.ot_q_proj = self._make_param((d_in, d_in), std=0.02)
        self.ot_k_proj = self._make_param((d_in, d_in), std=0.02)
        self.ot_v_proj = self._make_param((d_in, d_in), std=0.02)
        self.ot_o_proj = self._make_param((d_in, d_in), std=0.02)
        self.sinkhorn_log_eps = nn.Parameter(torch.zeros(()))

    def _init_ultrametric_tree_mix(self, d_in: int) -> None:
        # Content-addressed ultrametric (Bruhat-Tits-tree) mixer params. q/k/v/o projections map
        # tokens to content codes; ut_scale_dirs are L=8 learned resolution directions, ut_scale_bias
        # the per-scale agreement threshold, ut_scale_log_temp the sharpness (softplus-floored). The
        # affinity is the PRODUCT of per-scale agreements (see _op_ultrametric_tree_mix). L=8 is FIXED
        # so param accounting matches param_formula="D*D*4 + 8*D + 9": 4*D*D (projs) + 8*D (dirs) +
        # 8 (bias) + 1 (temp).
        self.ut_q_proj = self._make_param((d_in, d_in), std=0.02)
        self.ut_k_proj = self._make_param((d_in, d_in), std=0.02)
        self.ut_v_proj = self._make_param((d_in, d_in), std=0.02)
        self.ut_o_proj = self._make_param((d_in, d_in), std=0.02)
        self.ut_scale_dirs = self._make_param((8, d_in), std=0.02)
        self.ut_scale_bias = nn.Parameter(torch.zeros(8))
        self.ut_scale_log_temp = nn.Parameter(torch.zeros(()))

    def _init_fno_spectral_mix(self, d_in: int) -> None:
        # Fourier neural-operator (FNO) mixer params. in/out projections bookend a spectral branch:
        # rfft the sequence, apply a per-low-mode COMPLEX channel weight (stored as real/imag), zero
        # the high modes, irfft back (see _op_fno_spectral_mix). 4 low modes are FIXED so param
        # accounting matches param_formula="D*D*10 + D": in_proj D*D + out_proj D*D + 4*D*D*2 (modes
        # real+imag) + D (bias).
        self.fno_in_proj = self._make_param((d_in, d_in), std=0.02)
        self.fno_out_proj = self._make_param((d_in, d_in), std=0.02)
        self.fno_modes_real = self._make_param((4, d_in, d_in), std=0.02)
        self.fno_modes_imag = self._make_param((4, d_in, d_in), std=0.02)
        self.fno_bias = nn.Parameter(torch.zeros(d_in))

    def _init_relu_gated_moe(self, config: Dict, d_in: int) -> None:
        n_experts = int(config.get("n_experts", 8))
        self.gate_proj = self._make_param((n_experts, d_in), std=0.02)
        expert_list = []
        for _ in range(n_experts):
            p = nn.Parameter(torch.empty(d_in, d_in))
            p.data.normal_(std=0.02)
            expert_list.append(p)
        self.expert_weights = nn.ParameterList(expert_list)

    def _init_token_class_proj(self, config: Dict, d_in: int) -> None:
        n_classes = int(config.get("n_classes", 2))
        self.classifier_weight = self._make_param((n_classes, d_in), std=0.02)
        self.classifier_proj_back = self._make_param((d_in, n_classes), std=0.02)

    def _init_adaptive_rank_gate(self, d_in: int, d_out: int) -> None:
        self.weight_full = self._make_param((d_out, d_in), std=0.02)
        self.compress_param = nn.Parameter(torch.zeros(1))
        self.token_gate = self._make_param((1, d_in), std=0.02)
        rank = max(d_in // 8, 1)
        self.U_comp = self._make_param((rank, d_in), std=0.02)
        self.V_comp = self._make_param((d_out, rank), std=0.02)

    def _init_dual_compression_blend(self, d_in: int, d_out: int) -> None:
        rank = max(d_in // 8, 1)
        self.U_lr = self._make_param((rank, d_in), std=0.02)
        self.V_lr = self._make_param((d_out, rank), std=0.02)
        rank_bn = max(d_in // 4, 1)
        self.W_down = self._make_param((rank_bn, d_in), std=0.02)
        self.W_up = self._make_param((d_out, rank_bn), std=0.02)

    def _init_ternary_projection(self, config: Dict, d_in: int, d_out: int) -> None:
        self.weight = self._make_param((d_out, d_in), std=0.02)
        if config.get("bias"):
            self.bias = nn.Parameter(torch.zeros(d_out))

    def _init_latent_attention_compressor(self, d_in: int) -> None:
        latent_dim = max(d_in // 4, 16)
        self.kv_compress = self._make_param((latent_dim, d_in), std=0.02)
        self.kv_up = self._make_param((d_in * 2, latent_dim), std=0.02)

    def _init_signal_conditioned_compression(self, d_in: int) -> None:
        self.weight_full = self._make_param((d_in, d_in), std=0.02)
        rank = max(d_in // 8, 1)
        self.U_comp = self._make_param((rank, d_in), std=0.02)
        self.V_comp = self._make_param((d_in, rank), std=0.02)

    def _init_calibrated_branch_merge(self, config: Dict, d_in: int) -> None:
        n_branches = max(2, int(config.get("n_branches", 2)))
        self.branch_score_proj = self._make_param((n_branches, d_in), std=0.02)
        self.branch_bias = nn.Parameter(torch.zeros(n_branches))
        self.branch_gain = nn.Parameter(torch.zeros(n_branches))

    def _init_chebyshev_spectral_mix(self, config: Dict, d_in: int) -> None:
        order = max(2, min(config.get("chebyshev_order", 6), 16))
        for idx in range(order):
            std = order**-0.5
            p = self._make_param((d_in,), std=std)
            if idx == 1:
                p.data.add_(1.0)
            setattr(self, f"cheb_c{idx}", p)

    def _init_sparse_bottleneck_moe(self, config: Dict, d_in: int) -> None:
        n_ways = max(2, min(config.get("n_ways", 4), 16))
        hidden = d_in // n_ways
        self.gate_weight = self._make_param((d_in, n_ways), std=0.02)
        for idx in range(n_ways):
            setattr(
                self, f"expert_down_{idx}", self._make_param((d_in, hidden), std=0.02)
            )
            setattr(
                self, f"expert_up_{idx}", self._make_param((hidden, d_in), std=0.02)
            )
        self._expert_downs = [
            getattr(self, f"expert_down_{idx}") for idx in range(n_ways)
        ]
        self._expert_ups = [getattr(self, f"expert_up_{idx}") for idx in range(n_ways)]

    def _init_hetero_moe(self, d_in: int) -> None:
        self.gate_weight = self._make_param((3, d_in), std=0.02)
        self.attn_qkv = self._make_param((3 * d_in, d_in), std=0.02)
        self.attn_out = self._make_param((d_in, d_in), std=0.02)
        self.conv_proj = self._make_param((d_in, d_in), std=0.02)
        self.ssm_B_proj = self._make_param((d_in, d_in), std=0.02)
        self.ssm_C_proj = self._make_param((d_in, d_in), std=0.02)
        self.ssm_D = self._make_param((d_in,), std=0.02)

    def _init_arch_router(self, d_in: int) -> None:
        self.gate_weight = self._make_param((3, d_in), std=0.02)
        self.attn_qkv = self._make_param((3 * d_in, d_in), std=0.02)
        self.attn_out = self._make_param((d_in, d_in), std=0.02)
        self.arch_ffn = self._make_param((d_in, d_in), std=0.02)
        self.conv_proj = self._make_param((d_in, d_in), std=0.02)
        self.ssm_B_proj = self._make_param((d_in, d_in), std=0.02)
        self.ssm_C_proj = self._make_param((d_in, d_in), std=0.02)
        self.ssm_D = self._make_param((d_in,), std=0.02)
        self.arch_proj = self._make_param((d_in, d_in), std=0.02)
        hidden = d_in * 4
        self.mlp_up = self._make_param((hidden, d_in), std=0.02)
        self.mlp_down = self._make_param((d_in, hidden), std=0.02)

    def _init_compute_budget_router(self, d_in: int) -> None:
        self.gate_weight = self._make_param((3, d_in), std=0.02)
        self.cheap_proj = self._make_param((d_in, d_in), std=0.02)
        self.conv_proj = self._make_param((d_in, d_in), std=0.02)
        self.attn_qkv = self._make_param((3 * d_in, d_in), std=0.02)
        self.attn_out = self._make_param((d_in, d_in), std=0.02)
