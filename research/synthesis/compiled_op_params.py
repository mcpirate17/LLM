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

        dispatch = {
            "linear_proj": lambda: setattr(
                self, "weight", self._make_param((d_out, d_in), std=0.02)
            ),
            "linear_proj_down": lambda: setattr(
                self, "weight", self._make_param((d_out, d_in), std=0.02)
            ),
            "linear_proj_up": lambda: setattr(
                self, "weight", self._make_param((d_out, d_in), std=0.02)
            ),
            "fused_linear_gelu": lambda: (
                setattr(self, "weight", self._make_param((d_out, d_in), std=0.02)),
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
            "swiglu_mlp": lambda: self._init_swiglu_mlp(config, d_in),
            "rwkv_channel": lambda: self._init_rwkv_channel(config, d_in),
            "softmax_attention": lambda: self._init_attention_stack(
                "softmax_attention", d_in
            ),
            "linear_attention": lambda: self._init_attention_stack(
                "linear_attention", d_in
            ),
            "graph_attention": lambda: self._init_attention_stack(
                "graph_attention", d_in
            ),
            "diff_attention": lambda: self._init_diff_attention(d_in),
            "gated_delta": lambda: self._init_gated_delta(d_in),
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
            "mixture_of_recursions": lambda: self._init_mixture_of_recursions(d_in),
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
        if op_name in ("softmax_attention", "graph_attention"):
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
