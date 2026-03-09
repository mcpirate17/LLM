"""Registry and initialization dispatch for CompiledOp parameters."""
from typing import Callable, Dict, Any, Tuple, Optional
import math
import torch
import torch.nn as nn

InitFn = Callable[[Any, Any, Dict[str, Any], Any], None]
_INIT_TABLE: Dict[str, InitFn] = {}

def register_init(op_name: str) -> Callable[[InitFn], InitFn]:
    def decorator(fn: InitFn) -> InitFn:
        _INIT_TABLE[op_name] = fn
        return fn
    return decorator

def _make_param(self, shape, std=0.02, dtype=None):
    if hasattr(self, "_make_param"):
        return self._make_param(shape, std=std)
    raise NotImplementedError("Missing _make_param on primitive")

def _init_attention_stack(self, op_name: str, op, D_in: int) -> None:
    n_heads = max(1, D_in // 64)
    head_dim = D_in // n_heads
    self.n_heads = n_heads
    self.head_dim = head_dim
    if op_name in ('softmax_attention', 'graph_attention'):
        self.attn_scale = head_dim ** (-0.5)
    self.q_proj = nn.Linear(D_in, n_heads * head_dim, bias=False)
    self.k_proj = nn.Linear(D_in, n_heads * head_dim, bias=False)
    self.v_proj = nn.Linear(D_in, n_heads * head_dim, bias=False)
    self.o_proj = nn.Linear(n_heads * head_dim, D_in, bias=False)
    self.q_proj.weight.data.normal_(std=0.02)
    self.k_proj.weight.data.normal_(std=0.02)
    self.v_proj.weight.data.normal_(std=0.02)
    self.o_proj.weight.data.normal_(std=0.02)
    if op_name == 'graph_attention':
        self.edge_proj = nn.Linear(D_in, D_in, bias=False)
        self.edge_proj.weight.data.normal_(std=0.02)

def _init_math_space(self, op, D_in: int, D_out: int, config: Dict) -> None:
    if op.has_params:
        self.weight = self._make_param((D_out, D_in), std=0.02)
    if op.name in ('padic_expand', 'padic_residual'):
        self.weight = self._make_param((D_in, D_in * 2), std=0.02)
    elif op.name == 'rotor_transform':
        self.rotor = nn.Parameter(torch.randn(8) * 0.02)
    elif op.name == 'poincare_add':
        self.bias = nn.Parameter(torch.zeros(D_in))
    elif op.name == 'hyp_linear':
        self.weight = self._make_param((D_in, D_in), std=0.02)
    elif op.name == 'tropical_router':
        n_exp = int(config.get('n_experts', 8))
        self.centroids = nn.Parameter(torch.randn(n_exp, D_in) * 0.02)

def _init_moe_topk(self, config: Dict, d_in: int) -> None:
    n_experts = int(config.get('num_experts', 4))
    self.gate_weight = self._make_param((n_experts, d_in), std=0.02)
    hidden = int(d_in * float(config.get('mlp_ratio', 2.0)))
    self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d_in, hidden, bias=False), nn.GELU(), nn.Linear(hidden, d_in, bias=False)) for _ in range(n_experts)])
    for expert in self.experts:
        expert[0].weight.data.normal_(mean=0.0, std=0.02)
        expert[2].weight.data.normal_(mean=0.0, std=1.0 / math.sqrt(hidden if hidden > 0 else 1))

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
    hidden = int(d_in * float(config.get('mlp_ratio', 3.0)))
    self.gate_proj = nn.Linear(d_in, hidden, bias=False)
    self.up_proj = nn.Linear(d_in, hidden, bias=False)
    self.down_proj = nn.Linear(hidden, d_in, bias=False)
    self.gate_proj.weight.data.normal_(mean=0.0, std=0.02)
    self.up_proj.weight.data.normal_(mean=0.0, std=0.02)
    self.down_proj.weight.data.normal_(mean=0.0, std=1.0 / math.sqrt(hidden if hidden > 0 else 1))

def _init_rwkv_channel(self, config: Dict, d_in: int) -> None:
    hidden = int(d_in * float(config.get('mlp_ratio', 3.0)))
    self.mix_k = nn.Parameter(torch.ones(d_in) * 0.5)
    self.mix_r = nn.Parameter(torch.ones(d_in) * 0.5)
    self.key_proj = nn.Linear(d_in, hidden, bias=False)
    self.receptance_proj = nn.Linear(d_in, d_in, bias=False)
    self.value_proj = nn.Linear(hidden, d_in, bias=False)
    self.key_proj.weight.data.normal_(mean=0.0, std=0.02)
    self.receptance_proj.weight.data.normal_(mean=0.0, std=0.02)
    self.value_proj.weight.data.normal_(mean=0.0, std=1.0 / math.sqrt(hidden if hidden > 0 else 1))

def _init_state_space(self, d_in: int) -> None:
    state_dim = 16
    self.ssm_state_dim = state_dim
    self.ssm_A = nn.Parameter(torch.randn(d_in, state_dim) * 0.01)
    self.ssm_B = nn.Linear(d_in, d_in * state_dim, bias=False)
    self.ssm_C = nn.Linear(d_in * state_dim, d_in, bias=False)
    self.ssm_D = nn.Parameter(torch.ones(d_in))
    self.ssm_dt = nn.Linear(d_in, d_in)
    self.ssm_B.weight.data.normal_(std=0.02)
    self.ssm_C.weight.data.normal_(std=0.02)
    self.ssm_dt.weight.data.normal_(std=0.02)

def _init_conv_only(self, d_in: int) -> None:
    self.conv_dw = nn.Conv1d(d_in, d_in, 3, padding=2, groups=d_in)
    self.conv_proj = nn.Linear(d_in, d_in, bias=False)
    self.conv_proj.weight.data.normal_(std=0.02)

def _init_adaptive_lane_mixer(self, d_in: int) -> None:
    self.gate_proj = self._make_param((3, d_in), std=0.02)
    rank = max(d_in // 4, 1)
    self.U_mid = self._make_param((rank, d_in), std=0.02)
    self.V_mid = self._make_param((d_in, rank), std=0.02)
    hidden = d_in * 2
    self.heavy_mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_in))
    self.heavy_mlp[0].weight.data.normal_(std=0.02)
    self.heavy_mlp[2].weight.data.normal_(std=0.02)

def _init_mixed_recursion_gate(self, config: Dict, d_in: int) -> None:
    max_depth = int(config.get('max_depth', 3))
    projs = []
    for _ in range(max_depth):
        p = nn.Parameter(torch.empty(d_in, d_in))
        p.data.normal_(std=0.02)
        projs.append(p)
    self.step_projs = nn.ParameterList(projs)

def _init_token_type_classifier(self, config: Dict, d_in: int) -> None:
    n_classes = int(config.get('n_classes', 2))
    self.classifier_weight = self._make_param((n_classes, d_in), std=0.02)
    self.classifier_proj_back = self._make_param((d_in, n_classes), std=0.02)

def _init_progressive_compression_gate(self, d_in: int, d_out: int) -> None:
    self.weight_full = self._make_param((d_out, d_in), std=0.02)
    self.compress_param = nn.Parameter(torch.zeros(1))
    rank = max(d_in // 8, 1)
    self.U_comp = self._make_param((rank, d_in), std=0.02)
    self.V_comp = self._make_param((d_out, rank), std=0.02)

def _init_compression_mixture_experts(self, d_in: int, d_out: int) -> None:
    self.expert_weights = nn.Parameter(torch.ones(2))
    rank = max(d_in // 8, 1)
    self.U_lr = self._make_param((rank, d_in), std=0.02)
    self.V_lr = self._make_param((d_out, rank), std=0.02)
    rank_bn = max(d_in // 4, 1)
    self.W_down = self._make_param((rank_bn, d_in), std=0.02)
    self.W_up = self._make_param((d_out, rank_bn), std=0.02)

def _init_ternary_projection(self, config: Dict, d_in: int, d_out: int) -> None:
    self.weight = self._make_param((d_out, d_in), std=0.02)
    if config.get('bias'):
        self.bias = nn.Parameter(torch.zeros(d_out))

def _init_latent_attention_compressor(self, d_in: int) -> None:
    latent_dim = max(d_in // 4, 16)
    self.kv_compress = self._make_param((latent_dim, d_in), std=0.02)
    self.kv_up = self._make_param((d_in * 2, latent_dim), std=0.02)

def _init_routing_conditioned_compression(self, d_in: int) -> None:
    self.weight_full = self._make_param((d_in, d_in), std=0.02)
    rank = max(d_in // 8, 1)
    self.U_comp = self._make_param((rank, d_in), std=0.02)
    self.V_comp = self._make_param((d_in, rank), std=0.02)

@register_init("linear_proj")
def _init_linear_proj(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.weight = _make_param(self, (D_out, D_in), std=0.02)

@register_init("linear_proj_down")
def _init_linear_proj_down(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.weight = _make_param(self, (D_out, D_in), std=0.02)

@register_init("linear_proj_up")
def _init_linear_proj_up(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.weight = _make_param(self, (D_out, D_in), std=0.02)

@register_init("fused_linear_gelu")
def _init_fused_linear_gelu(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.weight = _make_param(self, (D_out, D_in), std=0.02)
    self.bias = nn.Parameter(torch.zeros(D_out))

@register_init("learnable_scale")
def _init_learnable_scale(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.scale = nn.Parameter(torch.ones(D_in))

@register_init("learnable_bias")
def _init_learnable_bias(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.bias = nn.Parameter(torch.zeros(D_in))

@register_init("selective_scan")
def _init_selective_scan(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.A_log = _make_param(self, (D_in,), std=0.1)
    self.dt_proj = _make_param(self, (D_in,), std=0.1)
    self.B_proj = nn.Linear(D_in, D_in, bias=False)
    self.C_proj = nn.Linear(D_in, D_in, bias=False)
    self.B_proj.weight.data.normal_(std=0.02)
    self.C_proj.weight.data.normal_(std=0.02)

@register_init("conv1d_seq")
def _init_conv1d_seq(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.conv_weight = _make_param(self, (D_in, 1, 3), std=1.0 / math.sqrt(3))

@register_init("topk_gate")
def _init_topk_gate(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.gate_proj = _make_param(self, (2, D_in), std=0.02)

@register_init("moe_topk")
def _init_moe_topk(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    _init_moe_topk(self, config, D_in)

@register_init("moe_2expert")
def _init_moe_2expert(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.gate_proj = _make_param(self, (2, D_in), std=0.02)
    self.expert_0_weight = _make_param(self, (D_in, D_in), std=0.02)
    self.expert_1_weight = _make_param(self, (D_in, D_in), std=0.02)

@register_init("nm_sparse_linear")
def _init_nm_sparse_linear(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.weight = _make_param(self, (D_out, D_in), std=0.02)
    self.sparsity_n = int(config.get('n', 2))
    self.sparsity_m = int(config.get('m', 4))

@register_init("block_sparse_linear")
def _init_block_sparse_linear(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.weight = _make_param(self, (D_out, D_in), std=0.02)
    self.block_size = max(1, int(config.get('block_size', 16)))
    self.block_density = float(max(0.05, min(1.0, config.get('block_density', 0.25))))

@register_init("rmsnorm")
def _init_rmsnorm(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.weight = nn.Parameter(torch.ones(D_in))

@register_init("layernorm")
def _init_layernorm(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.weight = nn.Parameter(torch.ones(D_in))
    self.bias = nn.Parameter(torch.zeros(D_in))

@register_init("gated_linear")
def _init_gated_linear(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.linear_weight = _make_param(self, (D_out, D_in), std=0.02)
    self.gate_weight = _make_param(self, (D_out, D_in), std=0.02)
    self.linear_bias = nn.Parameter(torch.zeros(D_out))
    self.gate_bias = nn.Parameter(torch.zeros(D_out))

@register_init("rwkv_time_mixing")
def _init_rwkv_time_mixing(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.w_decay = nn.Parameter(torch.ones(D_in) * -0.5)
    self.u_bonus = nn.Parameter(torch.zeros(D_in))
    self.W_k = _make_param(self, (D_in, D_in), std=0.02)
    self.W_v = _make_param(self, (D_in, D_in), std=0.02)
    self.W_r = _make_param(self, (D_in, D_in), std=0.02)
    self.W_o = _make_param(self, (D_in, D_in), std=0.02)

@register_init("embedding_lookup")
def _init_embedding_lookup(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.embed_table = nn.Embedding(int(config.get('vocab_size', 32000)), D_in)

@register_init("rope_rotate")
def _init_rope_rotate(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    None

@register_init("cosine_similarity")
def _init_cosine_similarity(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    None

@register_init("gather_topk")
def _init_gather_topk(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    None

@register_init("semi_structured_2_4_linear")
def _init_semi_structured_2_4_linear(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.weight = _make_param(self, (D_out, D_in), std=0.02)
    self.sparse_kernel_ready = bool(D_in % 4 == 0 and D_out % 4 == 0)

@register_init("basis_expansion")
def _init_basis_expansion(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.weight = nn.Parameter(torch.randn(4, D_in) * 0.5)

@register_init("integral_kernel")
def _init_integral_kernel(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.weight = nn.Parameter(torch.randn(D_in, D_in) * 0.02)

@register_init("fixed_point_iter")
def _init_fixed_point_iter(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.weight = nn.Parameter(torch.randn(D_in + 1, D_in) * 0.02)

@register_init("low_rank_proj")
def _init_low_rank_proj(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    _init_low_rank_proj(self, D_in)

@register_init("grouped_linear")
def _init_grouped_linear(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    _init_grouped_linear(self, D_in)

@register_init("bottleneck_proj")
def _init_bottleneck_proj(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    _init_bottleneck_proj(self, D_in)

@register_init("shared_basis_proj")
def _init_shared_basis_proj(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    _init_shared_basis_proj(self, D_in)

@register_init("tied_proj")
def _init_tied_proj(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    _init_tied_proj(self, D_in)

@register_init("swiglu_mlp")
def _init_swiglu_mlp(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    _init_swiglu_mlp(self, config, D_in)

@register_init("rwkv_channel")
def _init_rwkv_channel(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    _init_rwkv_channel(self, config, D_in)

@register_init("softmax_attention")
def _init_softmax_attention(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    _init_attention_stack(self, 'softmax_attention')

@register_init("linear_attention")
def _init_linear_attention(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    _init_attention_stack(self, 'linear_attention')

@register_init("graph_attention")
def _init_graph_attention(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    _init_attention_stack(self, 'graph_attention')

@register_init("state_space")
def _init_state_space(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    _init_state_space(self, D_in)

@register_init("conv_only")
def _init_conv_only(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    _init_conv_only(self, D_in)

@register_init("stdp_attention")
def _init_stdp_attention(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.log_tau = nn.Parameter(torch.tensor(0.0))

@register_init("adaptive_lane_mixer")
def _init_adaptive_lane_mixer(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    _init_adaptive_lane_mixer(self, D_in)

@register_init("mixed_recursion_gate")
def _init_mixed_recursion_gate(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    _init_mixed_recursion_gate(self, config, D_in)

@register_init("token_type_classifier")
def _init_token_type_classifier(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    _init_token_type_classifier(self, config, D_in)

@register_init("progressive_compression_gate")
def _init_progressive_compression_gate(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    _init_progressive_compression_gate(self, D_in, D_out)

@register_init("compression_mixture_experts")
def _init_compression_mixture_experts(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    _init_compression_mixture_experts(self, D_in, D_out)

@register_init("relu_gate_routing")
def _init_relu_gate_routing(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    self.gate_proj = _make_param(self, (int(config.get('n_experts', 8)), D_in), std=0.02)

@register_init("ternary_projection")
def _init_ternary_projection(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    _init_ternary_projection(self, config, D_in, D_out)

@register_init("latent_attention_compressor")
def _init_latent_attention_compressor(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    _init_latent_attention_compressor(self, D_in)

@register_init("routing_conditioned_compression")
def _init_routing_conditioned_compression(self, op, config: Dict, input_shape: Any):
    D_in = max(1, input_shape.dim)
    D_out = max(1, config.get("out_dim", D_in))
    _init_routing_conditioned_compression(self, D_in)
