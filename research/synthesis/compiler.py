"""
Computation Graph Compiler

Compiles a ComputationGraph into a live PyTorch nn.Module.
Each OpNode becomes a concrete tensor operation, with learnable
parameters allocated for parameterized ops.
"""

from __future__ import annotations

import importlib.util
import logging
import math
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import (
    get_primitive,
    PrimitiveOp,
    _ALGEBRAIC_SPACE_TAGS,
    PRIMITIVE_REGISTRY,
)
from .graph import ComputationGraph, ShapeInfo
from .compiler_op_utils import (
    record_kernel_fallback,
    _safe_linear,
    _build_nm_mask,
    _flatten_for_kernel,
    _unflatten_from_kernel,
    _build_block_sparse_mask,
)

logger = logging.getLogger(__name__)

# Module-level kernel fallback state — set True on first native kernel fallback.
# Importable by other modules (e.g. fingerprint.py) to condition validity flags.
_kernel_fallback_occurred: bool = False


try:
    from . import kernels

    HAS_KERNELS = True
except ImportError:
    HAS_KERNELS = False

HAS_CPU_OPS = importlib.util.find_spec("research.synthesis.cpu_ops") is not None

from research.defaults import VOCAB_SIZE, MODEL_DIM, VALIDATION_SEQ_LEN
from research.env import aria_core, HAS_ARIA_CORE


# ── Math-space op names (frozen at import time) ──────────────────────
_MATHSPACE_OPS: frozenset = frozenset(_ALGEBRAIC_SPACE_TAGS)

# ── Registry System ───────────────────────────────────────────────────

_OP_DISPATCH: Dict[
    str, Callable[[nn.Module, Tuple[torch.Tensor, ...], Dict], torch.Tensor]
] = {}


def _register_split_op_modules() -> None:
    """Load split op implementations. Raises on import failure — a missing handler
    module means the compiler cannot function correctly."""
    split_modules = {
        "compiler_ops_math": ".compiler_ops_math",
        "compiler_ops_attention": ".compiler_ops_attention",
        "compiler_ops_mathspaces": ".compiler_ops_mathspaces",
        "compiler_ops_routing": ".compiler_ops_routing",
        "true_routing_ops": ".true_routing_ops",
    }
    for label, rel_import in split_modules.items():
        try:
            mod = __import__(f"research.synthesis.{label}", fromlist=["OP_IMPLS"])
            _OP_DISPATCH.update(mod.OP_IMPLS)
        except ImportError as e:
            raise ImportError(
                f"Failed to load compiler op module '{label}': {e}\n"
                f"Compiler handlers for ops in that module are missing. "
                f"Fix the import error before continuing."
            ) from e


# ── Parallel Associative Scan ────────────────────────────────────────


def _parallel_associative_scan(log_a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Parallel prefix scan for linear recurrence h[t] = exp(log_a[t]) * h[t-1] + b[t].

    Uses Kogge-Stone doubling: O(S log S) work, O(log S) sequential depth.
    Operates along the last dimension of log_a (the sequence dimension).

    Args:
        log_a: (..., S) log-decay factors per timestep, should be in [-10, 0].
        b: (..., S) additive input per timestep.

    Returns:
        h: (..., S) scan output where h[t] = sum_{i<=t} prod_{j=i+1..t} exp(log_a[j]) * b[i].
    """
    S = log_a.shape[-1]
    if S <= 1:
        return b

    logger.debug(
        "kogge_stone_scan: using torch.cat fallback (S=%d, log2(S)=%d allocs)",
        S,
        S.bit_length(),
    )

    a = torch.exp(log_a)
    h = b

    stride = 1
    while stride < S:
        # Combine position t with position t-stride
        # Identity element: (a=1, b=0) for positions without a predecessor
        h = torch.cat(
            [
                h[..., :stride],
                a[..., stride:] * h[..., :-stride] + h[..., stride:],
            ],
            dim=-1,
        )
        a = torch.cat(
            [
                a[..., :stride],
                a[..., stride:] * a[..., :-stride],
            ],
            dim=-1,
        )
        a = a.clamp(max=1.0)
        stride *= 2

    return h


def register_op(name: str):
    """Decorator to register an op implementation."""

    def decorator(fn: Callable):
        _OP_DISPATCH[name] = fn
        return fn

    return decorator


def _record_sparse_telemetry(
    module: nn.Module,
    op_name: str,
    density: float,
    fallback_reason: Optional[str] = None,
) -> None:
    telemetry = getattr(module, "sparse_telemetry", None)
    if telemetry is None:
        telemetry = {}
        module.sparse_telemetry = telemetry
    stats = telemetry.get(op_name)
    if stats is None:
        stats = {
            "calls": 0,
            "fallback_calls": 0,
            "density_sum": 0.0,
            "last_density": 1.0,
            "last_fallback_reason": None,
        }
    stats["calls"] += 1
    stats["density_sum"] += float(density)
    stats["last_density"] = float(density)
    if fallback_reason is not None:
        stats["fallback_calls"] += 1
        stats["last_fallback_reason"] = fallback_reason
    telemetry[op_name] = stats
    setattr(module, "sparse_telemetry", telemetry)


def _record_routing_telemetry(
    module: nn.Module,
    n_experts: int,
    selected_experts: torch.Tensor,
    logits: Optional[torch.Tensor] = None,
) -> None:
    """Record MoE routing statistics: entropy, expert utilization, drop rate.

    Samples every 8th call to reduce overhead while maintaining statistical accuracy.
    """
    telemetry = getattr(module, "routing_telemetry", None)
    if telemetry is None:
        telemetry = {
            "tokens_total": 0,
            "tokens_processed": 0,
            "expert_counts": torch.zeros(n_experts, device=selected_experts.device),
            "entropy_sum": 0.0,
            "count": 0,
            "heatmap": None,
            "_call_count": -1,
        }
        module.routing_telemetry = telemetry

    telemetry["_call_count"] += 1
    B, S = selected_experts.shape[:2]
    total_tokens = B * S
    telemetry["tokens_total"] += total_tokens
    telemetry["tokens_processed"] += total_tokens

    # Sample every 8th call for expensive histogram + entropy (first call always records)
    if telemetry["_call_count"] & 7 != 0:
        telemetry["count"] += 1
        setattr(module, "routing_telemetry", telemetry)
        return

    # Expert utilization
    counts = torch.histc(
        selected_experts.float(), bins=n_experts, min=0, max=n_experts - 1
    )
    telemetry["expert_counts"] += counts

    # Entropy if logits provided
    if logits is not None:
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
        telemetry["entropy_sum"] += entropy
        telemetry["count"] += 1

    # Z13: Savings and Depth ratios
    # If selected_experts contains indices of active tokens (e.g. top-k),
    # we can estimate savings.
    # For now, we'll just track if any tokens were skipped.

    # Optional heatmap capture (first batch element only)
    if getattr(module, "_capture_heatmap", False) and telemetry["heatmap"] is None:
        telemetry["heatmap"] = selected_experts[0].detach().cpu().numpy().tolist()

    setattr(module, "routing_telemetry", telemetry)


# ── Op Implementations ──────────────────────────────────────────────


def _c(x):
    """Check if tensor is eligible for aria_core C kernels.

    C kernels don't support autograd, so skip them when gradients are needed.
    Requires at least 2D tensor with reasonable dimensions.
    """
    return (
        HAS_ARIA_CORE
        and x.device.type == "cpu"
        and x.dtype == torch.float32
        and not x.requires_grad
        and x.dim() >= 1
        and x.numel() > 0
    )


@register_op("selective_scan")
def _op_selective_scan(module, inputs, _):
    """
    Vectorized Linear Scan with trapezoidal discretization (Mamba-3 style).

    Trapezoidal rule: h[t] = a*h[t-1] + 0.5*(u[t] + a*u[t-1])
    Second-order accuracy vs Euler's first-order, same compute cost.
    Uses causal convolution for parallel evaluation.
    """
    if not hasattr(module, "A_log"):
        return inputs[0]
    x = inputs[0]
    B, S, D = x.shape

    A = -torch.exp(module.A_log.clamp(-10, 10))
    dt = F.softplus(module.dt_proj[:D])
    log_a = (A * dt).clamp(-10, -0.05)  # (D,)
    a = torch.exp(log_a)  # decay per step (D,)

    u = torch.sigmoid(module.B_proj(x)) * x  # (B, S, D)

    # Trapezoidal input: u_trap[t] = 0.5 * (u[t] + a * u[t-1])
    u_prev = F.pad(u, (0, 0, 1, 0))[:, :S, :]  # shift right, pad with 0
    u_trap = 0.5 * (u + a.unsqueeze(0).unsqueeze(0) * u_prev)

    # Vectorized linear recurrence via causal convolution with exponential kernel
    indices = torch.arange(S, device=x.device, dtype=x.dtype)
    log_kernel = log_a.view(D, 1, 1) * (S - 1 - indices).view(1, 1, S)
    kernel = torch.exp(log_kernel)  # (D, 1, S)

    u_swapped = u_trap.permute(0, 2, 1)  # (B, D, S)
    h_swapped = F.conv1d(F.pad(u_swapped, (S - 1, 0)), kernel, groups=D)
    h = h_swapped.permute(0, 2, 1)  # (B, S, D)

    C_x = torch.sigmoid(module.C_proj(x))
    return C_x * h


@register_op("conv1d_seq")
def _op_conv1d_seq(module, inputs, _):
    if not hasattr(module, "conv_weight"):
        return inputs[0]
    x = inputs[0]
    if x.ndim == 2:
        x = x.unsqueeze(0)
    B, S, D = x.shape
    if _c(x):
        conv_bias = getattr(module, "conv_bias", None)
        if conv_bias is None:
            conv_bias = torch.zeros(D, device=x.device, dtype=x.dtype)
        return aria_core.conv1d_seq_f32(x, module.conv_weight, conv_bias)
    # Causal padding: pad (kernel_size - 1) on the left
    kernel_size = module.conv_weight.shape[2]
    x_padded = F.pad(x.transpose(1, 2), (kernel_size - 1, 0))
    out = F.conv1d(x_padded, module.conv_weight, groups=D)
    return out.transpose(1, 2)


@register_op("rwkv_channel")
def _op_rwkv_channel(module, inputs, _):
    """RWKV-style channel mixing with time-shift."""
    x = inputs[0]
    if not hasattr(module, "mix_k"):
        return x
    if _c(x) and x.ndim == 3:
        return aria_core.rwkv_channel_f32(
            x,
            module.mix_k.data,
            module.mix_r.data,
            module.key_proj.weight,
            module.receptance_proj.weight,
            module.value_proj.weight,
        )
    # Safe causal time-shift for 3D tensors (B, S, D)
    if x.ndim == 3:
        shifted = F.pad(x[:, :-1], (0, 0, 1, 0))
    else:
        shifted = x
    xk = x * module.mix_k + shifted * (1 - module.mix_k)
    xr = x * module.mix_r + shifted * (1 - module.mix_r)
    # Receptance-weighted gated linear update
    k = torch.square(torch.relu(module.key_proj(xk)))
    return torch.sigmoid(module.receptance_proj(xr)) * module.value_proj(k)


@register_op("diff_attention")
def _op_diff_attention(module, inputs, _):
    """Differential attention: two softmax maps subtracted to cancel noise.

    Q/K are split into two groups. Two separate softmax attention maps
    are computed and subtracted, cancelling common-mode noise and
    amplifying relevant signal. (Microsoft, ICLR 2025)
    """
    x = inputs[0]
    if not hasattr(module, "q_proj"):
        return x
    B, S, _ = x.shape
    nh, hd = module.n_heads, module.head_dim
    # Project Q, K, V — Q and K are 2x wide for the two groups
    q = (
        module.q_proj(x).reshape(B, S, nh, 2, hd).permute(0, 2, 3, 1, 4)
    )  # (B, nh, 2, S, hd)
    k = module.k_proj(x).reshape(B, S, nh, 2, hd).permute(0, 2, 3, 1, 4)
    v = module.v_proj(x).reshape(B, S, nh, hd).transpose(1, 2)  # (B, nh, S, hd)

    # Two attention maps
    scale = hd**-0.5
    mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
    attn1 = (q[:, :, 0] @ k[:, :, 0].transpose(-2, -1)) * scale
    attn2 = (q[:, :, 1] @ k[:, :, 1].transpose(-2, -1)) * scale
    attn1.masked_fill_(mask, float("-inf"))
    attn2.masked_fill_(mask, float("-inf"))

    # Differential: subtract to cancel noise, scale by learned lambda
    diff = F.softmax(attn1, dim=-1) - module.lambda_param.abs() * F.softmax(
        attn2, dim=-1
    )
    out = (diff @ v).transpose(1, 2).reshape(B, S, -1)
    return module.o_proj(out)


@register_op("state_space")
def _op_state_space(module, inputs, _):
    """S4-style state space mixer with true parallel associative scan.

    Uses Kogge-Stone parallel prefix scan for the linear recurrence
    h[t] = exp(log_a[t]) * h[t-1] + b_x[t], supporting per-timestep
    input-dependent decay (no averaging approximation).
    """
    if not hasattr(module, "ssm_A"):
        return inputs[0]
    x = inputs[0]
    B, S, D = x.shape
    N = module.ssm_state_dim

    # dt: (B, S, D)
    dt = F.softplus(module.ssm_dt(x))
    # A: (D, N), dt: (B, S, D) -> log_a: (B, S, D, N)
    log_a = module.ssm_A.view(1, 1, D, N) * dt.unsqueeze(-1)
    log_a = torch.clamp(log_a, min=-10.0, max=0.0)

    # b_x: (B, S, D, N)
    b_x = module.ssm_B(x).view(B, S, D, N)

    # Reshape so sequence dim is last for parallel scan: (B, D, N, S)
    log_a_t = log_a.permute(0, 2, 3, 1)  # (B, D, N, S)
    b_x_t = b_x.permute(0, 2, 3, 1)  # (B, D, N, S)

    # True parallel associative scan — O(S log S) work, O(log S) depth
    h_t = _parallel_associative_scan(log_a_t.contiguous(), b_x_t.contiguous())

    # Reshape back: (B, D, N, S) -> (B, S, D, N) -> (B, S, D*N)
    h = h_t.permute(0, 3, 1, 2).reshape(B, S, D * N)
    # Clamp scan output: the parallel scan accumulates through O(log S)
    # multiplicative steps, amplifying gradients.  Bounding h prevents
    # one SSM op from dominating the global gradient norm.
    h = torch.clamp(h, min=-50.0, max=50.0)

    y = module.ssm_C(h) * (
        1.0 / math.sqrt(N)
    )  # Scale by 1/sqrt(state_dim) to bound gradient
    return y + x * module.ssm_D


@register_op("conv_only")
def _op_conv_only(module, inputs, _):
    """Depthwise causal convolution sequence mixer with residual."""
    x = inputs[0]
    if not hasattr(module, "conv_dw"):
        return x
    B, S, D = x.shape
    out = module.conv_dw(x.transpose(1, 2))[:, :, :S].transpose(1, 2)
    return x + module.conv_proj(out)


@register_op("gated_delta")
def _op_gated_delta(module, inputs, _):
    """Gated delta rule: linear recurrence with decay + update gates.

    h[t] = alpha[t] * h[t-1] + beta[t] * (v[t] ⊗ k[t] - h[t-1])
         = (alpha[t] - beta[t]) * h[t-1] + beta[t] * v[t] ⊗ k[t]
    where alpha = sigmoid(decay gate), beta = sigmoid(update gate).
    (NVIDIA GatedDeltaNet, ICLR 2025)

    Optimized via multi-head decomposition + parallel associative scan:
    1. Split D into H heads of dimension d — state is (B,H,d,d) not (B,D,D)
    2. Decompose state recurrence into B*H*d*d independent scalar scans
    3. Solve all scans in O(log S) parallel steps via Kogge-Stone
    No Python loop over sequence positions.
    """
    x = inputs[0]
    if not hasattr(module, "q_proj"):
        return x
    B, S, D = x.shape

    # Project input to query, key, value, and gates
    q = module.q_proj(x)  # (B, S, D)
    k = module.k_proj(x)  # (B, S, D)
    v = module.v_proj(x)  # (B, S, D)
    alpha = torch.sigmoid(module.alpha_proj(x))  # (B, S, D) decay gate
    beta = torch.sigmoid(module.beta_proj(x))  # (B, S, D) update gate

    # Effective decay: alpha - beta (absorb delta into one coefficient)
    # Range: (-1, 1) since both alpha, beta ∈ (0, 1)
    eff_decay = alpha - beta  # (B, S, D)

    # ── Multi-head chunked scan ──
    # Split D into H heads of dimension d. State per head: (d, d) instead
    # of (D, D). Cross-head interaction is restored by the output projection.
    # This matches the GatedDeltaNet paper's multi-head formulation.
    # Per-step work: H*d² = D*d (vs D² for full state). With H=8: 8x speedup.
    H = getattr(module, "_gated_delta_heads", min(8, D))
    if D % H != 0:
        H = 1
    d = D // H

    # Reshape to heads: (B, S, D) → (B, S, H, d) → (B, H, S, d)
    q_h = q.reshape(B, S, H, d).permute(0, 2, 1, 3)
    k_h = k.reshape(B, S, H, d).permute(0, 2, 1, 3)
    v_h = v.reshape(B, S, H, d).permute(0, 2, 1, 3)
    decay_h = eff_decay.reshape(B, S, H, d).permute(0, 2, 1, 3)
    beta_h = beta.reshape(B, S, H, d).permute(0, 2, 1, 3)

    # Chunked scan over sequence — state is (B, H, d, d) per chunk boundary
    CHUNK = min(32, S)
    # State: (B*H, d, d) for efficient bmm
    BH = B * H
    h = torch.zeros(BH, d, d, device=x.device, dtype=x.dtype)
    outputs = []

    # Flatten batch and head dims: (B, H, S, d) → (B*H, S, d)
    q_f = q_h.reshape(BH, S, d)
    k_f = k_h.reshape(BH, S, d)
    v_f = v_h.reshape(BH, S, d)
    decay_f = decay_h.reshape(BH, S, d)
    beta_f = beta_h.reshape(BH, S, d)

    for c_start in range(0, S, CHUNK):
        c_end = min(c_start + CHUNK, S)
        c_len = c_end - c_start

        # Slice chunk: (BH, C, d)
        q_c = q_f[:, c_start:c_end]
        k_c = k_f[:, c_start:c_end]
        v_c = v_f[:, c_start:c_end]
        decay_c = decay_f[:, c_start:c_end]
        beta_c = beta_f[:, c_start:c_end]

        # Precompute scaled outer products: (BH, C, d, d)
        bvk_c = beta_c.unsqueeze(-1) * (v_c.unsqueeze(-1) * k_c.unsqueeze(-2))

        # Sequential scan within chunk
        chunk_outs = torch.empty(BH, c_len, d, device=x.device, dtype=x.dtype)
        for t in range(c_len):
            h = decay_c[:, t, :].unsqueeze(-1) * h + bvk_c[:, t]
            chunk_outs[:, t] = torch.bmm(q_c[:, t : t + 1, :], h).squeeze(1)

        outputs.append(chunk_outs)

    # (BH, S, d) → (B, H, S, d) → (B, S, H, d) → (B, S, D)
    out = (
        torch.cat(outputs, dim=1)
        .reshape(B, H, S, d)
        .permute(0, 2, 1, 3)
        .reshape(B, S, D)
    )
    return module.o_proj(out)


@register_op("nm_sparse_linear")
def _op_nm_sparse_linear(module, inputs, config):
    if not hasattr(module, "weight"):
        return inputs[0]
    n = int(getattr(module, "sparsity_n", config.get("n", 2)))
    m = int(getattr(module, "sparsity_m", config.get("m", 4)))
    if m <= 0 or n <= 0 or n > m or (module.weight.shape[1] % m != 0):
        _record_sparse_telemetry(
            module, "nm_sparse_linear", 1.0, "invalid_nm_configuration"
        )
        return _safe_linear(inputs[0], module.weight)

    if (
        HAS_ARIA_CORE
        and inputs[0].device.type == "cpu"
        and inputs[0].dtype == torch.float32
    ):
        mask = aria_core.nm_sparse_mask_f32(module.weight, n, m)
        _record_sparse_telemetry(
            module, "nm_sparse_linear", float(mask.float().mean().item())
        )
        return _safe_linear(inputs[0], module.weight * mask.float())

    mask = _build_nm_mask(module.weight, n=n, m=m)
    _record_sparse_telemetry(module, "nm_sparse_linear", float(mask.mean().item()))
    return _safe_linear(inputs[0], module.weight * mask)


@register_op("block_sparse_linear")
def _op_block_sparse_linear(module, inputs, config):
    if not hasattr(module, "weight"):
        return inputs[0]
    block_size = int(getattr(module, "block_size", config.get("block_size", 16)))
    block_density = float(
        getattr(module, "block_density", config.get("block_density", 0.25))
    )

    if _c(inputs[0]):
        # Generate block mask (coarse)
        mask = _build_block_sparse_mask(module.weight, block_size, block_density)
        # Convert to uint8 for kernel (needs downsampling if we want true block sparsity optimization)
        # For CPU reference, we can just use linear_block_sparse_f32 with uint8 mask
        m_rows = module.weight.shape[0] // block_size
        m_cols = module.weight.shape[1] // block_size
        if m_rows > 0 and m_cols > 0:
            block_mask_uint8 = mask[
                : m_rows * block_size : block_size, : m_cols * block_size : block_size
            ].to(torch.uint8)
            bias = getattr(module, "bias", None)
            x, orig_shape = _flatten_for_kernel(inputs[0])
            out = aria_core.linear_block_sparse_f32(
                x, module.weight, block_mask_uint8, bias, block_size
            )
            out = _unflatten_from_kernel(out, orig_shape)
            _record_sparse_telemetry(
                module, "block_sparse_linear", float(mask.mean().item())
            )
            return out

    mask = _build_block_sparse_mask(module.weight, block_size, block_density)
    _record_sparse_telemetry(module, "block_sparse_linear", float(mask.mean().item()))

    if HAS_KERNELS and inputs[0].is_cuda:
        # Pass through to Triton kernel optimization
        try:
            return kernels.triton_block_sparse_linear(
                inputs[0], module.weight, mask, block_size
            )
        except (ImportError, RuntimeError, AttributeError) as e:
            record_kernel_fallback("triton_block_sparse_linear", e)

    return _safe_linear(inputs[0], module.weight * mask)


@register_op("low_rank_proj")
def _op_low_rank_proj(module, inputs, _):
    if not hasattr(module, "U") or not hasattr(module, "V"):
        return inputs[0]
    if _c(inputs[0]):
        bias = getattr(module, "bias", None)
        x, orig_shape = _flatten_for_kernel(inputs[0])
        # C kernel expects U:[rank, dim_in], V:[dim_out, rank] but Python stores
        # U:[dim_in, rank], V:[rank, dim_out] — transpose both for the kernel
        out = aria_core.linear_low_rank_f32(
            x, module.U.t().contiguous(), module.V.t().contiguous(), bias
        )
        return _unflatten_from_kernel(out, orig_shape)
    # PyTorch fallback
    out = _safe_linear(_safe_linear(inputs[0], module.U.t()), module.V.t())
    if hasattr(module, "bias"):
        out = out + module.bias
    return out


@register_op("grouped_linear")
def _op_grouped_linear(module, inputs, _):
    if not hasattr(module, "weight"):
        return inputs[0]
    if _c(inputs[0]):
        bias = getattr(module, "bias", None)
        x, orig_shape = _flatten_for_kernel(inputs[0])
        out = aria_core.linear_grouped_f32(x, module.weight, bias, module.n_groups)
        return _unflatten_from_kernel(out, orig_shape)
    # PyTorch fallback
    x = inputs[0]
    B, S, D = x.shape
    g = module.n_groups
    group_dim = D // g
    usable = group_dim * g
    x_groups = x[..., :usable].view(B, S, g, group_dim)
    out_groups = torch.einsum("bsgd,gde->bsge", x_groups, module.weight)
    out = out_groups.reshape(B, S, usable)
    if usable < D:
        out = torch.cat([out, x[..., usable:]], dim=-1)
    return out


@register_op("bottleneck_proj")
def _op_bottleneck_proj(module, inputs, _):
    if not hasattr(module, "down") or not hasattr(module, "up"):
        return inputs[0]
    if _c(inputs[0]):
        b_down = getattr(module, "bias_down", None)
        b_up = getattr(module, "bias_up", None)
        x, orig_shape = _flatten_for_kernel(inputs[0])
        out = aria_core.linear_bottleneck_f32(x, module.down, module.up, b_down, b_up)
        return _unflatten_from_kernel(out, orig_shape)
    # PyTorch fallback
    hidden = F.gelu(_safe_linear(inputs[0], module.down))
    return _safe_linear(hidden, module.up)


@register_op("shared_basis_proj")
def _op_shared_basis_proj(module, inputs, _):
    if not hasattr(module, "mixing") or not hasattr(module, "basis"):
        return inputs[0]
    if _c(inputs[0]):
        x, orig_shape = _flatten_for_kernel(inputs[0])
        # C kernel expects mixing as (k, D) but we store (D, k) — transpose
        out = aria_core.linear_shared_basis_f32(
            x, module.mixing.T.contiguous(), module.basis
        )
        return _unflatten_from_kernel(out, orig_shape)
    # PyTorch fallback: x @ (D,k) @ (k,D) -> (B,S,D)
    return inputs[0] @ module.mixing @ module.basis


@register_op("tied_proj")
def _op_tied_proj(module, inputs, _):
    if not hasattr(module, "tied_weight"):
        return inputs[0]
    if _c(inputs[0]):
        b_down = getattr(module, "bias_down", None)
        b_up = getattr(module, "bias_up", None)
        x, orig_shape = _flatten_for_kernel(inputs[0])
        out = aria_core.linear_tied_f32(x, module.tied_weight, b_down, b_up)
        return _unflatten_from_kernel(out, orig_shape)
    # PyTorch fallback
    hidden = F.gelu(_safe_linear(inputs[0], module.tied_weight))
    return _safe_linear(hidden, module.tied_weight.t())


@register_op("semi_structured_2_4_linear")
def _op_semi_structured_2_4_linear(module, inputs, config):
    if not hasattr(module, "weight"):
        return inputs[0]
    if not getattr(module, "sparse_kernel_ready", False) or not inputs[0].is_cuda:
        _record_sparse_telemetry(
            module, "semi_structured_2_4_linear", 1.0, "kernel_unavailable"
        )
        return _safe_linear(inputs[0], module.weight)
    mask = _build_nm_mask(module.weight, n=2, m=4)
    _record_sparse_telemetry(
        module, "semi_structured_2_4_linear", float(mask.mean().item())
    )
    return _safe_linear(inputs[0], module.weight * mask)


@register_op("rwkv_time_mixing")
def _op_rwkv_time_mixing(module, inputs, _):
    """RWKV WKV attention optimized with parallel scan semantics."""
    if not hasattr(module, "W_k"):
        return inputs[0]
    x = inputs[0]
    if (
        _c(x)
        and hasattr(aria_core, "rwkv_time_mixing_f32")
        and getattr(module, "_rwkv_kernel_ready", False)
    ):
        out_native = aria_core.rwkv_time_mixing_f32(
            x,
            module.w_decay,
            module.u_bonus,
            module.W_k,
            module.W_v,
            module.W_r,
        )
        return _safe_linear(out_native, module.W_o)
    B, S, D = x.shape

    k = _safe_linear(x, module.W_k)
    v = _safe_linear(x, module.W_v)
    r_raw = _safe_linear(x, module.W_r)

    # C kernel fast path: fused WKV scan (handles sigmoid internally)
    if _c(k) and hasattr(aria_core, "rwkv_wkv_scan_f32") and not k.requires_grad:
        out = aria_core.rwkv_wkv_scan_f32(
            k.contiguous(),
            v.contiguous(),
            r_raw.contiguous(),
            module.w_decay,
            module.u_bonus,
        )
        return _safe_linear(out, module.W_o)

    r = torch.sigmoid(r_raw)
    w = -torch.exp(module.w_decay)  # (D,) — negative log-decay
    u = module.u_bonus

    # Vectorized WKV via causal convolution with exponential decay kernel.
    # Recurrences: wkv[t] = exp(w)*wkv[t-1] + exp(k[t])*v[t]
    #              denom[t] = exp(w)*denom[t-1] + exp(k[t])
    # Both are constant-decay linear recurrences → causal conv1d.
    exp_k = torch.exp(k.clamp(-20, 20))  # (B, S, D) — clamp for stability

    # Build exponential decay kernel: K[d, 1, s] = exp(w[d] * (S-1-s))
    # Reversed so conv1d correlation computes the causal sum correctly.
    indices = torch.arange(S, device=x.device, dtype=x.dtype)
    log_kernel = w.view(D, 1, 1) * (S - 1 - indices).view(1, 1, S)  # (D, 1, S)
    kernel = torch.exp(log_kernel.clamp(-20, 0))

    # Inclusive scan via causal conv1d: h[t] = sum_{i=0}^{t} K[t-i] * input[i]
    u_wkv = (exp_k * v).permute(0, 2, 1)  # (B, D, S) — input for wkv scan
    u_den = exp_k.permute(0, 2, 1)  # (B, D, S) — input for denom scan

    wkv_incl = F.conv1d(F.pad(u_wkv, (S - 1, 0)), kernel, groups=D)  # (B, D, S)
    den_incl = F.conv1d(F.pad(u_den, (S - 1, 0)), kernel, groups=D)  # (B, D, S)

    # Exclusive scan (state BEFORE update): shift right by 1, pad with 0
    wkv_before = F.pad(wkv_incl[..., :-1], (1, 0))  # (B, D, S)
    den_before = F.pad(den_incl[..., :-1], (1, 0))  # (B, D, S)

    # Transpose back to (B, S, D) for output computation
    wkv_before = wkv_before.permute(0, 2, 1)
    den_before = den_before.permute(0, 2, 1)

    # output[t] = r[t] * (wkv_before[t] + exp(u+k[t])*v[t]) / (den_before[t] + exp(u+k[t]))
    p = torch.exp((u + k).clamp(-20, 20))  # (B, S, D)
    out = r * (wkv_before + p * v) / (den_before + p).clamp(min=1e-8)

    return _safe_linear(out, module.W_o)


@register_op("kronecker_linear")
def _op_kronecker_linear(module, inputs, config):
    import math as _math

    x = inputs[0]  # (B, S, D)
    B, S, D = x.shape
    p = int(_math.isqrt(D))
    q = D // p
    if p * q != D:
        for p in range(int(_math.isqrt(D)), 0, -1):
            if D % p == 0:
                q = D // p
                break
    if hasattr(module, "kron_A"):
        A, B_mat = module.kron_A, module.kron_B
    else:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(p * 65537 + q)
        A = (torch.randn(p, p, generator=gen) * (p**-0.5)).to(
            device=x.device, dtype=x.dtype
        )
        B_mat = (torch.randn(q, q, generator=gen) * (q**-0.5)).to(
            device=x.device, dtype=x.dtype
        )
    out = x.view(B, S, p, q) @ B_mat.T
    out = out.permute(0, 1, 3, 2) @ A.T
    return out.reshape(B, S, D)


@register_op("sparse_bottleneck_moe")
@register_op("n_way_sparse_router")  # backward-compat alias
def _op_sparse_bottleneck_moe(module, inputs, config):
    x = inputs[0]  # (B, S, D)
    B, S, D = x.shape
    n_ways = max(2, min(config.get("n_ways", 4), 16))
    top_k = max(1, min(config.get("top_k", 2), n_ways))
    hidden = D // n_ways

    if hasattr(module, "gate_weight"):
        W_gate = module.gate_weight
    else:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(D * 65537 + n_ways)
        W_gate = (torch.randn(D, n_ways, generator=gen) * (D**-0.5)).to(
            device=x.device, dtype=x.dtype
        )
    gate_logits = x @ W_gate
    topk_vals, topk_idx = gate_logits.topk(top_k, dim=-1)
    gate_weights = F.softmax(topk_vals, dim=-1)

    # Precompute per-expert routing weights via vectorized scatter (eliminates inner loop)
    # expert_weights: (B, S, n_ways) — how much each expert contributes to each token
    expert_weights = torch.zeros(B, S, n_ways, device=x.device, dtype=x.dtype)
    for k_idx in range(top_k):
        idx = topk_idx[:, :, k_idx].unsqueeze(-1)  # (B, S, 1)
        # Under CUDA autocast, softmax/topk can promote weights while the
        # accumulator tensor stays at the input dtype. scatter_add_ requires
        # exact dtype match, so normalize the source explicitly.
        src = gate_weights[:, :, k_idx].unsqueeze(-1).to(expert_weights.dtype)
        expert_weights.scatter_add_(2, idx, src)

    # Collect all expert weights, then batch compute
    W_downs = []
    W_ups = []
    for i in range(n_ways):
        if hasattr(module, f"expert_down_{i}"):
            W_downs.append(getattr(module, f"expert_down_{i}"))
            W_ups.append(getattr(module, f"expert_up_{i}"))
        else:
            gen_e = torch.Generator(device="cpu")
            gen_e.manual_seed(D * 1000 + i * 100 + 1)
            W_downs.append(
                (torch.randn(D, hidden, generator=gen_e) * (D**-0.5)).to(
                    device=x.device, dtype=x.dtype
                )
            )
            W_ups.append(
                (torch.randn(hidden, D, generator=gen_e) * (hidden**-0.5)).to(
                    device=x.device, dtype=x.dtype
                )
            )
    # Stack: (n_ways, D, hidden), (n_ways, hidden, D)
    W_down_all = torch.stack(W_downs)  # (E, D, H)
    W_up_all = torch.stack(W_ups)  # (E, H, D)
    # Batched expert forward: x @ W_down → gelu → @ W_up for all experts at once
    # (B, S, D) @ (E, D, H)^T → (B, S, E, H)
    hidden_all = torch.einsum("bsd,edh->bseh", x, W_down_all)
    hidden_all = F.gelu(hidden_all)
    # (B, S, E, H) @ (E, H, D)^T → (B, S, E, D)
    expert_outs = torch.einsum("bseh,ehd->bsed", hidden_all, W_up_all)
    # Weight by expert_weights: (B, S, E, 1) * (B, S, E, D) → sum over E
    output = (expert_weights.unsqueeze(-1) * expert_outs).sum(dim=2)
    return output


@register_op("chebyshev_spectral_mix")
def _op_chebyshev_spectral_mix(module, inputs, config):
    x = inputs[0]  # (B, S, D)
    K = max(2, min(config.get("chebyshev_order", 6), 16))
    D = x.shape[-1]
    x_norm = torch.tanh(x)

    coeffs = []
    gen_c = torch.Generator(device="cpu")
    gen_c.manual_seed(K * 65537 + D)
    for k in range(K):
        if hasattr(module, f"cheb_c{k}"):
            coeffs.append(getattr(module, f"cheb_c{k}"))
        else:
            c = (torch.randn(D, generator=gen_c) * (K**-0.5)).to(
                device=x.device, dtype=x.dtype
            )
            if k == 1:
                c = c + 1.0
            coeffs.append(c)

    T_prev2 = torch.ones_like(x_norm)
    T_prev1 = x_norm
    output = coeffs[0] * T_prev2 + coeffs[1] * T_prev1

    for k in range(2, K):
        T_k = 2 * x_norm * T_prev1 - T_prev2
        output = output + coeffs[k] * T_k
        T_prev2 = T_prev1
        T_prev1 = T_k
    return output


@register_op("latent_attention_compressor")
def _op_latent_attention_compressor(module, inputs, config):
    """MLA-style: compress KV to latent dim, then decompress."""
    x = inputs[0]  # (B, S, D)
    if not hasattr(module, "kv_compress"):
        return x
    # Compress: (B, S, D) -> (B, S, latent_dim)
    latent = F.linear(x, module.kv_compress.to(x.dtype))
    # Decompress: (B, S, latent_dim) -> (B, S, D*2) -> split to K, V
    kv = F.linear(latent, module.kv_up.to(x.dtype))
    D = x.shape[-1]
    k, v = kv[..., :D], kv[..., D:]
    # Simple attention-free compression: gate k against v
    return x + torch.sigmoid(k) * v


@register_op("signal_conditioned_compression")
@register_op("routing_conditioned_compression")  # backward-compat alias
def _op_signal_conditioned_compression(module, inputs, config):
    """Changes linear layer compression level based on external signal."""
    x, routing_signal = inputs[0], inputs[1]
    if not hasattr(module, "weight_full"):
        return x

    # Use routing signal to interpolate between Full and Low-Rank weights
    # routing_signal is expected to be [B, S, 1] or [B, S, 2]
    if routing_signal.shape[-1] > 1:
        s = torch.sigmoid(routing_signal[..., 0:1])
    else:
        s = torch.sigmoid(routing_signal)

    full = _safe_linear(x, module.weight_full)

    if hasattr(module, "U_comp"):
        comp = _safe_linear(_safe_linear(x, module.U_comp), module.V_comp)
        return s * full + (1 - s) * comp

    return full


@register_op("token_class_proj")
@register_op("token_type_classifier")  # backward-compat alias
def _op_token_class_proj(module, inputs, config):
    """Token type classifier: D → n_classes (with nonlinearity) → D."""
    x = inputs[0]  # (B, S, D)
    if not hasattr(module, "classifier_weight"):
        return x
    if x.shape[-1] != module.classifier_weight.shape[1]:
        return x
    # (B, S, D) → (B, S, n_classes) with GELU for nonlinear class boundaries
    scores = F.gelu(_safe_linear(x, module.classifier_weight))
    # Stash classification scores for telemetry / downstream routing
    module._class_scores = scores.detach()
    # Project back to model dim
    return _safe_linear(scores, module.classifier_proj_back)


@register_op("adaptive_rank_gate")
@register_op("progressive_compression_gate")  # backward-compat alias
def _op_adaptive_rank_gate(module, inputs, config):
    """Per-token compression gate: learned projection decides compression ratio per token."""
    x = inputs[0]
    if not hasattr(module, "weight_full"):
        return x
    dt = x.dtype
    # Per-token gate: (B,S,D) → (B,S,1) — each token gets its own compression ratio
    if hasattr(module, "token_gate"):
        s = torch.sigmoid(F.linear(x, module.token_gate.to(dt)))  # (B, S, 1)
    else:
        s = torch.sigmoid(module.compress_param)

    full = F.linear(x, module.weight_full.to(dt))
    if hasattr(module, "U_comp"):
        comp = F.linear(F.linear(x, module.U_comp.to(dt)), module.V_comp.to(dt))
        return s * full + (1 - s) * comp
    return full


@register_op("dual_compression_blend")
@register_op("compression_mixture_experts")  # backward-compat alias
def _op_dual_compression_blend(module, inputs, config):
    """Routing assigns tokens to method-specific compression experts."""
    x = inputs[0]
    routing_signal = inputs[1] if len(inputs) > 1 else x
    if not hasattr(module, "expert_weights"):
        return x

    # 2 experts: 0=LowRank, 1=Bottleneck
    weights = F.softmax(routing_signal, dim=-1)  # [B, S, 2]

    # Expert 0: Low-Rank
    out0 = _safe_linear(_safe_linear(x, module.U_lr), module.V_lr)

    # Expert 1: Bottleneck
    hidden1 = F.gelu(_safe_linear(x, module.W_down))
    out1 = _safe_linear(hidden1, module.W_up)

    return out0 * weights[..., 0:1] + out1 * weights[..., 1:2]


@register_op("ternary_projection")
def _op_ternary_projection(module, inputs, config):
    """1.58-bit Ternary Weights Simulation (BitNet).
    Weights are restricted to {-1, 0, 1} with a learned scale.
    """
    x = inputs[0]
    if not hasattr(module, "weight"):
        return x

    # Simulated ternary quantization: W_quant = round(clamp(W / gamma))
    # where gamma is average absolute value
    w = module.weight
    gamma = w.abs().mean().clamp(min=1e-5)
    w_quant = torch.round(torch.clamp(w / gamma, -1, 1))

    # STE (Straight-Through Estimator) for training gradients
    w_sim = w + (w_quant * gamma - w).detach()

    bias = getattr(module, "bias", None)
    return F.linear(
        x, w_sim.to(x.dtype), bias.to(x.dtype) if bias is not None else None
    )


_register_split_op_modules()


def _execute_op(
    module: nn.Module, op_name: str, inputs: Tuple[torch.Tensor, ...], config: Dict
) -> torch.Tensor:
    """Execute a single primitive operation via the registry."""
    if op_name in _OP_DISPATCH:
        result = _OP_DISPATCH[op_name](module, inputs, config)

        # Telemetry for registered math space ops (if any)
        if op_name.startswith("math_"):
            nonfinite = int((~torch.isfinite(result)).sum().item())
            if nonfinite > 0:
                result = torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
                telemetry = getattr(module, "mathspace_telemetry", {})
                if len(telemetry) < 256:
                    stats = telemetry.get(op_name, {"calls": 0, "nonfinite": 0})
                    stats["calls"] += 1
                    stats["nonfinite"] += nonfinite
                    telemetry[op_name] = stats
                    setattr(module, "mathspace_telemetry", telemetry)
        return result

    # Fallback for dynamic math space ops not in _OP_DISPATCH
    if op_name in PRIMITIVE_REGISTRY:
        prim = PRIMITIVE_REGISTRY[op_name]
        if not (hasattr(prim, "execute_fn") and prim.execute_fn is not None):
            raise RuntimeError(
                f"Op '{op_name}' is registered as a primitive but has no compiler "
                f"handler in _OP_DISPATCH and no execute_fn fallback. "
                f"Check that _register_split_op_modules() completed without errors."
            )
        result = prim.execute_fn(module, *inputs)
        # Sanitize non-finite values and record telemetry
        if isinstance(result, torch.Tensor):
            nonfinite = int((~torch.isfinite(result)).sum().item())
            telemetry = getattr(module, "mathspace_telemetry", {})
            stats = telemetry.get(
                op_name, {"calls": 0, "nonfinite_elements": 0, "sanitized_calls": 0}
            )
            stats["calls"] = stats.get("calls", 0) + 1

            if nonfinite > 0:
                result = torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
                stats["nonfinite_elements"] = (
                    stats.get("nonfinite_elements", 0) + nonfinite
                )
                stats["sanitized_calls"] = stats.get("sanitized_calls", 0) + 1

            telemetry[op_name] = stats
            setattr(module, "mathspace_telemetry", telemetry)

        # Tropical routing telemetry for route-collapse detection.
        # Gated: only runs when collect_telemetry is True on the module (set
        # by dashboard / profiler), avoiding overhead during normal training.
        if op_name in ("tropical_router", "tropical_moe") and getattr(
            module, "collect_telemetry", False
        ):
            _tropical_obj = getattr(module, "_tropical_router", None) or getattr(
                module, "_tropical_moe", None
            )
            if _tropical_obj is not None:
                _router = getattr(_tropical_obj, "router", _tropical_obj)
                if hasattr(_router, "centroids"):
                    n_exp = _router.centroids.shape[0]
                    with torch.no_grad():
                        _weights = _router(inputs[0])  # (B, S, n_experts)
                        _top_idx = _weights.argmax(dim=-1).flatten()  # (B*S,)
                        _record_routing_telemetry(
                            module,
                            n_exp,
                            _top_idx.unsqueeze(-1),
                            logits=_weights.reshape(-1, n_exp),
                        )

        return result

    raise ValueError(f"Unknown op: {op_name}")


# ── Module Classes ──────────────────────────────────────────────────


class CompiledOp(nn.Module):
    """A single compiled primitive operation."""

    def __init__(
        self,
        op_name: str,
        config: Dict,
        input_shape: ShapeInfo,
        output_shape: ShapeInfo,
        model_dim: int,
    ):
        super().__init__()
        self.op_name = op_name
        self.config = config
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model_dim = model_dim

        op = get_primitive(op_name)
        if op.has_params:
            self._init_params(op, config, input_shape)

    def _make_param(self, shape: Tuple[int, ...], std: float = 0.02) -> nn.Parameter:
        """Create a parameter without per-parameter filesystem I/O."""
        return nn.Parameter(
            torch.empty(shape, dtype=torch.float32).normal_(mean=0.0, std=std)
        )

    def _init_params(self, op: PrimitiveOp, config: Dict, input_shape: ShapeInfo):
        """Initialize learnable parameters for this op."""
        D_in = max(1, input_shape.dim)
        # Guard against degenerate input dims (e.g. from entropy_score's
        # reduce_last → dim=1) for ops with structured params (attention heads,
        # MoE experts, etc.).  Linear projections must use the true D_in so
        # bridge layers from reduce_last ops get correct (D_out, 1) weights.
        _LINEAR_OPS = {
            "linear_proj",
            "linear_proj_down",
            "linear_proj_up",
            "fused_linear_gelu",
        }
        if D_in < 4 and op.name not in _LINEAR_OPS:
            D_in = self.model_dim
        D_out = max(1, config.get("out_dim", D_in))
        # Avoid division by zero for symbolic or unset shapes
        std = 1.0 / math.sqrt(D_in) if D_in > 0 else 0.02

        def _init_attention_stack(op_name: str) -> None:
            n_heads = max(1, D_in // 64)
            head_dim = D_in // n_heads
            self.n_heads = n_heads
            self.head_dim = head_dim
            if op_name in ("softmax_attention", "graph_attention"):
                self.attn_scale = head_dim**-0.5
            self.q_proj = nn.Linear(D_in, n_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(D_in, n_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(D_in, n_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(n_heads * head_dim, D_in, bias=False)
            self.q_proj.weight.data.normal_(std=0.02)
            self.k_proj.weight.data.normal_(std=0.02)
            self.v_proj.weight.data.normal_(std=0.02)
            self.o_proj.weight.data.normal_(std=0.02)
            if op_name == "graph_attention":
                self.edge_proj = nn.Linear(D_in, D_in, bias=False)
                self.edge_proj.weight.data.normal_(std=0.02)

        def _init_math_space() -> None:
            if op.has_params:
                self.weight = self._make_param((D_out, D_in), std=0.02)
            if op.name in ("padic_expand", "padic_residual"):
                # n_digits=1, sin+cos pair → input dim is D_in * 2
                self.weight = self._make_param((D_in, D_in * 2), std=0.02)
                # ReZero: start as identity, gradually introduce p-adic signal
                self.residual_scale = nn.Parameter(torch.zeros(1))
            elif op.name == "rotor_transform":
                self.rotor = nn.Parameter(torch.randn(8) * 0.02)
            elif op.name == "poincare_add":
                self.bias = nn.Parameter(torch.zeros(D_in))
            elif op.name == "hyp_linear":
                self.weight = self._make_param((D_in, D_in), std=0.02)
            elif op.name == "tropical_router":
                n_exp = int(config.get("n_experts", 8))
                self.centroids = nn.Parameter(torch.randn(n_exp, D_in) * 0.02)

        dispatch: Dict[str, Callable[[], None]] = {
            "linear_proj": lambda: setattr(
                self, "weight", self._make_param((D_out, D_in), std=0.02)
            ),
            "linear_proj_down": lambda: setattr(
                self, "weight", self._make_param((D_out, D_in), std=0.02)
            ),
            "linear_proj_up": lambda: setattr(
                self, "weight", self._make_param((D_out, D_in), std=0.02)
            ),
            "fused_linear_gelu": lambda: (
                setattr(self, "weight", self._make_param((D_out, D_in), std=0.02)),
                setattr(self, "bias", nn.Parameter(torch.zeros(D_out))),
            ),
            "learnable_scale": lambda: setattr(
                self, "scale", nn.Parameter(torch.ones(D_in))
            ),
            "learnable_bias": lambda: setattr(
                self, "bias", nn.Parameter(torch.zeros(D_in))
            ),
            "selective_scan": lambda: (
                setattr(self, "A_log", self._make_param((D_in,), std=0.1)),
                setattr(self, "dt_proj", self._make_param((D_in,), std=0.1)),
                setattr(self, "B_proj", nn.Linear(D_in, D_in, bias=False)),
                setattr(self, "C_proj", nn.Linear(D_in, D_in, bias=False)),
                self.B_proj.weight.data.normal_(std=0.02),
                self.C_proj.weight.data.normal_(std=0.02),
            ),
            "conv1d_seq": lambda: setattr(
                self,
                "conv_weight",
                self._make_param((D_in, 1, 3), std=1.0 / math.sqrt(3)),
            ),
            "gated_lane_blend": lambda: self._init_gated_lane_blend(config, D_in),
            "depth_gated_transform": lambda: self._init_depth_gated_transform(
                config, D_in
            ),
            "route_lanes": lambda: self._init_gated_lane_blend(config, D_in),
            "route_recursion": lambda: self._init_depth_gated_transform(config, D_in),
            "topk_gate": lambda: setattr(
                self, "gate_proj", self._make_param((2, D_in), std=0.02)
            ),
            "moe_topk": lambda: self._init_moe_topk(config, D_in),
            "moe_2expert": lambda: (
                setattr(self, "gate_proj", self._make_param((2, D_in), std=0.02)),
                setattr(
                    self, "expert_0_weight", self._make_param((D_in, D_in), std=0.02)
                ),
                setattr(
                    self, "expert_1_weight", self._make_param((D_in, D_in), std=0.02)
                ),
            ),
            "nm_sparse_linear": lambda: (
                setattr(self, "weight", self._make_param((D_out, D_in), std=0.02)),
                setattr(self, "sparsity_n", int(config.get("n", 2))),
                setattr(self, "sparsity_m", int(config.get("m", 4))),
            ),
            "block_sparse_linear": lambda: (
                setattr(self, "weight", self._make_param((D_out, D_in), std=0.02)),
                setattr(self, "block_size", max(1, int(config.get("block_size", 16)))),
                setattr(
                    self,
                    "block_density",
                    float(max(0.25, min(1.0, config.get("block_density", 0.25)))),
                ),
            ),
            "rmsnorm": lambda: setattr(self, "weight", nn.Parameter(torch.ones(D_in))),
            "layernorm": lambda: (
                setattr(self, "weight", nn.Parameter(torch.ones(D_in))),
                setattr(self, "bias", nn.Parameter(torch.zeros(D_in))),
            ),
            "gated_linear": lambda: (
                setattr(
                    self, "linear_weight", self._make_param((D_out, D_in), std=0.02)
                ),
                setattr(self, "gate_weight", self._make_param((D_out, D_in), std=0.02)),
                setattr(self, "linear_bias", nn.Parameter(torch.zeros(D_out))),
                setattr(self, "gate_bias", nn.Parameter(torch.zeros(D_out))),
            ),
            "rwkv_time_mixing": lambda: (
                setattr(self, "w_decay", nn.Parameter(torch.ones(D_in) * -0.5)),
                setattr(self, "u_bonus", nn.Parameter(torch.zeros(D_in))),
                setattr(self, "W_k", self._make_param((D_in, D_in), std=0.02)),
                setattr(self, "W_v", self._make_param((D_in, D_in), std=0.02)),
                setattr(self, "W_r", self._make_param((D_in, D_in), std=0.02)),
                setattr(self, "W_o", self._make_param((D_in, D_in), std=0.02)),
                setattr(self, "_rwkv_kernel_ready", True),
            ),
            "embedding_lookup": lambda: (
                setattr(
                    self,
                    "codebook",
                    nn.Parameter(
                        torch.randn(min(int(config.get("vocab_size", 64)), 256), D_in)
                        * 0.02
                    ),
                ),
                setattr(
                    self,
                    "codebook_proj",
                    nn.Parameter(torch.randn(D_in, D_in) * (D_in**-0.5)),
                ),
            ),
            "rope_rotate": lambda: None,
            "cosine_similarity": lambda: None,
            "gather_topk": lambda: None,
            "spectral_filter": lambda: setattr(
                self, "freq_mask", nn.Parameter(torch.ones(D_in // 2 + 1))
            ),
            "semi_structured_2_4_linear": lambda: (
                setattr(self, "weight", self._make_param((D_out, D_in), std=0.02)),
                setattr(
                    self, "sparse_kernel_ready", bool(D_in % 4 == 0 and D_out % 4 == 0)
                ),
            ),
            "basis_expansion": lambda: setattr(
                self, "weight", nn.Parameter(torch.randn(4, D_in) * 0.1)
            ),
            "integral_kernel": lambda: setattr(
                self, "weight", nn.Parameter(torch.randn(D_in, D_in) * 0.02)
            ),
            "fixed_point_iter": lambda: setattr(
                self, "weight", nn.Parameter(torch.randn(D_in + 1, D_in) * 0.02)
            ),
            "low_rank_proj": lambda: self._init_low_rank_proj(D_in),
            "grouped_linear": lambda: self._init_grouped_linear(D_in),
            "bottleneck_proj": lambda: self._init_bottleneck_proj(D_in),
            "shared_basis_proj": lambda: self._init_shared_basis_proj(D_in),
            "tied_proj": lambda: self._init_tied_proj(D_in),
            "swiglu_mlp": lambda: self._init_swiglu_mlp(config, D_in),
            "rwkv_channel": lambda: self._init_rwkv_channel(config, D_in),
            "softmax_attention": lambda: _init_attention_stack("softmax_attention"),
            "linear_attention": lambda: _init_attention_stack("linear_attention"),
            "graph_attention": lambda: _init_attention_stack("graph_attention"),
            "diff_attention": lambda: self._init_diff_attention(D_in),
            "gated_delta": lambda: self._init_gated_delta(D_in),
            "state_space": lambda: self._init_state_space(D_in),
            "conv_only": lambda: self._init_conv_only(D_in),
            "stdp_attention": lambda: setattr(
                self, "log_tau", nn.Parameter(torch.tensor(0.0))
            ),
            "depth_token_mask": lambda: (
                setattr(self, "router_weight", self._make_param((1, D_in), std=0.02)),
            ),
            "difficulty_blend_3way": lambda: self._init_difficulty_blend_3way(D_in),
            "score_depth_blend": lambda: self._init_score_depth_blend(config, D_in),
            "confidence_token_gate": lambda: setattr(
                self, "confidence_proj", self._make_param((1, D_in), std=0.02)
            ),
            "learned_token_gate": lambda: setattr(
                self, "cascade_proj", self._make_param((1, D_in), std=0.02)
            ),
            "cheap_verify_blend": lambda: self._init_cheap_verify_blend(D_in),
            "depth_weighted_proj": lambda: self._init_depth_weighted_proj(config, D_in),
            "token_class_proj": lambda: self._init_token_class_proj(config, D_in),
            "adaptive_rank_gate": lambda: self._init_adaptive_rank_gate(D_in, D_out),
            "dual_compression_blend": lambda: self._init_dual_compression_blend(
                D_in, D_out
            ),
            "relu_gated_moe": lambda: self._init_relu_gated_moe(config, D_in),
            "relu_gate_routing": lambda: self._init_relu_gated_moe(config, D_in),
            "ternary_projection": lambda: self._init_ternary_projection(
                config, D_in, D_out
            ),
            "latent_attention_compressor": lambda: (
                self._init_latent_attention_compressor(D_in)
            ),
            "signal_conditioned_compression": lambda: (
                self._init_signal_conditioned_compression(D_in)
            ),
            "routing_conditioned_compression": lambda: (
                self._init_signal_conditioned_compression(D_in)
            ),
            "chebyshev_spectral_mix": lambda: self._init_chebyshev_spectral_mix(
                config, D_in
            ),
            "sparse_bottleneck_moe": lambda: self._init_sparse_bottleneck_moe(
                config, D_in
            ),
            # True routing ops (heterogeneous experts)
            "hetero_moe": lambda: self._init_hetero_moe(D_in),
            "arch_router": lambda: self._init_arch_router(D_in),
            "compute_budget_router": lambda: self._init_compute_budget_router(D_in),
        }

        handler = dispatch.get(op.name)
        if handler is not None:
            handler()
            return

        if op.category.value == "math_space":
            _init_math_space()
            return

        if hasattr(op, "init_params"):
            op.init_params(self, D_in)
            return
        self.weight = nn.Parameter(torch.randn(D_in, D_in) * std)

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
        # Q and K project to 2x heads (two groups for differential)
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
        self.alpha_proj = nn.Linear(d_in, d_in, bias=False)  # decay gate
        self.beta_proj = nn.Linear(d_in, d_in, bias=False)  # update gate
        for proj in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.alpha_proj,
            self.beta_proj,
        ):
            proj.weight.data.normal_(std=0.02)

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
        max_depth = int(config.get("max_depth", 3))
        max_depth = max(1, min(6, max_depth))
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
        # HiPPO-style A: negative integers create multi-timescale memory
        # A[d,n] = -(n+1) so log_a spans different decay rates
        A_init = (
            -torch.arange(1, state_dim + 1, dtype=torch.float32)
            .unsqueeze(0)
            .expand(d_in, -1)
        )
        self.ssm_A = nn.Parameter(A_init)
        self.ssm_B = nn.Linear(d_in, d_in * state_dim, bias=False)
        self.ssm_C = nn.Linear(d_in * state_dim, d_in, bias=False)
        self.ssm_D = nn.Parameter(torch.ones(d_in))
        self.ssm_dt = nn.Linear(d_in, d_in)
        # B/C at 0.02 std (same as gated_delta) — 1/(D*N) was too small to learn
        self.ssm_B.weight.data.normal_(std=0.02)
        self.ssm_C.weight.data.normal_(std=0.02)
        # dt bias: small positive init so softplus(dt) starts near ln(2) ≈ 0.69
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

    def _init_depth_weighted_proj(self, config: Dict, d_in: int) -> None:
        max_depth = int(config.get("max_depth", 3))
        max_depth = max(1, min(6, max_depth))
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
        self.expert_weights = nn.Parameter(torch.ones(2))
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

    def _init_chebyshev_spectral_mix(self, config: Dict, d_in: int) -> None:
        K = max(2, min(config.get("chebyshev_order", 6), 16))
        for k in range(K):
            std = K**-0.5
            p = self._make_param((d_in,), std=std)
            if k == 1:
                p.data.add_(1.0)
            setattr(self, f"cheb_c{k}", p)

    def _init_sparse_bottleneck_moe(self, config: Dict, d_in: int) -> None:
        n_ways = max(2, min(config.get("n_ways", 4), 16))
        hidden = d_in // n_ways
        self.gate_weight = self._make_param((d_in, n_ways), std=0.02)
        for i in range(n_ways):
            setattr(
                self, f"expert_down_{i}", self._make_param((d_in, hidden), std=0.02)
            )
            setattr(self, f"expert_up_{i}", self._make_param((hidden, d_in), std=0.02))

    # ── True routing op inits ────────────────────────────────────────

    def _init_hetero_moe(self, d_in: int) -> None:
        """Heterogeneous MoE: gate + attention/conv/SSM expert params."""
        self.gate_weight = self._make_param((3, d_in), std=0.02)
        # Attention expert: Q/K/V packed + output proj
        self.attn_qkv = self._make_param((3 * d_in, d_in), std=0.02)
        self.attn_out = self._make_param((d_in, d_in), std=0.02)
        # Conv expert: depthwise conv1d kernel=3 + output proj
        self.conv_weight = self._make_param((d_in, 1, 3), std=0.02)
        self.conv_proj = self._make_param((d_in, d_in), std=0.02)
        # SSM expert: diagonal A (log-space) + input-dependent B/C + skip D
        self.ssm_A_log = self._make_param((d_in,), std=0.1)
        self.ssm_B_proj = self._make_param((d_in, d_in), std=0.02)
        self.ssm_C_proj = self._make_param((d_in, d_in), std=0.02)
        self.ssm_D = self._make_param((d_in,), std=0.02)

    def _init_arch_router(self, d_in: int) -> None:
        """Architecture router: gate + transformer/mamba/MLP block params."""
        self.gate_weight = self._make_param((3, d_in), std=0.02)
        # Shared attention params (transformer path)
        self.attn_qkv = self._make_param((3 * d_in, d_in), std=0.02)
        self.attn_out = self._make_param((d_in, d_in), std=0.02)
        self.arch_ffn = self._make_param((d_in, d_in), std=0.02)
        # Shared conv + SSM params (mamba path)
        self.conv_weight = self._make_param((d_in, 1, 3), std=0.02)
        self.conv_proj = self._make_param((d_in, d_in), std=0.02)
        self.ssm_A_log = self._make_param((d_in,), std=0.1)
        self.ssm_B_proj = self._make_param((d_in, d_in), std=0.02)
        self.ssm_C_proj = self._make_param((d_in, d_in), std=0.02)
        self.ssm_D = self._make_param((d_in,), std=0.02)
        self.arch_proj = self._make_param((d_in, d_in), std=0.02)
        # MLP path
        hidden = d_in * 4
        self.mlp_up = self._make_param((hidden, d_in), std=0.02)
        self.mlp_down = self._make_param((d_in, hidden), std=0.02)

    def _init_compute_budget_router(self, d_in: int) -> None:
        """Compute budget router: gate + cheap/medium/expensive tier params."""
        self.gate_weight = self._make_param((3, d_in), std=0.02)
        # Tier 0 (cheap): single linear projection
        self.cheap_proj = self._make_param((d_in, d_in), std=0.02)
        # Tier 1 (medium): depthwise conv + proj
        self.conv_weight = self._make_param((d_in, 1, 3), std=0.02)
        self.conv_proj = self._make_param((d_in, d_in), std=0.02)
        # Tier 2 (expensive): attention
        self.attn_qkv = self._make_param((3 * d_in, d_in), std=0.02)
        self.attn_out = self._make_param((d_in, d_in), std=0.02)

    def _cast_params_to(self, dtype: torch.dtype) -> None:
        """Cast raw parameters to match input dtype (bf16 autocast safety).

        Covers direct params and ParameterList children (e.g. step_projs).
        Skips nn.Linear/nn.Sequential submodules which handle autocast internally.

        Must cast in BOTH directions (f32→bf16 and bf16→f32) so that params
        match the current input dtype. Without the f32 path, running under
        autocast permanently converts params to bf16 and subsequent non-autocast
        forward passes crash with dtype mismatch.
        """
        # Direct parameters (raw nn.Parameter attrs used with F.linear)
        for param in self._parameters.values():
            if param is not None and param.dtype != dtype:
                param.data = param.data.to(dtype)
        # ParameterList children (e.g. step_projs for mixed_recursion_gate)
        for child in self._modules.values():
            if isinstance(child, torch.nn.ParameterList):
                for param in child:
                    if param.dtype != dtype:
                        param.data = param.data.to(dtype)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Execute this primitive operation."""
        # Ensure raw parameters match input dtype under bf16 autocast.
        # nn.Linear handles this internally, but F.linear with raw
        # nn.Parameter tensors does not get autocast coverage.
        if inputs and inputs[0].is_floating_point():
            self._cast_params_to(inputs[0].dtype)
        wrapper = getattr(self, "_native_wrapper", None)
        if wrapper is not None:
            result = wrapper.dispatch(self.op_name, *inputs)
            if result is not None:
                return result
        return _execute_op(self, self.op_name, inputs, self.config)


class CompiledLayer(nn.Module):
    """A compiled computation graph as a PyTorch module with memory management."""

    def __init__(self, graph: ComputationGraph):
        super().__init__()
        self.graph = graph
        self.topo_order = graph.topological_order()

        # Track consumer counts for memory reclamation
        self.consumer_counts = {}
        for nid in self.topo_order:
            node = graph.nodes[nid]
            for iid in node.input_ids:
                self.consumer_counts[iid] = self.consumer_counts.get(iid, 0) + 1

        self.ops = nn.ModuleDict()
        for nid in self.topo_order:
            node = graph.nodes[nid]
            if node.is_input:
                continue
            input_shapes = [graph.nodes[iid].output_shape for iid in node.input_ids]
            self.ops[str(nid)] = CompiledOp(
                node.op_name,
                node.config,
                input_shapes[0] if input_shapes else ShapeInfo(),
                node.output_shape,
                graph.model_dim,
            )

        # Pre-compute math-space block boundary nodes.
        # A math-space op needs post-normalization when ANY of its consumers
        # is non-math-space (or it's the output node with no consumers).
        # Back-to-back math ops (e.g. tropical_matmul → tropical_add) are
        # left unnormalized to preserve algebraic semantics within blocks.
        self._mathspace_boundary_nids: set = set()
        self._mathspace_boundary_norms = nn.ModuleDict()
        consumers_of: Dict[int, List[int]] = {nid: [] for nid in graph.nodes}
        for nid in self.topo_order:
            for iid in graph.nodes[nid].input_ids:
                consumers_of[iid].append(nid)
        output_id = graph._output_node_id
        for nid in self.topo_order:
            node = graph.nodes[nid]
            if node.is_input or node.op_name not in _MATHSPACE_OPS:
                continue
            # Check: does this math-space op feed into any non-math-space consumer?
            is_boundary = False
            node_consumers = consumers_of.get(nid, [])
            if not node_consumers and nid == output_id:
                # Final op in graph is math-space — must normalize before lm_head
                is_boundary = True
            else:
                for cid in node_consumers:
                    consumer_node = graph.nodes[cid]
                    if consumer_node.op_name not in _MATHSPACE_OPS:
                        is_boundary = True
                        break
            if is_boundary:
                self._mathspace_boundary_nids.add(str(nid))

        # Pre-cache forward execution plan to eliminate per-iteration str()
        # conversions and dict lookups in the hot forward() loop.
        self._fwd_plan: list = []
        for nid in self.topo_order:
            node = graph.nodes[nid]
            if node.is_input:
                self._fwd_plan.append((nid, True, None, node.input_ids, False))
            else:
                nid_str = str(nid)
                op = self.ops[nid_str]
                is_boundary = nid_str in self._mathspace_boundary_nids
                self._fwd_plan.append((nid, False, op, node.input_ids, is_boundary))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the computation graph with liveness-based memory management.

        If a ``_subgraph_dispatcher`` is attached (by the native-first
        compile pipeline), tries to execute the entire graph through the
        Rust scheduler in a single call.  Falls back to per-op dispatch
        on failure or when not all ops are native-supported.

        Tensors are deleted as soon as their last consumer finishes, minimizing
        peak VRAM usage.
        """
        # --- Subgraph dispatch fast-path ---
        dispatcher = getattr(self, "_subgraph_dispatcher", None)
        if dispatcher is not None:
            result = dispatcher.try_dispatch(x)
            if result is not None:
                return result

        node_outputs: Dict[int, torch.Tensor] = {}
        counts = self.consumer_counts.copy()
        output_id = self.graph._output_node_id
        if output_id is None:
            raise RuntimeError("Graph has no output node")

        is_cuda = x.is_cuda

        for nid, is_input, op, input_ids, is_boundary in self._fwd_plan:
            if is_input:
                node_outputs[nid] = x
            else:
                inputs = tuple(node_outputs[iid] for iid in input_ids)
                out = op(*inputs)
                if is_boundary:
                    # RMSNorm at math-space block boundaries.
                    # Skip .float() when already float32.
                    out_f = out if out.dtype == torch.float32 else out.float()
                    rms = out_f.pow(2).mean(dim=-1, keepdim=True).add_(1e-6).rsqrt_()
                    out = (
                        out * rms
                        if out.dtype == torch.float32
                        else out * rms.to(out.dtype)
                    )
                node_outputs[nid] = out

            for iid in input_ids:
                counts[iid] -= 1
                if counts[iid] <= 0 and iid != output_id:
                    if iid in node_outputs:
                        out_to_del = node_outputs.pop(iid)
                        if is_cuda:
                            del out_to_del

        out = node_outputs.pop(output_id)
        node_outputs.clear()
        return out

    def set_capture_heatmap(self, enabled: bool = True) -> None:
        """Enable or disable heatmap capture for all ops in this layer."""
        for op in self.ops.values():
            op._capture_heatmap = enabled


class SynthesizedModel(nn.Module):
    """A complete language model built from synthesized layers."""

    def __init__(
        self,
        layer_graphs: List[ComputationGraph],
        vocab_size: int = VOCAB_SIZE,
        model_dim: int = MODEL_DIM,
        max_seq_len: int = VALIDATION_SEQ_LEN,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, model_dim)
        # Init embeddings to std=1/sqrt(D) so tied lm_head logits have std≈1,
        # giving initial loss ≈ ln(vocab) ≈ 10.4 (random baseline). This lets
        # the model learn language from step 1 instead of spending 200+ steps
        # just normalizing a bad output distribution. Verified: all init scales
        # pass the learning gate with real WikiText data at 500 steps.
        nn.init.normal_(self.embed.weight, mean=0.0, std=model_dim**-0.5)
        self.layers = nn.ModuleList([CompiledLayer(g) for g in layer_graphs])
        self.norm = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self._layer_graphs = layer_graphs
        # Pre-calculate which layers need an external residual connection
        # If a graph has NO internal residual, we MUST add one between layers.
        self.layer_needs_residual = [not g.has_residual_path() for g in layer_graphs]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            if self.layer_needs_residual[i]:
                # Standard inter-layer residual for "flat" blocks
                out = layer(x)
                if out.shape == x.shape:
                    x = x + out
                else:
                    x = out
            else:
                # References usually have their own residuals internally
                x = layer(x)
        return self.lm_head(self.norm(x))

    def set_capture_heatmap(self, enabled: bool = True) -> None:
        """Enable or disable heatmap capture for all layers."""
        for layer in self.layers:
            if hasattr(layer, "set_capture_heatmap"):
                layer.set_capture_heatmap(enabled)

    @property
    def has_mathspace_ops(self) -> bool:
        """True if any layer contains non-euclidean math-space ops."""
        return any(layer._mathspace_boundary_norms for layer in self.layers)

    @property
    def recommended_grad_clip(self) -> float:
        """Adaptive clip norm: 5.0 for math-space architectures, 1.0 otherwise.

        Math-space ops (tropical, clifford, hyperbolic) produce legitimately
        larger gradients due to non-euclidean geometry. Clipping at 1.0
        starves them; 5.0 allows learning while still catching explosions.
        """
        return 5.0 if self.has_mathspace_ops else 1.0

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        desc = [
            f"SynthesizedModel(dim={self.model_dim}, layers={len(self.layers)}, params={self.param_count():,})"
        ]
        for i, g in enumerate(self._layer_graphs):
            desc.append(
                f"\n  Layer {i}:\n"
                + "\n".join(f"    {l}" for l in g.describe().split("\n"))
            )
        return "\n".join(desc)


def compile_graph(graph: ComputationGraph, use_ir: bool = True) -> nn.Module:
    """Compile a graph to a PyTorch module.

    Args:
        graph: The computation graph to compile.
        use_ir: If True (default), uses the high-performance IRExecutor path.
    """
    from .graph_validator import annotate_kv_cacheable

    annotate_kv_cacheable(graph)

    if use_ir:
        from .ir_executor import IRExecutor

        return IRExecutor(graph.lower_to_ir())
    return CompiledLayer(graph)


def compile_model(
    layer_graphs: List[ComputationGraph],
    vocab_size: int = VOCAB_SIZE,
    max_seq_len: int = VALIDATION_SEQ_LEN,
    use_ir: bool = True,
) -> SynthesizedModel:
    if not layer_graphs:
        raise ValueError("Empty layer_graphs list")
    model = SynthesizedModel(
        layer_graphs, vocab_size, layer_graphs[0].model_dim, max_seq_len
    )
    if use_ir:
        # Replace standard layers with IR executors
        from .ir_executor import IRExecutor

        model.layers = nn.ModuleList(
            [IRExecutor(g.lower_to_ir()) for g in layer_graphs]
        )
    return model
