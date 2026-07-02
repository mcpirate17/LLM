from __future__ import annotations

import math
from typing import Callable, Dict

import torch
import torch.nn.functional as F

from .compiler_op_utils import _safe_linear, _c
from research.env import aria_core


def _scan_broadcast_view(a_slice: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    extra_dims = target.ndim - a_slice.ndim
    if extra_dims <= 0:
        return a_slice
    return a_slice.reshape(*a_slice.shape[:-1], *((1,) * extra_dims), a_slice.shape[-1])


def _kogge_stone_scan_inplace_(a: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    seq_len = a.shape[-1]
    stride = 1
    while stride < seq_len:
        next_a = torch.empty_like(a)
        next_h = torch.empty_like(h)
        next_a[..., :stride] = a[..., :stride]
        next_h[..., :stride] = h[..., :stride]
        next_h[..., stride:] = h[..., stride:] + (
            _scan_broadcast_view(a[..., stride:], h[..., stride:]) * h[..., :-stride]
        )
        next_a[..., stride:] = a[..., stride:] * a[..., :-stride]
        next_a[..., stride:].clamp_(max=1.0)
        a = next_a
        h = next_h
        stride *= 2
    return h


def _parallel_associative_scan(log_a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    seq_len = log_a.shape[-1]
    if seq_len <= 1:
        return b
    return _kogge_stone_scan_inplace_(torch.exp(log_a), b)


def _op_selective_scan(module, inputs, _):
    if not hasattr(module, "A_log"):
        return inputs[0]
    x = inputs[0]
    B, S, D = x.shape
    A = -torch.exp(module.A_log.clamp(-10, 10))
    dt = F.softplus(module.dt_proj[:D])
    log_a = (A * dt).clamp(-10, -0.05)
    a = torch.exp(log_a)
    u = torch.sigmoid(module.B_proj(x)) * x
    u_prev = F.pad(u, (0, 0, 1, 0))[:, :S, :]
    u_trap = 0.5 * (u + a.unsqueeze(0).unsqueeze(0) * u_prev)
    indices = torch.arange(S, device=x.device, dtype=x.dtype)
    log_kernel = log_a.view(D, 1, 1) * (S - 1 - indices).view(1, 1, S)
    kernel = torch.exp(log_kernel)
    u_swapped = u_trap.permute(0, 2, 1)
    h_swapped = F.conv1d(F.pad(u_swapped, (S - 1, 0)), kernel, groups=D)
    h = h_swapped.permute(0, 2, 1)
    C_x = torch.sigmoid(module.C_proj(x))
    return C_x * h


def _op_conv1d_seq(module, inputs, _):
    if not hasattr(module, "conv_weight"):
        return inputs[0]
    x = inputs[0]
    if x.ndim == 2:
        x = x.unsqueeze(0)
    _, _, D = x.shape
    if _c(x):
        conv_bias = getattr(module, "conv_bias", None)
        if conv_bias is None:
            conv_bias = torch.zeros(D, device=x.device, dtype=x.dtype)
        return aria_core.conv1d_seq_f32(x, module.conv_weight, conv_bias)
    kernel_size = module.conv_weight.shape[2]
    x_padded = F.pad(x.transpose(1, 2), (kernel_size - 1, 0))
    conv_weight = module.conv_weight
    if conv_weight.dtype != x.dtype:
        conv_weight = conv_weight.to(x.dtype)
    out = F.conv1d(x_padded, conv_weight, groups=D)
    return out.transpose(1, 2)


def _op_rwkv_channel(module, inputs, _):
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
    shifted = x.clone()
    if x.ndim == 3 and x.shape[1] > 1:
        shifted[:, 1:] = x[:, :-1]
    xk = x * module.mix_k + shifted * (1 - module.mix_k)
    xr = x * module.mix_r + shifted * (1 - module.mix_r)
    k = torch.square(
        torch.relu(_safe_linear(xk, module.key_proj.weight, module.key_proj.bias))
    )
    receptance = torch.sigmoid(
        _safe_linear(xr, module.receptance_proj.weight, module.receptance_proj.bias)
    )
    value = _safe_linear(k, module.value_proj.weight, module.value_proj.bias)
    return receptance * value


def _op_diff_attention(module, inputs, _):
    x = inputs[0]
    if not hasattr(module, "q_proj"):
        return x
    B, S, _ = x.shape
    nh, hd = module.n_heads, module.head_dim
    q = (
        _safe_linear(x, module.q_proj.weight, module.q_proj.bias)
        .reshape(B, S, nh, 2, hd)
        .permute(0, 2, 3, 1, 4)
    )
    k = (
        _safe_linear(x, module.k_proj.weight, module.k_proj.bias)
        .reshape(B, S, nh, 2, hd)
        .permute(0, 2, 3, 1, 4)
    )
    v = (
        _safe_linear(x, module.v_proj.weight, module.v_proj.bias)
        .reshape(B, S, nh, hd)
        .transpose(1, 2)
    )
    out1 = F.scaled_dot_product_attention(
        q[:, :, 0],
        k[:, :, 0],
        v,
        dropout_p=0.0,
        is_causal=True,
        scale=hd**-0.5,
    )
    out2 = F.scaled_dot_product_attention(
        q[:, :, 1],
        k[:, :, 1],
        v,
        dropout_p=0.0,
        is_causal=True,
        scale=hd**-0.5,
    )
    out = (out1 - module.lambda_param.abs() * out2).transpose(1, 2).reshape(B, S, -1)
    return _safe_linear(out, module.o_proj.weight, module.o_proj.bias)


def _op_state_space(module, inputs, _):
    if not hasattr(module, "ssm_A"):
        return inputs[0]
    x = inputs[0]
    B, S, D = x.shape
    N = module.ssm_state_dim
    dt = F.softplus(module.ssm_dt(x))
    log_a = module.ssm_A.view(1, 1, D, N) * dt.unsqueeze(-1)
    log_a = torch.clamp(log_a, min=-10.0, max=0.0)
    b_x = module.ssm_B(x).view(B, S, D, N)
    log_a_t = log_a.permute(0, 2, 3, 1)
    b_x_t = b_x.permute(0, 2, 3, 1)
    h_t = _parallel_associative_scan(log_a_t.contiguous(), b_x_t.contiguous())
    h = h_t.permute(0, 3, 1, 2).reshape(B, S, D * N)
    h = torch.clamp(h, min=-50.0, max=50.0)
    y = module.ssm_C(h) * (1.0 / math.sqrt(N))
    return y + x * module.ssm_D


def _op_conv_only(module, inputs, _):
    x = inputs[0]
    if not hasattr(module, "conv_dw"):
        return x
    _, S, _ = x.shape
    out = module.conv_dw(x.transpose(1, 2))[:, :, :S].transpose(1, 2)
    return x + module.conv_proj(out)


# Forget-gate-bias init for the gated_delta retention gate. Must match the
# aria_core C++ kernel constant kGatedDeltaDecayBias (gated_delta_compiled.cpp /
# gated_delta_backward_compiled.cpp) — the native path and this torch reference
# are pinned equal by test_native_gated_delta_backward_matches_python.
_GATED_DELTA_DECAY_BIAS = 2.5


def _op_gated_delta(module, inputs, _):
    x = inputs[0]
    if not hasattr(module, "q_proj"):
        return x
    B, S, D = x.shape
    q = _safe_linear(x, module.q_proj.weight, module.q_proj.bias)
    k = _safe_linear(x, module.k_proj.weight, module.k_proj.bias)
    v = _safe_linear(x, module.v_proj.weight, module.v_proj.bias)
    # decay = alpha is the state-retention gate, shifted by a positive forget-gate
    # bias so the recurrent state is *kept* at init (≈0.92). The old
    # ``alpha - beta`` centred decay at 0 (both gates ≈0.5) and wiped the state →
    # the mamba2/gated_delta baseline scored 0.0 everywhere. beta stays the delta
    # write strength below.
    alpha = torch.sigmoid(module.alpha_proj(x) + _GATED_DELTA_DECAY_BIAS)
    beta = torch.sigmoid(module.beta_proj(x))
    eff_decay = alpha
    H = getattr(module, "_gated_delta_heads", min(8, D))
    if D % H != 0:
        H = 1
    d = D // H
    q_h = q.reshape(B, S, H, d).permute(0, 2, 1, 3)
    k_h = k.reshape(B, S, H, d).permute(0, 2, 1, 3)
    v_h = v.reshape(B, S, H, d).permute(0, 2, 1, 3)
    decay_h = eff_decay.reshape(B, S, H, d).permute(0, 2, 1, 3)
    beta_h = beta.reshape(B, S, H, d).permute(0, 2, 1, 3)
    CHUNK = min(32, S)
    BH = B * H
    h = torch.zeros(BH, d, d, device=x.device, dtype=x.dtype)
    out_f = torch.empty(BH, S, d, device=x.device, dtype=x.dtype)
    q_f = q_h.reshape(BH, S, d)
    k_f = k_h.reshape(BH, S, d)
    v_f = v_h.reshape(BH, S, d)
    decay_f = decay_h.reshape(BH, S, d)
    beta_f = beta_h.reshape(BH, S, d)
    for c_start in range(0, S, CHUNK):
        c_end = min(c_start + CHUNK, S)
        c_len = c_end - c_start
        q_c = q_f[:, c_start:c_end]
        k_c = k_f[:, c_start:c_end]
        v_c = v_f[:, c_start:c_end]
        decay_c = decay_f[:, c_start:c_end]
        beta_c = beta_f[:, c_start:c_end]
        bvk_c = beta_c.unsqueeze(-1) * (v_c.unsqueeze(-1) * k_c.unsqueeze(-2))
        bvk_c[:, 0].add_(decay_c[:, 0, :].unsqueeze(-1) * h)
        a_flat = decay_c.permute(0, 2, 1).reshape(BH * d, c_len)
        b_flat = bvk_c.permute(0, 2, 3, 1).reshape(BH * d, d, c_len)
        scan_h = _kogge_stone_scan_inplace_(a_flat.clamp(min=1e-8), b_flat)
        h_all = scan_h.reshape(BH, d, d, c_len).permute(0, 3, 1, 2)
        h = h_all[:, -1]
        out_f[:, c_start:c_end] = torch.einsum("bcd,bcde->bce", q_c, h_all)
    out = out_f.reshape(B, H, S, d).permute(0, 2, 1, 3).reshape(B, S, D)
    return _safe_linear(out, module.o_proj.weight, module.o_proj.bias)


def _op_dplr_gated_delta(module, inputs, _):
    """Diagonal-plus-low-rank gated delta recurrence."""
    x = inputs[0]
    if not hasattr(module, "diag_proj"):
        return x
    B, S, D = x.shape
    q = _safe_linear(x, module.q_proj.weight, module.q_proj.bias)
    k = _safe_linear(x, module.k_proj.weight, module.k_proj.bias)
    v = _safe_linear(x, module.v_proj.weight, module.v_proj.bias)
    diag = torch.sigmoid(
        _safe_linear(x, module.diag_proj.weight, module.diag_proj.bias)
    )
    beta = torch.sigmoid(
        _safe_linear(x, module.beta_proj.weight, module.beta_proj.bias)
    )
    low_rank = _safe_linear(
        torch.tanh(_safe_linear(x, module.lr_in.weight, module.lr_in.bias)),
        module.lr_out.weight,
        module.lr_out.bias,
    )

    scale = D**-0.5
    state = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)
    out = torch.empty(B, S, D, device=x.device, dtype=x.dtype)
    v_eff = v + low_rank
    chunk = min(32, S)
    for c_start in range(0, S, chunk):
        c_end = min(c_start + chunk, S)
        c_len = c_end - c_start
        q_c = q[:, c_start:c_end]
        k_c = k[:, c_start:c_end]
        v_c = v_eff[:, c_start:c_end]
        diag_c = diag[:, c_start:c_end]
        beta_c = beta[:, c_start:c_end]
        write_c = beta_c.unsqueeze(-1) * (v_c.unsqueeze(-1) * k_c.unsqueeze(-2))
        write_c[:, 0].add_(diag_c[:, 0, :].unsqueeze(-1) * state)
        a_flat = diag_c.permute(0, 2, 1).reshape(B * D, c_len)
        b_flat = write_c.permute(0, 2, 3, 1).reshape(B * D, D, c_len)
        scan_h = _kogge_stone_scan_inplace_(a_flat, b_flat)
        h_all = scan_h.reshape(B, D, D, c_len).permute(0, 3, 1, 2)
        state = h_all[:, -1]
        out[:, c_start:c_end] = (
            torch.einsum("bcd,bcde->bce", q_c, h_all) * scale
        )
    return _safe_linear(out, module.o_proj.weight, module.o_proj.bias)


def _op_rwkv_time_mixing(module, inputs, _):
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
    if _c(k) and hasattr(aria_core, "rwkv_wkv_scan_f32"):
        out = aria_core.rwkv_wkv_scan_f32(
            k.contiguous(),
            v.contiguous(),
            r_raw.contiguous(),
            module.w_decay,
            module.u_bonus,
        )
        return _safe_linear(out, module.W_o)
    r = torch.sigmoid(r_raw)
    w = -torch.exp(module.w_decay)
    u = module.u_bonus
    exp_k = torch.exp(k.clamp(-20, 20))
    indices = torch.arange(S, device=x.device, dtype=x.dtype)
    log_kernel = w.view(D, 1, 1) * (S - 1 - indices).view(1, 1, S)
    kernel = torch.exp(log_kernel.clamp(-20, 0))
    if kernel.dtype != x.dtype:
        kernel = kernel.to(x.dtype)
    u_wkv = (exp_k * v).permute(0, 2, 1)
    u_den = exp_k.permute(0, 2, 1)
    wkv_incl = F.conv1d(F.pad(u_wkv, (S - 1, 0)), kernel, groups=D)
    den_incl = F.conv1d(F.pad(u_den, (S - 1, 0)), kernel, groups=D)
    wkv_before = F.pad(wkv_incl[..., :-1], (1, 0)).permute(0, 2, 1)
    den_before = F.pad(den_incl[..., :-1], (1, 0)).permute(0, 2, 1)
    p = torch.exp((u + k).clamp(-20, 20))
    out = r * (wkv_before + p * v) / (den_before + p).clamp(min=1e-8)
    return _safe_linear(out, module.W_o)


def _op_difficulty_routed_attention(module, inputs, _):
    """Route only hard tokens through attention; easy tokens stay on the cheap path."""
    x = inputs[0]
    if not hasattr(module, "difficulty_proj"):
        return x
    B, S, D = x.shape
    diff_scores = module.difficulty_proj(x).squeeze(-1)
    diff_gate = torch.sigmoid(diff_scores)
    easy_out = module.easy_proj(x)
    hard_mask = diff_gate >= 0.5
    if not hard_mask.any():
        return easy_out

    H = min(8, D)
    if D % H != 0:
        H = 1
    d = D // H
    output = easy_out.clone()

    for b in range(B):
        hard_idx = torch.nonzero(hard_mask[b], as_tuple=False).squeeze(-1)
        n_hard = int(hard_idx.numel())
        if n_hard == 0:
            continue
        if n_hard == S:
            x_h = x[b : b + 1]
        else:
            x_h = x[b : b + 1, hard_idx]
        q = module.q_proj(x_h).reshape(1, n_hard, H, d).transpose(1, 2)
        k = module.k_proj(x_h).reshape(1, n_hard, H, d).transpose(1, 2)
        v = module.v_proj(x_h).reshape(1, n_hard, H, d).transpose(1, 2)
        hard_out = (
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=True,
            )
            .transpose(1, 2)
            .reshape(1, n_hard, D)
        )
        hard_out = module.o_proj(hard_out).squeeze(0)
        if n_hard == S:
            blend = diff_gate[b].unsqueeze(-1)
            output[b] = blend * hard_out + (1 - blend) * easy_out[b]
            continue
        blend = diff_gate[b, hard_idx].unsqueeze(-1)
        output[b, hard_idx] = blend * hard_out + (1 - blend) * easy_out[b, hard_idx]
    return output


def _op_strided_attention(module, inputs, _):
    """Multi-head attention where each head uses a different stride."""
    x = inputs[0]
    if not hasattr(module, "q_proj"):
        return x
    B, S, D = x.shape
    q = module.q_proj(x)
    k = module.k_proj(x)
    v = module.v_proj(x)
    H = min(8, D)
    if D % H != 0:
        H = 1
    d = D // H
    q = q.reshape(B, S, H, d).permute(0, 2, 1, 3)
    k = k.reshape(B, S, H, d).permute(0, 2, 1, 3)
    v = v.reshape(B, S, H, d).permute(0, 2, 1, 3)

    scale = d**-0.5
    out_heads = []
    for h in range(H):
        stride = max(1, 2 ** (h % 4))  # strides: 1, 2, 4, 8, 1, 2, 4, 8
        # Gather strided positions
        indices = torch.arange(0, S, stride, device=x.device)
        k_s = k[:, h, indices]  # (B, S//stride, d)
        v_s = v[:, h, indices]
        q_h = q[:, h]  # (B, S, d)

        attn = torch.matmul(q_h, k_s.transpose(-2, -1)) * scale
        # Causal: position i can only attend to strided positions <= i
        pos_mask = indices.unsqueeze(0) > torch.arange(S, device=x.device).unsqueeze(1)
        attn = attn.masked_fill(pos_mask.unsqueeze(0), float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        out_h = torch.matmul(attn, v_s)  # (B, S, d)
        out_heads.append(out_h)

    out = torch.cat(out_heads, dim=-1)  # (B, S, D) if H*d == D
    if out.shape[-1] != D:
        out = out[..., :D]
    return module.o_proj(out)


def _op_gated_progressive_attention(module, inputs, _):
    """Compute attention only for tokens whose gate is active enough to justify it."""
    x = inputs[0]
    if not hasattr(module, "q_proj"):
        return x
    B, S, D = x.shape

    gate = torch.sigmoid(module.gate_proj(x))
    token_gate = gate.mean(dim=-1)
    active_mask = token_gate >= 0.5
    if not active_mask.any():
        return x

    H = min(8, D)
    if D % H != 0:
        H = 1
    d = D // H
    output = x.clone()

    for b in range(B):
        active_idx = torch.nonzero(active_mask[b], as_tuple=False).squeeze(-1)
        n_active = int(active_idx.numel())
        if n_active == 0:
            continue
        if n_active == S:
            x_a = x[b : b + 1]
        else:
            x_a = x[b : b + 1, active_idx]
        q = module.q_proj(x_a).reshape(1, n_active, H, d).transpose(1, 2)
        k = module.k_proj(x_a).reshape(1, n_active, H, d).transpose(1, 2)
        v = module.v_proj(x_a).reshape(1, n_active, H, d).transpose(1, 2)
        attn_out = (
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=True,
            )
            .transpose(1, 2)
            .reshape(1, n_active, D)
        )
        attn_out = module.o_proj(attn_out).squeeze(0)
        if n_active == S:
            output[b] = gate[b] * attn_out + (1 - gate[b]) * x[b]
            continue
        output[b, active_idx] = (
            gate[b, active_idx] * attn_out
            + (1 - gate[b, active_idx]) * x[b, active_idx]
        )
    return output


def _op_gated_linear_attention(module, inputs, _):
    """GLA: linear attention with data-dependent decay gates."""
    x = inputs[0]
    if not hasattr(module, "q_proj"):
        return x
    B, S, D = x.shape
    q = module.q_proj(x)
    k = module.k_proj(x)
    v = module.v_proj(x)
    g = torch.sigmoid(module.gate_proj(x))  # decay gate

    H = min(8, D)
    if D % H != 0:
        H = 1
    d = D // H
    q = q.reshape(B, S, H, d).permute(0, 2, 1, 3)
    k = k.reshape(B, S, H, d).permute(0, 2, 1, 3)
    v = v.reshape(B, S, H, d).permute(0, 2, 1, 3)
    g = g.reshape(B, S, H, d).permute(0, 2, 1, 3)

    # Feature map for linear attention (ELU+1 kernel)
    q = F.elu(q) + 1
    k = F.elu(k) + 1

    # Chunked GLA: accumulate KV state with gated decay
    BH = B * H
    q_f = q.reshape(BH, S, d)
    k_f = k.reshape(BH, S, d)
    v_f = v.reshape(BH, S, d)
    g_f = g.reshape(BH, S, d)

    CHUNK = min(32, S)
    state = torch.zeros(BH, d, d, device=x.device, dtype=x.dtype)
    out_chunks = []

    for c_start in range(0, S, CHUNK):
        c_end = min(c_start + CHUNK, S)
        q_c = q_f[:, c_start:c_end]
        k_c = k_f[:, c_start:c_end]
        v_c = v_f[:, c_start:c_end]
        g_c = g_f[:, c_start:c_end]

        # Intra-chunk: causal linear attention
        kv_c = torch.bmm(k_c.transpose(-2, -1), v_c)  # (BH, d, d)
        # Apply gated decay to accumulated state
        decay = g_c.mean(dim=1, keepdim=True).squeeze(1)  # avg gate per chunk
        state = state * decay.unsqueeze(-1) + kv_c
        # Query against accumulated state
        out_c = torch.bmm(q_c, state)
        out_chunks.append(out_c)

    out = torch.cat(out_chunks, dim=1).reshape(B, H, S, d)
    out = out.permute(0, 2, 1, 3).reshape(B, S, D)
    return module.o_proj(out)


def _op_long_conv_hyena(module, inputs, _):
    """Hyena: implicit long convolution + multiplicative gating."""
    x = inputs[0]
    if not hasattr(module, "in_proj"):
        return x
    B, S, D = x.shape
    orig_dtype = x.dtype

    # Project to gate and value
    proj = module.in_proj(x)  # (B, S, 2D)
    gate, val = proj.chunk(2, dim=-1)  # each (B, S, D)
    gate = torch.sigmoid(gate)

    # Generate implicit convolution kernel from positions
    positions = torch.linspace(0, 1, S, device=x.device).unsqueeze(-1)  # (S, 1)
    kernel = module.kernel_net(positions)  # (S, D)

    # Apply convolution via FFT (O(S log S))
    # torch.fft does not support bf16/fp16 on the current CUDA path.
    # Compute the FFT in fp32, then restore the original dtype.
    kernel_fft_in = (
        kernel.float() if kernel.dtype in (torch.bfloat16, torch.float16) else kernel
    )
    val_fft_in = val.float() if val.dtype in (torch.bfloat16, torch.float16) else val
    # Causal: zero out future kernel positions
    kernel_fft = torch.fft.rfft(kernel_fft_in, n=2 * S, dim=0)
    val_fft = torch.fft.rfft(val_fft_in, n=2 * S, dim=1)
    conv_out = torch.fft.irfft(kernel_fft.unsqueeze(0) * val_fft, n=2 * S, dim=1)
    conv_out = conv_out[:, :S, :]  # causal: take first S positions
    if conv_out.dtype != orig_dtype:
        conv_out = conv_out.to(orig_dtype)

    # Multiplicative gating
    out = gate * conv_out
    return module.out_proj(out)


def _op_associative_memory(module, inputs, _):
    """Modern Hopfield: content-addressed retrieval with exponential capacity."""
    x = inputs[0]
    if not hasattr(module, "query_proj"):
        return x
    B, S, D = x.shape

    queries = module.query_proj(x)  # (B, S, D)
    keys = module.memory_proj(x)  # (B, S, D) -- stored patterns
    values = module.value_proj(x)  # (B, S, D)
    beta = torch.clamp(module.beta, min=0.1, max=10.0)

    # Hopfield energy: softmax(beta * Q . K^T) . V
    # Same as attention but with learnable temperature
    energy = torch.matmul(queries, keys.transpose(-2, -1)) * beta / (D**0.5)

    # Causal mask
    causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
    energy = energy.masked_fill(causal.unsqueeze(0), float("-inf"))

    # Retrieve: softmax over stored patterns
    retrieval_weights = torch.softmax(energy, dim=-1)
    retrieved = torch.matmul(retrieval_weights, values)

    return module.o_proj(retrieved)


def _op_mixture_of_recursions(module, inputs, _):
    """MoR: shared block applied variable times per token based on router."""
    x = inputs[0]
    if not hasattr(module, "depth_router"):
        return x
    B, S, D = x.shape
    MAX_DEPTH = 4

    # Route: predict depth per token
    depth_logits = module.depth_router(x.detach())  # (B, S, 4)
    # Soft routing: weighted combination of 1-4 recursion depths
    depth_weights = torch.softmax(depth_logits, dim=-1)  # (B, S, 4)

    # Apply shared block iteratively, accumulate weighted outputs
    h = x
    accumulated = torch.zeros_like(x)
    for d in range(MAX_DEPTH):
        h = module.block_norm(h)
        g = torch.sigmoid(module.block_gate(h))
        up = module.block_ffn_up(h)
        h_new = module.block_ffn_down(g * F.silu(up))
        h = x + h_new  # residual from input, not previous step
        # Weight this depth's output by its routing probability
        accumulated = accumulated + depth_weights[:, :, d : d + 1] * h

    return accumulated


def _op_token_hodge_mixer(module, inputs, _):
    """Causal finite-incidence token mixer over edges and local triangles."""
    x = inputs[0]
    if not hasattr(module, "edge_proj"):
        return x
    prev = F.pad(x[:, :-1], (0, 0, 1, 0))
    edge = x - prev
    edge_flow = module.edge_proj(edge)
    prev_edge_flow = F.pad(edge_flow[:, :-1], (0, 0, 1, 0))
    divergence = edge_flow - prev_edge_flow

    prev_edge = F.pad(edge[:, :-1], (0, 0, 1, 0))
    triangle_boundary = module.face_proj(edge - prev_edge)
    causal_face_memory = triangle_boundary.cumsum(dim=1) / (
        torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
    )
    gate = torch.sigmoid(module.gate_proj(x))
    mixed = x + gate * (divergence + causal_face_memory)
    return module.out_proj(mixed)


def _op_wavelet_packet_mix(module, inputs, config):
    """Causal Haar wavelet-packet mixer with learned low/high channel recombination."""
    x = inputs[0]
    if not hasattr(module, "low_proj"):
        return x
    levels = 2
    if isinstance(config, dict):
        levels = max(1, min(4, int(config.get("levels", levels))))
    low = x
    high_accum = torch.zeros_like(x)
    inv_sqrt2 = 2.0**-0.5
    for _ in range(levels):
        prev = F.pad(low[:, :-1], (0, 0, 1, 0))
        next_low = (low + prev) * inv_sqrt2
        high = (low - prev) * inv_sqrt2
        high_accum = high_accum + high
        low = next_low
    gate = torch.sigmoid(module.gate_proj(x))
    mixed = (
        module.low_proj(low) * module.wavelet_low_scale
        + module.high_proj(high_accum / float(levels)) * module.wavelet_high_scale
    )
    return module.out_proj(gate * mixed + (1.0 - gate) * x)


def _op_retention_mix(module, inputs, _):
    """RetNet-style causal exponential retention in channel state form."""
    x = inputs[0]
    if not hasattr(module, "retention_log_decay"):
        return x
    B, S, D = x.shape
    q = module.q_proj(x)
    k = module.k_proj(x)
    v = module.v_proj(x)
    log_decay = -F.softplus(module.retention_log_decay).to(
        device=x.device, dtype=x.dtype
    )
    phase = torch.cos(module.retention_phase).to(device=x.device, dtype=x.dtype)
    log_a = log_decay.view(1, D, 1).expand(B, D, S).contiguous()
    kv_t = (k * v).permute(0, 2, 1).contiguous()
    state = _parallel_associative_scan(log_a, kv_t).permute(0, 2, 1)
    norm_t = k.abs().permute(0, 2, 1).contiguous()
    norm = _parallel_associative_scan(log_a, norm_t).permute(0, 2, 1)
    out = (q * state * phase.view(1, 1, D)) / norm.clamp(min=1e-6)
    return module.o_proj(out)


def _op_product_key_memory(module, inputs, _):
    """Factorized product-key memory lookup with sparse top-k value retrieval."""
    x = inputs[0]
    if not hasattr(module, "key_left"):
        return x
    B, S, D = x.shape
    left = x[..., : module.pkm_left_dim]
    right = x[..., module.pkm_left_dim : module.pkm_left_dim + module.pkm_right_dim]
    if right.shape[-1] < module.pkm_right_dim:
        right = F.pad(right, (0, module.pkm_right_dim - right.shape[-1]))
    left_scores = torch.matmul(left, module.key_left.t())
    right_scores = torch.matmul(right, module.key_right.t())
    pair_scores = (left_scores.unsqueeze(-1) + right_scores.unsqueeze(-2)).reshape(
        B, S, -1
    )
    k_top = min(module.pkm_top_k, pair_scores.shape[-1])
    top_scores, top_idx = torch.topk(pair_scores, k=k_top, dim=-1)
    weights = torch.softmax(top_scores, dim=-1)
    gathered = module.memory_values[top_idx.reshape(-1)].reshape(B, S, k_top, D)
    retrieved = (weights.unsqueeze(-1) * gathered).sum(dim=-2)
    return module.o_proj(retrieved)


OP_IMPLS: Dict[str, Callable] = {
    "selective_scan": _op_selective_scan,
    "conv1d_seq": _op_conv1d_seq,
    "rwkv_channel": _op_rwkv_channel,
    "diff_attention": _op_diff_attention,
    "state_space": _op_state_space,
    "conv_only": _op_conv_only,
    "gated_delta": _op_gated_delta,
    "dplr_gated_delta": _op_dplr_gated_delta,
    "rwkv_time_mixing": _op_rwkv_time_mixing,
    "difficulty_routed_attention": _op_difficulty_routed_attention,
    "strided_attention": _op_strided_attention,
    "gated_progressive_attention": _op_gated_progressive_attention,
    "gated_linear_attention": _op_gated_linear_attention,
    "long_conv_hyena": _op_long_conv_hyena,
    "associative_memory": _op_associative_memory,
    "mixture_of_recursions": _op_mixture_of_recursions,
    "token_hodge_mixer": _op_token_hodge_mixer,
    "wavelet_packet_mix": _op_wavelet_packet_mix,
    "retention_mix": _op_retention_mix,
    "product_key_memory": _op_product_key_memory,
}
