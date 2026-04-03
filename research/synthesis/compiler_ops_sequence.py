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
    out = F.conv1d(x_padded, module.conv_weight, groups=D)
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
    shifted = F.pad(x[:, :-1], (0, 0, 1, 0)) if x.ndim == 3 else x
    xk = x * module.mix_k + shifted * (1 - module.mix_k)
    xr = x * module.mix_r + shifted * (1 - module.mix_r)
    k = torch.square(torch.relu(module.key_proj(xk)))
    return torch.sigmoid(module.receptance_proj(xr)) * module.value_proj(k)


def _op_diff_attention(module, inputs, _):
    x = inputs[0]
    if not hasattr(module, "q_proj"):
        return x
    B, S, _ = x.shape
    nh, hd = module.n_heads, module.head_dim
    q = module.q_proj(x).reshape(B, S, nh, 2, hd).permute(0, 2, 3, 1, 4)
    k = module.k_proj(x).reshape(B, S, nh, 2, hd).permute(0, 2, 3, 1, 4)
    v = module.v_proj(x).reshape(B, S, nh, hd).transpose(1, 2)
    scale = hd**-0.5
    mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
    attn1 = (q[:, :, 0] @ k[:, :, 0].transpose(-2, -1)) * scale
    attn2 = (q[:, :, 1] @ k[:, :, 1].transpose(-2, -1)) * scale
    attn1.masked_fill_(mask, float("-inf"))
    attn2.masked_fill_(mask, float("-inf"))
    diff = F.softmax(attn1, dim=-1) - module.lambda_param.abs() * F.softmax(
        attn2, dim=-1
    )
    out = (diff @ v).transpose(1, 2).reshape(B, S, -1)
    return module.o_proj(out)


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


def _op_gated_delta(module, inputs, _):
    x = inputs[0]
    if not hasattr(module, "q_proj"):
        return x
    B, S, D = x.shape
    q = module.q_proj(x)
    k = module.k_proj(x)
    v = module.v_proj(x)
    alpha = torch.sigmoid(module.alpha_proj(x))
    beta = torch.sigmoid(module.beta_proj(x))
    eff_decay = alpha - beta
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
    return module.o_proj(out)


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
    w = -torch.exp(module.w_decay)
    u = module.u_bonus
    exp_k = torch.exp(k.clamp(-20, 20))
    indices = torch.arange(S, device=x.device, dtype=x.dtype)
    log_kernel = w.view(D, 1, 1) * (S - 1 - indices).view(1, 1, S)
    kernel = torch.exp(log_kernel.clamp(-20, 0))
    u_wkv = (exp_k * v).permute(0, 2, 1)
    u_den = exp_k.permute(0, 2, 1)
    wkv_incl = F.conv1d(F.pad(u_wkv, (S - 1, 0)), kernel, groups=D)
    den_incl = F.conv1d(F.pad(u_den, (S - 1, 0)), kernel, groups=D)
    wkv_before = F.pad(wkv_incl[..., :-1], (1, 0)).permute(0, 2, 1)
    den_before = F.pad(den_incl[..., :-1], (1, 0)).permute(0, 2, 1)
    p = torch.exp((u + k).clamp(-20, 20))
    out = r * (wkv_before + p * v) / (den_before + p).clamp(min=1e-8)
    return _safe_linear(out, module.W_o)


OP_IMPLS: Dict[str, Callable] = {
    "selective_scan": _op_selective_scan,
    "conv1d_seq": _op_conv1d_seq,
    "rwkv_channel": _op_rwkv_channel,
    "diff_attention": _op_diff_attention,
    "state_space": _op_state_space,
    "conv_only": _op_conv_only,
    "gated_delta": _op_gated_delta,
    "rwkv_time_mixing": _op_rwkv_time_mixing,
}
