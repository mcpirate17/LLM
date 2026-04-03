from __future__ import annotations

import torch
import torch.nn.functional as F

from .compiler_op_utils import (
    _build_block_sparse_mask,
    _build_nm_mask,
    _flatten_for_kernel,
    _record_sparse_telemetry,
    _safe_linear,
    _sparse_density_sampled,
    _unflatten_from_kernel,
    _c,
    record_kernel_fallback,
)
from research.env import aria_core, HAS_ARIA_CORE

_NM_MASK_REFRESH_INTERVAL: int = 100


def _op_nm_sparse_linear(module, inputs, config):
    if not hasattr(module, "weight"):
        return inputs[0]
    n = int(getattr(module, "sparsity_n", config.get("n", 2)))
    m = int(getattr(module, "sparsity_m", config.get("m", 4)))
    if m <= 0 or n <= 0 or n > m or (module.weight.shape[1] % m != 0):
        return _safe_linear(inputs[0], module.weight)

    cache = getattr(module, "_nm_mask_cache", None)
    call_count = getattr(module, "_nm_call_count", 0) + 1
    module._nm_call_count = call_count
    if cache is None or call_count % _NM_MASK_REFRESH_INTERVAL == 0:
        if (
            HAS_ARIA_CORE
            and inputs[0].device.type == "cpu"
            and inputs[0].dtype == torch.float32
        ):
            mask = aria_core.nm_sparse_mask_f32(module.weight, n, m).float()
        else:
            mask = _build_nm_mask(module.weight, n=n, m=m)
        module._nm_mask_cache = mask
    else:
        mask = cache
    return _safe_linear(inputs[0], module.weight * mask)


def _op_block_sparse_linear(module, inputs, config):
    if not hasattr(module, "weight"):
        return inputs[0]
    block_size = int(getattr(module, "block_size", config.get("block_size", 16)))
    block_density = float(
        getattr(module, "block_density", config.get("block_density", 0.25))
    )

    if _c(inputs[0]):
        mask = _build_block_sparse_mask(module.weight, block_size, block_density)
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
                module, "block_sparse_linear", _sparse_density_sampled(mask, module)
            )
            return out

    mask = _build_block_sparse_mask(module.weight, block_size, block_density)
    _record_sparse_telemetry(
        module, "block_sparse_linear", _sparse_density_sampled(mask, module)
    )

    try:
        from . import kernels

        if inputs[0].is_cuda:
            return kernels.triton_block_sparse_linear(
                inputs[0], module.weight, mask, block_size
            )
    except (ImportError, RuntimeError, AttributeError) as exc:
        record_kernel_fallback("triton_block_sparse_linear", exc)

    return _safe_linear(inputs[0], module.weight * mask)


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
        module, "semi_structured_2_4_linear", _sparse_density_sampled(mask, module)
    )
    return _safe_linear(inputs[0], module.weight * mask)


def _op_kronecker_linear(module, inputs, config):
    x = inputs[0]
    B, S, D = x.shape
    p = int((D) ** 0.5)
    q = D // p
    if p * q != D:
        for candidate in range(int((D) ** 0.5), 0, -1):
            if D % candidate == 0:
                p = candidate
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


def _op_sparse_bottleneck_moe(module, inputs, config):
    x = inputs[0]
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
    expert_weights = torch.zeros(B, S, n_ways, device=x.device, dtype=x.dtype)
    expert_weights.scatter_add_(2, topk_idx, gate_weights.to(expert_weights.dtype))

    cached_downs = getattr(module, "_expert_downs", None)
    cached_ups = getattr(module, "_expert_ups", None)
    if cached_downs is not None and len(cached_downs) == n_ways:
        W_downs = list(cached_downs)
        W_ups = list(cached_ups)
    else:
        W_downs = []
        W_ups = []
        for idx in range(n_ways):
            if hasattr(module, f"expert_down_{idx}"):
                W_downs.append(getattr(module, f"expert_down_{idx}"))
                W_ups.append(getattr(module, f"expert_up_{idx}"))
            else:
                gen_e = torch.Generator(device="cpu")
                gen_e.manual_seed(D * 1000 + idx * 100 + 1)
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
    W_down_all = torch.stack(W_downs)
    W_up_all = torch.stack(W_ups)
    hidden_all = torch.einsum("bsd,edh->bseh", x, W_down_all)
    hidden_all = F.gelu(hidden_all)
    expert_outs = torch.einsum("bseh,ehd->bsed", hidden_all, W_up_all)
    return (expert_weights.unsqueeze(-1) * expert_outs).sum(dim=2)


def _op_chebyshev_spectral_mix(module, inputs, config):
    x = inputs[0]
    order = max(2, min(config.get("chebyshev_order", 6), 16))
    D = x.shape[-1]
    x_norm = torch.tanh(x)
    coeffs = []
    gen_c = torch.Generator(device="cpu")
    gen_c.manual_seed(order * 65537 + D)
    for idx in range(order):
        if hasattr(module, f"cheb_c{idx}"):
            coeffs.append(getattr(module, f"cheb_c{idx}"))
        else:
            c = (torch.randn(D, generator=gen_c) * (order**-0.5)).to(
                device=x.device, dtype=x.dtype
            )
            if idx == 1:
                c = c + 1.0
            coeffs.append(c)
    t_prev2 = torch.ones_like(x_norm)
    t_prev1 = x_norm
    output = coeffs[0] * t_prev2 + coeffs[1] * t_prev1
    for idx in range(2, order):
        t_k = 2 * x_norm * t_prev1 - t_prev2
        output = output + coeffs[idx] * t_k
        t_prev2 = t_prev1
        t_prev1 = t_k
    return output


def _op_latent_attention_compressor(module, inputs, config):
    x = inputs[0]
    if not hasattr(module, "kv_compress"):
        return x
    latent = F.linear(x, module.kv_compress.to(x.dtype))
    kv = F.linear(latent, module.kv_up.to(x.dtype))
    D = x.shape[-1]
    k, v = kv[..., :D], kv[..., D:]
    return x + torch.sigmoid(k) * v


def _op_signal_conditioned_compression(module, inputs, config):
    x, routing_signal = inputs[0], inputs[1]
    if not hasattr(module, "weight_full"):
        return x
    if routing_signal.shape[-1] > 1:
        s = torch.sigmoid(routing_signal[..., 0:1])
    else:
        s = torch.sigmoid(routing_signal)
    full = _safe_linear(x, module.weight_full)
    if hasattr(module, "U_comp"):
        comp = _safe_linear(_safe_linear(x, module.U_comp), module.V_comp)
        return s * full + (1 - s) * comp
    return full


def _op_token_class_proj(module, inputs, config):
    x = inputs[0]
    if not hasattr(module, "classifier_weight"):
        return x
    if x.shape[-1] != module.classifier_weight.shape[1]:
        return x
    scores = F.gelu(_safe_linear(x, module.classifier_weight))
    module._class_scores = scores.detach()
    return _safe_linear(scores, module.classifier_proj_back)


def _op_adaptive_rank_gate(module, inputs, config):
    x = inputs[0]
    if not hasattr(module, "weight_full"):
        return x
    dt = x.dtype
    if hasattr(module, "token_gate"):
        s = torch.sigmoid(F.linear(x, module.token_gate.to(dt)))
    else:
        s = torch.sigmoid(module.compress_param)
    full = F.linear(x, module.weight_full.to(dt))
    if hasattr(module, "U_comp"):
        comp = F.linear(F.linear(x, module.U_comp.to(dt)), module.V_comp.to(dt))
        return s * full + (1 - s) * comp
    return full


def _op_dual_compression_blend(module, inputs, config):
    x = inputs[0]
    routing_signal = inputs[1] if len(inputs) > 1 else x
    if not hasattr(module, "expert_weights"):
        return x
    weights = F.softmax(routing_signal, dim=-1)
    out0 = _safe_linear(_safe_linear(x, module.U_lr), module.V_lr)
    hidden1 = F.gelu(_safe_linear(x, module.W_down))
    out1 = _safe_linear(hidden1, module.W_up)
    return out0 * weights[..., 0:1] + out1 * weights[..., 1:2]


def _op_ternary_projection(module, inputs, config):
    x = inputs[0]
    if not hasattr(module, "weight"):
        return x
    w = module.weight
    gamma = w.abs().mean().clamp(min=1e-5)
    w_quant = torch.round(torch.clamp(w / gamma, -1, 1))
    w_sim = w + (w_quant * gamma - w).detach()
    bias = getattr(module, "bias", None)
    return F.linear(
        x, w_sim.to(x.dtype), bias.to(x.dtype) if bias is not None else None
    )


OP_IMPLS = {
    "nm_sparse_linear": _op_nm_sparse_linear,
    "block_sparse_linear": _op_block_sparse_linear,
    "semi_structured_2_4_linear": _op_semi_structured_2_4_linear,
    "kronecker_linear": _op_kronecker_linear,
    "sparse_bottleneck_moe": _op_sparse_bottleneck_moe,
    "n_way_sparse_router": _op_sparse_bottleneck_moe,
    "chebyshev_spectral_mix": _op_chebyshev_spectral_mix,
    "latent_attention_compressor": _op_latent_attention_compressor,
    "signal_conditioned_compression": _op_signal_conditioned_compression,
    "routing_conditioned_compression": _op_signal_conditioned_compression,
    "token_class_proj": _op_token_class_proj,
    "token_type_classifier": _op_token_class_proj,
    "adaptive_rank_gate": _op_adaptive_rank_gate,
    "progressive_compression_gate": _op_adaptive_rank_gate,
    "dual_compression_blend": _op_dual_compression_blend,
    "compression_mixture_experts": _op_dual_compression_blend,
    "ternary_projection": _op_ternary_projection,
}
