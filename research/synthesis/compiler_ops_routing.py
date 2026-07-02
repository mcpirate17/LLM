from __future__ import annotations

import logging
import math
from typing import Callable, Dict

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F

from .compiler_op_utils import (
    HAS_ARIA_CORE,
    aria_core,
    _c,
    _get_stacked_params,
    _is_inference_tensor,
    _record_routing_telemetry,
    _safe_linear,
    record_kernel_fallback,
)
from .routing_runtime import (
    branch_rms,
    get_routing_progress,
    scheduled_int,
    scheduled_scalar,
    stage_name,
)

_CONSTANT_CACHE_MAX = 128
_DCT_BASIS_CACHE: dict[
    tuple[int, int, tuple[str, int | None], torch.dtype], torch.Tensor
] = {}
_LEGENDRE_BASIS_CACHE: dict[
    tuple[int, int, tuple[str, int | None], torch.dtype], torch.Tensor
] = {}
_TRIL_CACHE: dict[
    tuple[int, tuple[str, int | None], torch.dtype], torch.Tensor
] = {}
_GRAPH_EIGBASIS_CACHE: dict[
    tuple[int, tuple[str, int | None], float], torch.Tensor
] = {}


def _device_cache_key(device: torch.device) -> tuple[str, int | None]:
    dev = torch.device(device)
    return dev.type, dev.index


def _cache_get(cache: dict, key: tuple) -> torch.Tensor | None:
    cached = cache.get(key)
    if cached is None:
        return None
    if torch.is_grad_enabled() and _is_inference_tensor(cached):
        return None
    return cached


def _cache_put(cache: dict, key: tuple, value: torch.Tensor) -> torch.Tensor:
    if len(cache) >= _CONSTANT_CACHE_MAX:
        cache.clear()
    cache[key] = value
    return value


def _cached_tril(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (int(seq_len), _device_cache_key(device), dtype)
    cached = _cache_get(_TRIL_CACHE, key)
    if cached is not None:
        return cached
    return _cache_put(
        _TRIL_CACHE,
        key,
        torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype)),
    )


def _capture_routing_trace(module) -> bool:
    return bool(getattr(module, "_capture_routing_trace", False))


def _routing_scores_from_x(x: torch.Tensor) -> torch.Tensor:
    """Simple, deterministic score: mean over channels."""
    return x.mean(dim=-1)


def _apply_moe_load_balance(
    module, logits: torch.Tensor, n_experts: int, gamma: float = 0.001
) -> torch.Tensor:
    """Auxiliary-loss-free load balancing (DeepSeek-V3 style).

    Adds a per-expert bias to gate logits. During training, the bias is updated
    outside of backprop based on observed expert load — overloaded experts get
    negative bias, underloaded get positive. No gradient contamination.
    """
    # Initialize bias buffer if needed (not a Parameter — no gradients).
    # Some routing ops reuse the same helper with different expert counts across
    # model rebuilds or replay paths. A stale bias vector is what produced the
    # historical 3-vs-2 / 3-vs-8 crashes in stage1.
    bias_buf = getattr(module, "_moe_balance_bias", None)
    if (
        bias_buf is None
        or not isinstance(bias_buf, torch.Tensor)
        or bias_buf.ndim != 1
        or bias_buf.numel() != n_experts
    ):
        module._moe_balance_bias = torch.zeros(
            n_experts, device=logits.device, dtype=logits.dtype
        )
    bias = module._moe_balance_bias.to(device=logits.device, dtype=logits.dtype)

    # Apply bias to logits (forward only — bias is detached)
    logits = logits + bias.detach()

    # Update bias based on load (training only, no grad)
    if getattr(module, "training", False) and gamma > 0:
        with torch.no_grad():
            # Count how many tokens selected each expert
            selected = logits.argmax(dim=-1)  # (B, S)
            counts = (
                torch.bincount(selected.flatten(), minlength=n_experts)
                .float()
                .to(logits.device)[:n_experts]
            )
            target = counts.sum() / n_experts
            # Increase bias for underloaded, decrease for overloaded
            module._moe_balance_bias = bias + gamma * (target - counts)

    return logits


def _op_topk_gate(module, inputs, _):
    if not hasattr(module, "gate_proj"):
        return inputs[0]
    x = inputs[0]
    B, S, D = x.shape
    if (
        _c(x)
        and hasattr(aria_core, "topk_gate_f32")
        and isinstance(module.gate_proj, torch.Tensor)
        and module.gate_proj.dim() == 2
        and module.gate_proj.shape[0] >= 2
        and module.gate_proj.shape[1] == D
    ):
        try:
            native_out = aria_core.topk_gate_f32(x, module.gate_proj, 2)
            if isinstance(native_out, torch.Tensor) and native_out.shape == x.shape:
                return native_out
        except (ImportError, RuntimeError, AttributeError) as e:
            record_kernel_fallback("topk_gate_f32", e)
    logits = _safe_linear(x, module.gate_proj)
    gate_weights = F.softmax(logits, dim=-1)

    # Record routing telemetry
    _record_routing_telemetry(module, 2, gate_weights.argmax(dim=-1), logits=logits)

    half = D // 2
    out = torch.cat(
        [
            x[..., :half] * gate_weights[..., 0:1],
            x[..., half : 2 * half] * gate_weights[..., 1:2],
        ],
        dim=-1,
    )
    if D > 2 * half:
        out = torch.cat([out, x[..., 2 * half :]], dim=-1)
    return out


def _moe_sequential_dispatch(
    x: torch.Tensor,
    experts: torch.nn.ModuleList,
    n_actual: int,
    weights: torch.Tensor,
    indices: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Sequential MoE dispatch via sort + bincount + split.

    Processes only the tokens assigned to each expert (no redundant compute).
    Faster than batched einsum on CPU where parallelism is limited.
    """
    B, S, D = x.shape
    BS = B * S
    output = torch.zeros_like(x)
    x_flat = x.reshape(BS, D)
    idx_flat = indices.reshape(BS, top_k)
    w_flat = weights.reshape(BS, top_k)
    output_flat = output.view(BS, D)

    for k_idx in range(top_k):
        expert_ids = idx_flat[:, k_idx]
        slot_weights = w_flat[:, k_idx].unsqueeze(-1)

        sort_order = expert_ids.argsort(stable=True)
        sorted_x = x_flat[sort_order]
        sorted_w = slot_weights[sort_order]

        expert_counts = torch.bincount(expert_ids, minlength=n_actual).tolist()
        result_sorted = torch.empty_like(sorted_x)
        start = 0
        for e_idx, count in enumerate(expert_counts):
            if count == 0:
                continue
            x_chunk = sorted_x.narrow(0, start, count)
            w_chunk = sorted_w.narrow(0, start, count)
            out = experts[e_idx](x_chunk)
            result_sorted.narrow(0, start, count).copy_(
                out.to(x.dtype) * w_chunk.to(x.dtype)
            )
            start += count

        output_flat.index_add_(0, sort_order, result_sorted)

    return output


def _moe_batched_expert_forward(
    x: torch.Tensor,
    W_down: torch.Tensor,
    W_up: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Batched MoE forward: compute all experts in parallel via einsum.

    Instead of looping over experts and processing token subsets, this stacks
    all expert weights and computes ALL expert outputs for ALL tokens in a
    single batched matmul. Only the top-k expert outputs per token are kept,
    weighted by the routing weights.

    This trades O(E) redundant compute for eliminating the Python expert loop.
    For typical n_experts (4-8), the GPU parallelism gain far exceeds the
    wasted compute.

    Args:
        x: (B, S, D) input tokens
        W_down: (E, H, D) stacked expert down-projection weights
        W_up: (E, D, H) stacked expert up-projection weights
        weights: (B, S, top_k) routing weights per token
        indices: (B, S, top_k) expert indices per token
        top_k: number of experts per token
    """
    # All experts in parallel: x @ W_down^T → GELU → @ W_up^T
    # (B,S,D) @ (E,H,D)^T → (B,S,E,H)
    hidden = torch.einsum("bsd,ehd->bseh", x, W_down)
    hidden = F.gelu(hidden)
    # (B,S,E,H) @ (E,D,H)^T → (B,S,E,D)
    expert_outs = torch.einsum("bseh,edh->bsed", hidden, W_up)

    # Gather only the top-k expert outputs per token
    # indices: (B,S,top_k) → expand to (B,S,top_k,D)
    idx_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1])
    selected = torch.gather(expert_outs, dim=2, index=idx_expanded)  # (B,S,K,D)

    # Weight and sum: (B,S,K,1) * (B,S,K,D) → sum → (B,S,D)
    return (weights.unsqueeze(-1) * selected).sum(dim=2)


def _moe_get_stacked_weights(module, n_experts: int, dtype: torch.dtype):
    """Stack expert Linear→GELU→Linear weights into batched tensors, cached."""
    cache_key = f"_moe_stacked_{dtype}"
    cached = getattr(module, cache_key, None)
    if cached is not None and not any(_is_inference_tensor(t) for t in cached):
        return cached
    W_downs = []
    W_ups = []
    for i in range(n_experts):
        expert = module.experts[i]
        W_downs.append(expert[0].weight.to(dtype))  # Linear(D, H).weight is (H, D)
        W_ups.append(expert[2].weight.to(dtype))  # Linear(H, D).weight is (D, H)
    stacked = (torch.stack(W_downs), torch.stack(W_ups))  # (E,H,D), (E,D,H)
    if not torch.is_inference_mode_enabled():
        object.__setattr__(module, cache_key, stacked)
    return stacked


def _op_moe_topk(module, inputs, config):
    """Sparse Mixture-of-Experts channel mixer."""
    x = inputs[0]
    B, S, D = x.shape

    n_experts = int(config.get("num_experts", 4))
    top_k = int(config.get("top_k", 2))

    if not hasattr(module, "gate_weight"):
        return x

    logits = _safe_linear(x, module.gate_weight)
    logits = _apply_moe_load_balance(module, logits, n_experts)
    weights, indices = logits.topk(top_k, dim=-1)
    weights = F.softmax(weights, dim=-1)

    # Record routing telemetry
    _record_routing_telemetry(module, n_experts, indices, logits=logits)

    if not hasattr(module, "experts"):
        return _safe_linear(x, module.weight) if hasattr(module, "weight") else x

    n_actual = len(module.experts)

    if not x.is_cuda:
        return _moe_sequential_dispatch(
            x, module.experts, n_actual, weights, indices, top_k
        )

    # GPU: batched einsum computes all experts in parallel via cuBLAS.
    # Faster than per-token Triton dispatch for typical E=4-8, BS<4K.
    W_down, W_up = _moe_get_stacked_weights(module, n_actual, x.dtype)
    return _moe_batched_expert_forward(x, W_down, W_up, weights, indices, top_k)


def _op_moe_2expert(module, inputs, config):
    """Lightweight 2-expert MoE with learned gating."""
    x = inputs[0]
    B, S, D = x.shape

    if not hasattr(module, "gate_proj"):
        return x

    # Compute gate scores with load balancing
    dt = x.dtype
    logits = _safe_linear(x, module.gate_proj)  # (B, S, 2)
    logits = _apply_moe_load_balance(module, logits, 2)
    weights = F.softmax(logits, dim=-1)  # (B, S, 2)

    # Record routing telemetry
    _record_routing_telemetry(module, 2, weights.argmax(dim=-1), logits=logits)

    # Each expert is a simple linear projection
    e0 = _safe_linear(x, module.expert_0_weight.to(dt))  # (B, S, D)
    e1 = _safe_linear(x, module.expert_1_weight.to(dt))  # (B, S, D)

    # Weighted combination
    output = weights[..., 0:1] * e0 + weights[..., 1:2] * e1
    return output


def _op_swiglu_mlp(module, inputs, _):
    """SwiGLU MLP channel mixer."""
    x = inputs[0]
    if not hasattr(module, "gate_proj"):
        return x
    # The CPU aria_core SwiGLU kernel is slower than the dense PyTorch path on
    # the screening/eval shapes we actually execute. Keep the fast native
    # kernel out of this hot path until the kernel itself is fixed.
    gate = _safe_linear(x, module.gate_proj.weight, module.gate_proj.bias)
    up = _safe_linear(x, module.up_proj.weight, module.up_proj.bias)
    return _safe_linear(
        F.silu(gate) * up, module.down_proj.weight, module.down_proj.bias
    )


def _op_feature_sparsity(module, inputs, config):
    """Feature sparsity: zero out all but top-k positions along last dim.

    Input:  (B, S, D)
    Output: (B, S, D) with only the top-k values per (B, S) slice kept.
    Uses STE with density scaling to keep gradient magnitude stable.
    """
    x = inputs[0]
    D = x.shape[-1]
    k = min(int(config.get("k", max(1, D // 8))), D)
    topk_vals, topk_idx = x.topk(k, dim=-1)  # (B, S, k)
    _record_routing_telemetry(module, D, topk_idx, logits=x)
    # Build sparse mask and scatter top-k values back
    mask = torch.zeros_like(x)
    mask.scatter_(-1, topk_idx, 1.0)
    # STE: forward uses hard mask, backward passes through
    # Sqrt scaling compensates for sparsity — capped at 4.0 to prevent
    # grad explosion when k is small (e.g., k=1, D=512 → sqrt(512)=22.6).
    scale = min((D / k) ** 0.5, 4.0)
    return x * (mask.detach() - x.detach() + x) * scale


def _op_gated_lane_blend(module, inputs, config):
    """Learned difficulty-based lane blend: score tokens, soft-weight N internal linear projections."""
    x = inputs[0]
    B, S, D = x.shape
    n_lanes = int(config.get("n_lanes", 3))

    if not hasattr(module, "lane_scorer"):
        return x
    dt = x.dtype

    lane_logits = _safe_linear(x, module.lane_scorer.to(dt))
    lane_indices = lane_logits.argmax(dim=-1)
    _record_routing_telemetry(module, n_lanes, lane_indices, logits=lane_logits)

    if hasattr(module, "lane_projs") and len(module.lane_projs) >= n_lanes:
        W_all = _get_stacked_params(module, "lane_projs", n_lanes, dt)
        if _c(x):
            y = aria_core.gated_lane_blend_f32(
                x.float(), module.lane_scorer.float(), W_all.float()
            )
            return y.to(dt)
        lane_weights = F.softmax(lane_logits, dim=-1)
        all_outs = torch.einsum("bsd,lod->bslo", x, W_all)
        return (lane_weights.unsqueeze(-1) * all_outs).sum(dim=2)
    return x


def _op_depth_gated_transform(module, inputs, config):
    """Learned difficulty-based depth gate: variable-depth linear transforms per token."""
    x = inputs[0]
    B, S, D = x.shape
    max_depth = max(1, min(6, int(config.get("max_depth", 3))))

    if not hasattr(module, "depth_scorer"):
        return x
    dt = x.dtype

    depth_logits = _safe_linear(x, module.depth_scorer.to(dt))
    depth_indices = depth_logits.argmax(dim=-1)
    _record_routing_telemetry(module, max_depth, depth_indices, logits=depth_logits)

    if hasattr(module, "depth_projs") and len(module.depth_projs) >= max_depth:
        W_all = _get_stacked_params(module, "depth_projs", max_depth, dt)
        if _c(x):
            y = aria_core.depth_gated_transform_f32(
                x.float(), module.depth_scorer.float(), W_all.float()
            )
            return y.to(dt)
        depth_weights = F.softmax(depth_logits, dim=-1)
        all_outs = torch.einsum("bsd,kod->bsko", x, W_all)
        return (depth_weights.unsqueeze(-1) * all_outs).sum(dim=2)
    return x


def _op_adjacent_token_merge(module, inputs, config):
    """Causal token compression: merge each dropped token FORWARD into its
    successor (token p merges INTO p+1), then restore the original seq_len via
    a backward (nearest-kept-≤-i) hold.

    Causality invariant: output[i] depends only on inputs at positions ≤ i.
    Each surviving token absorbs only *earlier* dropped tokens, and the restore
    maps every output position to the most recent kept token at-or-before it.
    Enforced by research/tests/test_adjacent_token_merge_causality.py.

    2026-05-23 fix: the prior implementation merged token p INTO p-1 (backward),
    making output[p-1] depend on x[p] — a one-step next-token leak. Because the
    binding_range/screening/curriculum probes are causal next-token tasks
    (scored at position i against input[i+1]), that leak handed them the label
    and inflated the binding scores of every model containing this op. See
    [[project_adjacent_token_merge_leak]].
    """
    x = inputs[0]
    B, S, D = x.shape
    n_keep = int(config.get("n_keep", S // 2))
    n_keep = max(1, min(n_keep, S))
    n_merge = S - n_keep

    if n_merge <= 0:
        return x

    use_c_kernel = (
        HAS_ARIA_CORE
        and x.device.type == "cpu"
        and x.dtype == torch.float32
        and not x.requires_grad
    )
    if use_c_kernel:
        y, _restore_map = aria_core.token_merge_simple_f32(x, n_keep)
    else:
        # Deterministic causal stride merge: drop every `stride`-th token and
        # fold it FORWARD into its successor (token p merges INTO p+1). This is
        # causal because each surviving token only absorbs *earlier* tokens.
        stride = max(2, S // n_keep)
        # Positions to drop: 1, 1+stride, 1+2*stride, ... (up to n_merge)
        drop_positions = torch.arange(1, S, stride, device=x.device)[:n_merge]

        merged = torch.zeros(B, S, device=x.device, dtype=torch.bool)
        merge_targets = (
            torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1).clone()
        )

        # Forward merge: token at drop_pos merges INTO drop_pos+1 (vectorized).
        # Exclude the final position (it has no successor to fold into) so it
        # stays kept rather than leaking backward into its predecessor.
        valid_drops = drop_positions[(drop_positions > 0) & (drop_positions < S - 1)]
        merged[:, valid_drops] = True
        merge_targets[:, valid_drops] = valid_drops + 1

        # Build merged output: average merged pairs
        target_idx = merge_targets.unsqueeze(-1).expand(-1, -1, D)
        out = x.scatter_add(1, target_idx, x * merged.unsqueeze(-1).float())
        count_map = torch.ones(B, S, 1, device=x.device, dtype=x.dtype)
        count_map.scatter_add_(
            1, merge_targets.unsqueeze(-1), merged.unsqueeze(-1).float()
        )
        out = out / count_map.clamp(min=1)

        # Gather kept tokens via searchsorted (vectorized, no Python loop)
        kept_mask = ~merged
        kept_cumsum = kept_mask.float().cumsum(dim=-1)
        slots = (
            torch.arange(1, n_keep + 1, device=x.device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(B, -1)
        )
        kept_indices = torch.searchsorted(
            kept_cumsum.contiguous(), slots.contiguous()
        ).clamp(max=S - 1)
        y = out.gather(1, kept_indices.unsqueeze(-1).expand(-1, -1, D))

    # Telemetry (lightweight dict update)
    telem = getattr(module, "routing_telemetry", None)
    if telem is None:
        telem = {
            "tokens_total": 0,
            "tokens_processed": 0,
            "merge_kept": 0,
            "merge_dropped": 0,
            "expert_counts": torch.zeros(1, device=x.device),
            "entropy_sum": 0.0,
            "count": 0,
            "heatmap": None,
        }
        module.routing_telemetry = telem
    telem["tokens_total"] += B * S
    telem["tokens_processed"] += B * n_keep
    telem["merge_kept"] += B * n_keep
    telem["merge_dropped"] += B * n_merge
    # Binary entropy of keep/merge decision: H = -p*log(p) - (1-p)*log(1-p)
    p_keep = n_keep / S
    if 0 < p_keep < 1:
        telem["entropy_sum"] += -(
            p_keep * math.log(p_keep) + (1 - p_keep) * math.log(1 - p_keep)
        )
    telem["count"] += 1

    # Restore to original seq length via causal nearest-kept mapping
    if use_c_kernel:
        restore_map = (
            torch.arange(S, device=x.device)
            .unsqueeze(0)
            .expand(B, -1)
            .clamp(max=n_keep - 1)
        )
    else:
        positions = (
            torch.arange(S, device=x.device, dtype=kept_indices.dtype)
            .unsqueeze(0)
            .expand(B, -1)
        )
        restore_map = (
            torch.searchsorted(
                kept_indices.contiguous(), positions.contiguous(), right=True
            )
            - 1
        ).clamp(min=0, max=n_keep - 1)
    return y.gather(1, restore_map.unsqueeze(-1).expand(-1, -1, D))


# ── Routing Control Ops (Phase 2) ────────────────────────────────────


def _op_depth_token_mask(module, inputs, config):
    x = inputs[0]
    B, S, D = x.shape
    capacity = float(config.get("capacity_factor", 0.75))
    if hasattr(module, "router_weight"):
        scores = _safe_linear(x, module.router_weight.to(x.dtype)).squeeze(-1)
    else:
        scores = _routing_scores_from_x(x)
    # Causal sparsity: deterministic stride-based mask that keeps
    # ~capacity fraction of positions without peeking at future tokens.
    # Every position knows its own index — no future information needed.
    stride = max(1, int(1.0 / max(1.0 - capacity, 0.01)))
    pos = torch.arange(S, device=x.device)
    keep_mask = ((pos % stride) != (stride - 1)).float().unsqueeze(0).expand(B, -1)
    # Blend with score-based soft gate for gradient flow (causal mean)
    cumsum = scores.cumsum(dim=-1)
    counts = torch.arange(1, S + 1, device=scores.device, dtype=scores.dtype)
    causal_mean = cumsum / counts
    soft_gate = torch.sigmoid(4.0 * (scores - causal_mean))
    gate = soft_gate * keep_mask
    _record_routing_telemetry(
        module, S, (gate > 0.5).nonzero(as_tuple=False)[:, 1:], logits=scores
    )
    return x * gate.unsqueeze(-1)


def _op_confidence_token_gate(module, inputs, config):
    """True early exit: confidence gate identifies easy tokens.

    Easy tokens are attenuated (zeroed) so downstream FFN work is wasted on
    them.  Their hidden states are stored for auxiliary loss computation
    against the model's shared lm_head in the training loop.
    """
    x = inputs[0]
    threshold = float(config.get("threshold", 0.5))
    if hasattr(module, "confidence_proj"):
        scores = _safe_linear(x, module.confidence_proj.to(x.dtype)).squeeze(
            -1
        )  # (B, S)
    else:
        scores = _routing_scores_from_x(x)
    gate = torch.sigmoid(scores)
    easy_mask = (gate > threshold).float()
    # STE: hard gate forward, soft gate backward
    gate_ste = easy_mask - gate.detach() + gate

    _record_routing_telemetry(module, 2, easy_mask.long(), logits=gate)

    # Store hidden states + gate for aux loss (training only)
    if x.requires_grad:
        module._early_exit_aux = {
            "hidden": x,  # (B, S, D)
            "gate": gate_ste,  # (B, S) — high = easy
        }

    # Hard tokens pass through, easy tokens zeroed (outer residual recovers them)
    return x * (1 - gate_ste).unsqueeze(-1)


def _op_learned_token_gate(module, inputs, config):
    """Learned progressive cascade: soft gate scales tokens by learned difficulty."""
    x = inputs[0]
    threshold = float(config.get("threshold", 0.5))
    if hasattr(module, "cascade_proj"):
        scores = _safe_linear(x, module.cascade_proj.to(x.dtype)).squeeze(-1)  # (B, S)
    else:
        scores = _routing_scores_from_x(x)
    gate = torch.sigmoid(scores)
    _record_routing_telemetry(module, 2, (gate > threshold).long(), logits=gate)
    return x * gate.unsqueeze(-1)


def _op_cheap_verify_blend(module, inputs, config):
    """Speculative execution: cheap path always runs, learned gate blends full path."""
    x = inputs[0]
    if not hasattr(module, "cheap_proj"):
        return x
    dt = x.dtype
    # Cheap path: lightweight linear projection (always runs)
    cheap_out = _safe_linear(x, module.cheap_proj.to(dt))
    # Learned verification gate: decides how much of the full input to blend
    gate = torch.sigmoid(_safe_linear(x, module.verify_gate.to(dt)).squeeze(-1))
    _record_routing_telemetry(module, 2, (gate > 0.5).long(), logits=gate)
    # Blend: cheap_out + gate * (x - cheap_out) = lerp(cheap_out, x, gate)
    return cheap_out + gate.unsqueeze(-1) * (x - cheap_out)


def _op_hybrid_token_gate(module, inputs, config):
    """Cheap token-level gate that separates default traffic from informative tokens."""
    x = inputs[0]
    threshold = scheduled_scalar(module, config, key="threshold", default=0.5)
    gate_temperature = max(
        1e-4, scheduled_scalar(module, config, key="gate_temperature", default=1.0)
    )
    if hasattr(module, "hybrid_gate_proj"):
        scores = _safe_linear(x, module.hybrid_gate_proj.to(x.dtype)).squeeze(-1)
    else:
        scores = _routing_scores_from_x(x)
    gate = torch.sigmoid(scores / gate_temperature)
    keep_mask = gate >= threshold
    gate_ste = keep_mask.to(x.dtype).detach() - gate.detach() + gate
    progress = get_routing_progress(module)
    _record_routing_telemetry(
        module,
        2,
        keep_mask.long(),
        logits=torch.stack([1.0 - gate, gate], dim=-1),
        keep_mask=keep_mask,
        routing_mode="hybrid_token_gate",
        gate_type="single_token",
        span_type="single",
        default_path_count=(~keep_mask),
        routed_token_count=keep_mask,
        trace_payload=(
            {
                "curriculum_stage": stage_name(
                    progress,
                    float(config.get("curriculum_warmup_frac", 0.25)),
                    float(config.get("curriculum_mid_frac", 0.65)),
                ),
                "keep_mask_sample": keep_mask[0].detach().cpu().to(torch.int64).tolist()
                if keep_mask.numel() > 0
                else [],
            }
            if _capture_routing_trace(module)
            else None
        ),
    )
    return x * gate_ste.unsqueeze(-1)


def _minimum_keep_target(
    seq_len: int,
    span_width: int,
    min_keep_fraction: float,
) -> int:
    requested = max(span_width, int(math.ceil(seq_len * max(0.0, min_keep_fraction))))
    return min(seq_len, max(1, requested))


def _minimum_keep_count(
    token_present: torch.Tensor,
    keep_target: int,
) -> torch.Tensor:
    present_counts = token_present.sum(dim=-1)
    requested = max(1, int(keep_target))
    requested_t = torch.full_like(present_counts, requested)
    return torch.minimum(present_counts, requested_t)


def _rescue_keep_mask(
    gate: torch.Tensor,
    token_present: torch.Tensor,
    threshold: float,
    min_keep_count: torch.Tensor,
    max_keep_count: int,
) -> torch.Tensor:
    keep_mask = (gate >= threshold) & token_present
    max_keep = min(int(max_keep_count), int(gate.shape[-1]))
    if max_keep <= 0:
        return keep_mask
    gated_scores = gate.masked_fill(~token_present, -1.0)
    topk_scores, topk_idx = gated_scores.topk(max_keep, dim=-1)
    rescue_mask = torch.zeros_like(keep_mask)
    rank_mask = torch.arange(max_keep, device=gate.device).unsqueeze(
        0
    ) < min_keep_count.unsqueeze(-1)
    rescue_mask.scatter_(1, topk_idx, rank_mask & (topk_scores >= 0.0))
    return keep_mask | rescue_mask


def _build_sparse_spans(
    x: torch.Tensor,
    keep_mask: torch.Tensor,
    span_width: int,
    *,
    gate_strength: torch.Tensor | None = None,
):
    B, S, D = x.shape
    span_features = torch.zeros_like(x)
    span_strength = torch.zeros((B, S), device=x.device, dtype=x.dtype)
    coverage = torch.zeros((B, S), device=x.device, dtype=torch.int64)
    span_positions = torch.full(
        (B, S, span_width), -1, device=x.device, dtype=torch.int64
    )
    min_kept = 1 if span_width <= 1 else 2
    if span_width > S:
        span_counts = torch.zeros((B,), device=x.device, dtype=torch.int64)
        return span_features, span_positions, span_counts, coverage, span_strength

    x_prefix = torch.cat([x.new_zeros(B, 1, D), x.cumsum(dim=1)], dim=1)
    span_sums = x_prefix[:, span_width:] - x_prefix[:, :-span_width]
    keep_int = keep_mask.to(torch.int64)
    keep_prefix = torch.cat(
        [
            torch.zeros((B, 1), device=x.device, dtype=torch.int64),
            keep_int.cumsum(dim=1),
        ],
        dim=1,
    )
    kept_counts = keep_prefix[:, span_width:] - keep_prefix[:, :-span_width]
    valid_windows = kept_counts >= min_kept

    end_slice = slice(span_width - 1, S)
    span_features[:, end_slice] = span_sums / float(span_width)
    span_features[:, end_slice] *= valid_windows.unsqueeze(-1).to(x.dtype)
    coverage[:, end_slice] = valid_windows.to(torch.int64)
    span_counts = valid_windows.sum(dim=1).to(torch.int64)

    if gate_strength is not None:
        strength_prefix = torch.cat(
            [gate_strength.new_zeros(B, 1), gate_strength.cumsum(dim=1)],
            dim=1,
        )
        strength_sum = (
            strength_prefix[:, span_width:] - strength_prefix[:, :-span_width]
        )
        span_strength[:, end_slice] = (
            strength_sum / float(span_width)
        ) * valid_windows.to(x.dtype)
    else:
        span_strength[:, end_slice] = valid_windows.to(x.dtype)

    span_base = torch.arange(S, device=x.device, dtype=torch.int64).unfold(
        0, span_width, 1
    )
    expanded_positions = span_base.unsqueeze(0).expand(B, -1, -1)
    span_positions[:, end_slice] = torch.where(
        valid_windows.unsqueeze(-1),
        expanded_positions,
        span_positions[:, end_slice],
    )
    return span_features, span_positions, span_counts, coverage, span_strength


def _op_sparse_span_builder(module, inputs, config):
    """Build sparse fused pair/triplet features over informative token windows.

    Output shape is per-position ``[B, S, D]``: the span ending at token t
    is placed at output position t (matching the Python reference). The
    native ``sparse_span_extract_f32`` kernel packs valid spans densely
    into slots ``[0, span_counts[b])``, which is a different tensor layout
    and breaks downstream autoregressive causality — we scatter that packed
    output back to the per-position layout before returning.
    """
    x = inputs[0]
    span_width = max(1, min(int(config.get("span_width", 3)), x.shape[1]))
    fallback_behavior = str(config.get("fallback_behavior", "default_path"))
    keep_mask = x.abs().sum(dim=-1) > 1e-8
    if _c(x) and hasattr(aria_core, "sparse_span_extract_f32"):
        try:
            packed_features, span_positions, span_counts, coverage = (
                aria_core.sparse_span_extract_f32(
                    x, keep_mask.to(torch.int64).contiguous(), span_width
                )
            )
            # Scatter packed spans onto the per-position layout using each
            # span's end index. Spans with all-negative positions are the
            # unused-slot sentinels — ignored by the validity mask below.
            end_positions = span_positions[..., -1]  # [B, S]
            valid = end_positions >= 0
            span_features = torch.zeros_like(x)
            b_idx, k_idx = torch.where(valid)
            span_features[b_idx, end_positions[b_idx, k_idx]] = packed_features[
                b_idx, k_idx
            ]
            span_strength = coverage.to(x.dtype)
        except (ImportError, RuntimeError, AttributeError) as e:
            record_kernel_fallback("sparse_span_extract_f32", e)
            span_features, span_positions, span_counts, coverage, span_strength = (
                _build_sparse_spans(x, keep_mask, span_width)
            )
    else:
        span_features, span_positions, span_counts, coverage, span_strength = (
            _build_sparse_spans(x, keep_mask, span_width)
        )
    _record_routing_telemetry(
        module,
        1,
        torch.zeros((x.shape[0], x.shape[1]), device=x.device, dtype=torch.int64),
        keep_mask=keep_mask,
        routing_mode="sparse_span_builder",
        gate_type="single_token",
        span_type=f"sparse_{'triplet' if span_width >= 3 else 'pair' if span_width == 2 else 'single'}",
        sparse_span_count=span_counts,
        sparse_span_width=span_width,
        sparse_span_coverage_tokens=(coverage > 0),
        default_path_count=(coverage == 0),
        routed_token_count=(coverage > 0),
        route_strength=span_strength[coverage > 0],
        trace_payload=(
            {
                "fallback_behavior": fallback_behavior,
                "span_positions_sample": span_positions[0][span_positions[0, :, 0] >= 0]
                .detach()
                .cpu()
                .tolist()
                if span_counts.numel() > 0
                else [],
            }
            if _capture_routing_trace(module)
            else None
        ),
    )
    return span_features


def _op_hybrid_sparse_router(module, inputs, config):
    """Two-stage routed execution: single-token gate then sparse fused-span lane routing."""
    x = inputs[0]
    span_width = max(1, min(int(config.get("span_width", 3)), x.shape[1]))
    lane_count = max(2, min(int(config.get("lane_count", 3)), 8))
    confidence_threshold = scheduled_scalar(
        module,
        config,
        key="confidence_threshold",
        default=0.45,
    )
    min_keep_fraction = scheduled_scalar(
        module,
        config,
        key="min_keep_fraction",
        default=0.125,
    )
    route_temperature = scheduled_scalar(
        module,
        config,
        key="route_temperature",
        default=1.0,
    )
    if hasattr(module, "hybrid_gate_proj"):
        gate_scores = _safe_linear(x, module.hybrid_gate_proj.to(x.dtype)).squeeze(-1)
    else:
        gate_scores = _routing_scores_from_x(x)
    token_present = x.abs().sum(dim=-1) > 1e-8
    keep_target = _minimum_keep_target(x.shape[1], span_width, min_keep_fraction)
    if _c(x) and hasattr(aria_core, "token_gate_trace_f32"):
        try:
            keep_mask_i64, gate_conf = aria_core.token_gate_trace_f32(
                gate_scores.contiguous(), confidence_threshold
            )
            keep_mask = keep_mask_i64.bool() & token_present
            min_keep_count = _minimum_keep_count(token_present, keep_target)
            keep_mask = keep_mask | _rescue_keep_mask(
                gate_conf,
                token_present,
                confidence_threshold,
                min_keep_count,
                keep_target,
            )
        except (ImportError, RuntimeError, AttributeError) as e:
            record_kernel_fallback("token_gate_trace_f32", e)
            gate_conf = torch.sigmoid(gate_scores)
            min_keep_count = _minimum_keep_count(token_present, keep_target)
            keep_mask = _rescue_keep_mask(
                gate_conf,
                token_present,
                confidence_threshold,
                min_keep_count,
                keep_target,
            )
    else:
        gate_conf = torch.sigmoid(gate_scores)
        min_keep_count = _minimum_keep_count(token_present, keep_target)
        keep_mask = _rescue_keep_mask(
            gate_conf,
            token_present,
            confidence_threshold,
            min_keep_count,
            keep_target,
        )

    span_features, span_positions, span_counts, coverage, span_strength = (
        _build_sparse_spans(
            x,
            keep_mask,
            span_width,
            gate_strength=gate_conf * token_present.to(x.dtype),
        )
    )
    default_out = (
        _safe_linear(x, module.hybrid_default_proj.to(x.dtype))
        if hasattr(module, "hybrid_default_proj")
        else x
    )
    if hasattr(module, "hybrid_lane_proj"):
        lane_logits = _safe_linear(span_features, module.hybrid_lane_proj.to(x.dtype))
    else:
        lane_logits = torch.stack(
            [span_features.mean(dim=-1) + i * 0.1 for i in range(lane_count)], dim=-1
        )
    lane_logits = lane_logits / max(route_temperature, 1e-4)
    lane_probs = F.softmax(lane_logits, dim=-1)
    lane_assignments = lane_probs.argmax(dim=-1)
    lane_conf = lane_probs.max(dim=-1).values
    valid_span_mask = span_positions[..., 0] >= 0
    active_route_mask = valid_span_mask & keep_mask
    lane_hist = torch.bincount(
        lane_assignments[valid_span_mask].reshape(-1), minlength=lane_count
    ).to(torch.float32)
    if (
        hasattr(module, "hybrid_lane_weights")
        and len(module.hybrid_lane_weights) >= lane_count
    ):
        lane_weights = torch.stack(
            [module.hybrid_lane_weights[i].to(x.dtype) for i in range(lane_count)],
            dim=0,
        )
        lane_outputs = torch.einsum("bsd,lod->bslo", span_features, lane_weights)
    else:
        lane_outputs = span_features.unsqueeze(2).expand(-1, -1, lane_count, -1)
    confidence_scale = torch.clamp(
        (lane_conf - confidence_threshold) / max(1.0 - confidence_threshold, 1e-4),
        min=0.0,
        max=1.0,
    )
    route_scale = span_strength * torch.where(
        active_route_mask,
        torch.maximum(confidence_scale, lane_conf),
        torch.zeros_like(lane_conf),
    )
    routed_updates = (
        lane_outputs
        * lane_probs.unsqueeze(-1)
        * route_scale.unsqueeze(-1).unsqueeze(-1)
    ).sum(dim=2)
    out = default_out + routed_updates

    span_type = f"sparse_{'triplet' if span_width >= 3 else 'pair' if span_width == 2 else 'single'}"
    _record_routing_telemetry(
        module,
        lane_count,
        lane_assignments,
        logits=lane_logits,
        keep_mask=keep_mask,
        lane_histogram=lane_hist,
        routing_mode="hybrid_sparse_router",
        gate_type="single_token",
        span_type=span_type,
        sparse_span_count=span_counts,
        sparse_span_width=span_width,
        sparse_span_coverage_tokens=(coverage > 0),
        default_path_count=(~active_route_mask),
        routed_token_count=active_route_mask,
        route_strength=route_scale[active_route_mask],
        lane_count=lane_count,
        trace_payload=(
            {
                "keep_mask_sample": keep_mask[0].detach().cpu().to(torch.int64).tolist()
                if keep_mask.numel() > 0
                else [],
                "lane_assignments_sample": lane_assignments[0].detach().cpu().tolist()
                if lane_assignments.numel() > 0
                else [],
            }
            if _capture_routing_trace(module)
            else None
        ),
    )
    return out


def _op_lane_conditioned_block(module, inputs, config):
    x = inputs[0]
    if hasattr(module, "lane_block_weight"):
        return F.gelu(_safe_linear(x, module.lane_block_weight.to(x.dtype)))
    return F.gelu(x)


def _op_default_path(module, inputs, config):
    return inputs[0]


def _op_depth_weighted_proj(module, inputs, config):
    """Learned adaptive recursion: per-token depth from learned scorer, per-step transforms."""
    x = inputs[0]
    configured_depth = int(config.get("max_depth", 3))
    max_depth = scheduled_int(
        module,
        config,
        key="active_depth",
        default=configured_depth,
        minimum=1,
        maximum=max(1, configured_depth),
    )
    max_depth = max(1, min(6, max_depth))

    if hasattr(module, "depth_scorer"):
        # Learned depth scoring: (B, S, D) → (B, S, max_depth)
        dt = x.dtype
        depth_logits = _safe_linear(x, module.depth_scorer.to(dt))
        depth_weights = F.softmax(depth_logits, dim=-1)  # (B, S, max_depth)
        _record_routing_telemetry(
            module, max_depth, depth_weights.argmax(dim=-1), logits=depth_logits
        )
        # Apply per-step transforms weighted by depth probability — batched einsum
        if hasattr(module, "step_projs") and len(module.step_projs) >= max_depth:
            W_all = _get_stacked_params(module, "step_projs", max_depth, dt)
            all_outs = torch.einsum("bsd,kod->bsko", x, W_all)
            return (depth_weights.unsqueeze(-1) * all_outs).sum(dim=2)
        return x
    # Fallback for legacy models without learned params
    scores = _routing_scores_from_x(x)
    depth_scores = torch.stack([scores + (i * 0.1) for i in range(max_depth)], dim=-1)
    depth = depth_scores.argmax(dim=-1) + 1
    scale = 1.0 + 0.05 * depth.float()
    return x * scale.unsqueeze(-1)


def _op_padic_depth_route(module, inputs, config):
    """Novel non-softmax recursion routing (replaces depth_weighted_proj's softmax scorer).

    Per-token recursion depth is chosen by the token's INTRINSIC p-adic valuation — an
    ultrametric / tree-distance signal — reciprocal-weighted (inverse-distance, NOT softmax)
    over ``max_depth`` learnable depth anchors. Because the routing signal is the token's
    fixed ultrametric structure rather than a free learned linear gate, the weights track the
    real spread of token hierarchy and structurally resist the single-expert collapse the
    softmax depth-router suffers (6/8 routers collapsed at scale, 2026-06-29). Per-step
    transforms are the same ``step_projs`` mixture as ``depth_weighted_proj``.
    """
    from ..mathspaces.padic import padic_valuation

    x = inputs[0]
    if not hasattr(module, "step_projs") or not hasattr(module, "depth_anchors"):
        return x
    dt = x.dtype
    k = min(max(1, int(config.get("max_depth", 3))), len(module.step_projs))
    # intrinsic ultrametric routing signal, standardized for activation-scale robustness
    val = padic_valuation(x.float()).mean(dim=-1)  # (B, S)
    val = (val - val.mean()) / (val.std() + 1e-5)
    anchors = module.depth_anchors[:k].to(val.dtype)  # (k,)
    sharp = F.softplus(module.route_log_sharpness.to(val.dtype)) + 0.5
    dist = (val.unsqueeze(-1) - anchors).abs()  # (B, S, k)
    # bounded reciprocal (Cauchy/Lorentzian) proximity in [0,1] — inverse-distance, non-softmax,
    # gradient-safe as dist -> 0 (unlike a raw dist**-sharp pole).
    inv = 1.0 / (1.0 + (dist * sharp).pow(2))
    weights = (inv / inv.sum(dim=-1, keepdim=True)).to(dt)  # (B, S, k)
    _record_routing_telemetry(module, k, weights.argmax(dim=-1))
    W_all = _get_stacked_params(module, "step_projs", k, dt)
    all_outs = torch.einsum("bsd,kod->bsko", x, W_all)
    return (weights.unsqueeze(-1) * all_outs).sum(dim=2)


def _op_padic_gated_mixer(module, inputs, config):
    """Learned p-adic gated mixer (the LEARNED counterpart to the fixed padic_gate).

    Highway/GLU mix between a learned projection and identity, gated by a LEARNED function of
    both the token and its p-adic valuation (log-magnitude / ultrametric scale signal):
        g = sigmoid(Wg x + Wv valuation(x) + b);  out = g * (Wp x) + (1 - g) * x
    Why this and not the fixed padic_gate or the p-adic *router*: the fixed-valuation router kept
    the gate alive but killed induction (0.477->0.006, 2026-06-29) because a rigid signal carries no
    task information. Here the gate is LEARNED (recovers capability) while the valuation term gives
    it scale/hierarchy structure a linear gate can't see; sigmoid per-channel (no softmax expert
    competition) makes it structurally collapse-proof. Degenerates to a plain highway gate if the
    model drives Wv->0, so it is never worse than a learned GLU.
    """
    from ..mathspaces.padic import padic_valuation

    x = inputs[0]
    if not hasattr(module, "gate_x"):
        return x
    dt = x.dtype
    val = padic_valuation(x.float()).clamp(-10.0, 10.0).to(dt)
    gate = torch.sigmoid(
        _safe_linear(x, module.gate_x)
        + _safe_linear(val, module.gate_v)
        + module.gate_bias.to(dt)
    )
    proj = _safe_linear(x, module.proj_w)
    return gate * proj + (1.0 - gate) * x


def _op_sinkhorn_ot_mix(module, inputs, config):
    """Novel non-QKV optimal-transport sequence mixer (entropic Sinkhorn / Wasserstein).

    Each token's projected query is transported onto every key under an entropic
    optimal-transport plan found by log-domain Sinkhorn iterations with UNIFORM BALANCED
    marginals (row mass == column mass == 1/S). This is structurally distinct from softmax
    attention: softmax yields row-stochastic weights with NO column constraint (mass piles onto
    a few keys -> the multi-slot binding interference wall), whereas a balanced OT plan enforces
    a doubly-stochastic transport budget that pushes distinct queries toward distinct keys.
    Transport is the geometry of matching; the balanced marginal is what makes it bind.

    Cost = squared Euclidean distance between L2-normalized query/key projections (content-driven,
    scale-bounded in [0, 4]); the Gaussian-style kernel exp(-cost/eps) keeps similar content
    cheaper to transport. eps is LEARNED (softplus-floored) so the model slides from soft-mean
    transport (eps -> inf) to near-hard assignment (eps -> 0). The column-marginal constraint
    forbids single-key collapse, so the plan stays non-degenerate. Python-only dispatch (no
    native softmax bypass); out = Wo(plan @ V) + x.
    """
    x = inputs[0]
    if not hasattr(module, "ot_q_proj"):
        return x
    dt = x.dtype
    B, S, D = x.shape
    q = torch.matmul(x, module.ot_q_proj.to(dt))
    k = torch.matmul(x, module.ot_k_proj.to(dt))
    v = torch.matmul(x, module.ot_v_proj.to(dt))
    # L2-normalize so the transport cost is bounded in [0, 4].
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    cost = (q.unsqueeze(2) - k.unsqueeze(1)).pow(2).sum(dim=-1)  # (B, S, S)
    eps = F.softplus(module.sinkhorn_log_eps.to(dt)) + 0.1
    log_K = -cost / eps
    n_iters = max(1, min(32, int(config.get("sinkhorn_iters", 8))))
    log_uniform = torch.log(torch.tensor(1.0 / S, dtype=dt, device=x.device))
    f = torch.zeros(B, S, dtype=dt, device=x.device)
    g = torch.zeros(B, S, dtype=dt, device=x.device)
    for _ in range(n_iters):
        f = log_uniform - torch.logsumexp(log_K + g.unsqueeze(1), dim=2)
        g = log_uniform - torch.logsumexp(log_K + f.unsqueeze(2), dim=1)
    log_T = f.unsqueeze(2) + log_K + g.unsqueeze(1)
    T = torch.exp(log_T)  # balanced doubly-stochastic within Sinkhorn tolerance
    module._last_ot_plan = T.detach()  # for novelty-guard address-entropy + tests
    o = torch.bmm(T, v)  # (B, S, D): transport-mass-weighted read of values
    return torch.matmul(o, module.ot_o_proj.to(dt)) + x


def _op_ultrametric_tree_mix(module, inputs, config):
    """Novel non-QKV content-addressed ultrametric (p-adic / Bruhat-Tits-tree) sequence mixer.

    Extends the validated padic direction from PER-TOKEN valuation (a scalar intrinsic signal,
    padic_gated_mixer) to PAIRWISE ultrametric tree-distance between tokens — the content-addressed
    retrieval pathway the single-pass padic gate lacks (rdr_padic 42.7M floor-result: reasoning at
    floor because a single-pass gate has no associative retrieval). Query i reads from each earlier
    key j weighted by an ultrametric kernel, NOT by softmax (exp of an additive bilinear):

        K_ij = prod_{l=1..L} sigmoid((tau_l - |(q_i - k_j) . w_l|) / temp)

    The PRODUCT over scales enforces nested agreement: a pair must agree at EVERY resolution to keep
    mass, so a single hard scale-disagreement drives K_ij -> 0. That is the strong triangle inequality
    d(x,z) <= max(d(x,y), d(y,z)), which gives the similarity graph a hierarchical-tree (Bruhat-Tits)
    topology rather than softmax's flat Euclidean exp-geometry. Softmax-over-dot-product provably
    cannot replicate this: when two keys have equal additive score against a query, softmax gives them
    equal mass, but the ultrametric product still separates them by scale structure. Causal (look-back
    only, j <= i), row-normalized; out = Wo(K_norm @ V) + x. Python-only dispatch (no native softmax
    bypass).
    """
    x = inputs[0]
    if not hasattr(module, "ut_q_proj"):
        return x
    dt = x.dtype
    B, S, D = x.shape
    q = torch.matmul(x, module.ut_q_proj.to(dt))  # (B, S, D)
    k = torch.matmul(x, module.ut_k_proj.to(dt))
    v = torch.matmul(x, module.ut_v_proj.to(dt))
    dirs = module.ut_scale_dirs.to(dt)  # (L=8, D)
    bias = module.ut_scale_bias.to(dt)  # (L=8,)
    temp = F.softplus(module.ut_scale_log_temp.to(dt)) + 0.1
    diff = q.unsqueeze(2) - k.unsqueeze(1)  # (B, S, S, D): query i vs key j
    proj = torch.matmul(diff, dirs.t())  # (B, S, S, L): signed per-scale difference
    # Per-scale soft agreement: ~1 when |diff.w_l| << tau_l, ~0 when >> tau_l.
    agree = torch.sigmoid((bias - proj.abs()) / temp)  # (B, S, S, L)
    # Ultrametric kernel = product across scales (nested agreement -> strong triangle inequality).
    K = agree.prod(dim=-1)  # (B, S, S)
    # Causal look-back: forbid future keys (j > i).
    causal = _cached_tril(S, x.device, dt)
    K = K * causal
    denom = K.sum(dim=2, keepdim=True).clamp_min(1e-6)
    W = K / denom  # row-normalized causal retrieval weights
    module._last_ut_weights = W.detach()  # for novelty-guard address-entropy + tests
    o = torch.bmm(W, v)  # (B, S, D): ultrametric-weighted read of values
    return torch.matmul(o, module.ut_o_proj.to(dt)) + x


def _op_fno_spectral_mix(module, inputs, config):
    """Novel non-QKV Fourier neural-operator (FNO) sequence mixer.

    Learns a COMPLEX low-pass spectral filter across the sequence axis: rfft the (projected) token
    sequence, apply a per-low-mode complex channel weight, zero the high Fourier modes (learned
    low-pass truncation), irfft back. This is a GLOBAL linear operator applied in frequency space
    (O(S log S)) — every output token depends on every input token via the Fourier basis, in one op.
    Structurally != softmax: no score-weighted aggregation, no exp/normalize — mixing is complex
    multiplication on retained low modes, not exp-of-additive-bilinear. Different geometry from
    optimal transport (sinkhorn_ot_mix) and the ultrametric tree (ultrametric_tree_mix): spectral /
    function-space mixing, targeting long-range (W2) gaps with param-efficient frequency selectivity.

        Xf = rfft(Wi x)                       # (B, S//2+1, D) complex
        Yf[:, :k] = Xf[:, :k] * (Wr + i Wi)   # per-low-mode complex channel weight, high modes -> 0
        out = Wo(irfft(Yf) + b) + x

    FFT runs in float32 (precision) and casts back. Python-only dispatch (no native softmax bypass).
    """
    x = inputs[0]
    if not hasattr(module, "fno_in_proj"):
        return x
    dt = x.dtype
    B, S, D = x.shape
    h = torch.matmul(
        x, module.fno_in_proj.to(dt)
    ).float()  # (B, S, D); float32 for FFT precision
    Xf = torch.fft.rfft(h, dim=1, norm="ortho")  # (B, S//2+1, D) complex64
    k = min(
        module.fno_modes_real.shape[0], Xf.shape[1]
    )  # keep only available low modes
    Wr = module.fno_modes_real[:k].float()  # (k, D, D)
    Wi = module.fno_modes_imag[:k].float()
    Xr = Xf.real[:, :k, :]  # (B, k, D)
    Xi = Xf.imag[:, :k, :]
    # Per-mode complex channel mixing: Y = Xf * (Wr + i Wi)  (complex matmul on the channel axis).
    Yr = torch.einsum("bkd,kde->bke", Xr, Wr) - torch.einsum("bkd,kde->bke", Xi, Wi)
    Yi = torch.einsum("bkd,kde->bke", Xr, Wi) + torch.einsum("bkd,kde->bke", Xi, Wr)
    # Reassemble the full spectrum: low modes filtered, high modes zeroed (low-pass truncation).
    pad = Xf.shape[1] - k
    if pad > 0:
        zr = torch.zeros(B, pad, D, device=x.device, dtype=Yr.dtype)
        zi = torch.zeros(B, pad, D, device=x.device, dtype=Yi.dtype)
        full_real = torch.cat([Yr, zr], dim=1)
        full_imag = torch.cat([Yi, zi], dim=1)
    else:
        full_real, full_imag = Yr, Yi
    y = torch.fft.irfft(
        torch.complex(full_real, full_imag), n=S, dim=1, norm="ortho"
    )  # (B, S, D)
    y = (y + module.fno_bias.float()).to(dt)
    return torch.matmul(y, module.fno_out_proj.to(dt)) + x


def _causal_lag(x: torch.Tensor, lag: int) -> torch.Tensor:
    if lag <= 0:
        return x
    if lag >= x.shape[1]:
        return torch.zeros_like(x)
    return F.pad(x[:, :-lag], (0, 0, lag, 0))


def _dct_basis(seq_len: int, n_modes: int, device, dtype) -> torch.Tensor:
    key = (int(seq_len), int(n_modes), _device_cache_key(device), dtype)
    cached = _cache_get(_DCT_BASIS_CACHE, key)
    if cached is not None:
        return cached
    pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    mode = torch.arange(n_modes, device=device, dtype=dtype).unsqueeze(0)
    basis = torch.cos(math.pi * (pos + 0.5) * mode / float(seq_len))
    basis[:, 0] *= seq_len**-0.5
    if n_modes > 1:
        basis[:, 1:] *= (2.0 / float(seq_len)) ** 0.5
    return _cache_put(_DCT_BASIS_CACHE, key, basis)


def _legendre_basis(seq_len: int, n_modes: int, device, dtype) -> torch.Tensor:
    key = (int(seq_len), int(n_modes), _device_cache_key(device), dtype)
    cached = _cache_get(_LEGENDRE_BASIS_CACHE, key)
    if cached is not None:
        return cached
    z = torch.linspace(-1.0, 1.0, seq_len, device=device, dtype=dtype)
    cols = [
        torch.ones_like(z),
        z,
        0.5 * (3.0 * z.square() - 1.0),
        0.5 * (5.0 * z.pow(3) - 3.0 * z),
    ][:n_modes]
    basis = torch.stack(cols, dim=1)
    basis = basis / basis.norm(dim=0, keepdim=True).clamp_min(1e-6)
    return _cache_put(_LEGENDRE_BASIS_CACHE, key, basis)


def _graph_eigbasis(seq_len: int, device: torch.device, decay: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device, dtype=torch.float32)
    dist = (idx[:, None] - idx[None, :]).abs()
    adj = torch.exp(-dist / decay)
    adj = adj - torch.diag_embed(torch.diagonal(adj))
    lap = torch.diag(adj.sum(dim=-1)) - adj
    jitter = torch.linspace(0.0, 1e-4, seq_len, device=device, dtype=torch.float32)
    lap = lap + torch.diag(jitter)
    _, basis = torch.linalg.eigh(lap)
    return basis


def _cached_graph_eigbasis(
    seq_len: int, device: torch.device, decay: torch.Tensor
) -> torch.Tensor:
    decay_key = round(float(decay.detach().cpu()), 6)
    key = (int(seq_len), _device_cache_key(device), decay_key)
    cached = _cache_get(_GRAPH_EIGBASIS_CACHE, key)
    if cached is not None:
        return cached
    return _cache_put(_GRAPH_EIGBASIS_CACHE, key, _graph_eigbasis(seq_len, device, decay))


def _basis_reconstruct(
    x: torch.Tensor, basis: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    k = min(basis.shape[1], scale.shape[0])
    b = basis[:, :k].float()
    coeff = torch.einsum("bsd,sk->bkd", x.float(), b)
    coeff = coeff * scale[:k].float().unsqueeze(0)
    return torch.einsum("sk,bkd->bsd", b, coeff).to(x.dtype)


def _op_causal_gradient_mix(module, inputs, _):
    x = inputs[0]
    if not hasattr(module, "grad_proj"):
        return x
    grad = x - _causal_lag(x, 1)
    update = _safe_linear(grad, module.grad_proj)
    gate = torch.sigmoid(module.grad_gate.to(dtype=x.dtype, device=x.device)).view(
        1, 1, -1
    )
    return x + gate * update


def _op_causal_laplacian_mix(module, inputs, _):
    x = inputs[0]
    if not hasattr(module, "lap_proj"):
        return x
    lap = x - 2.0 * _causal_lag(x, 1) + _causal_lag(x, 2)
    update = _safe_linear(lap, module.lap_proj)
    gate = torch.sigmoid(module.lap_gate.to(dtype=x.dtype, device=x.device)).view(
        1, 1, -1
    )
    return x + gate * update


def _op_lie_derivative_flow_mix(module, inputs, _):
    x = inputs[0]
    if not hasattr(module, "lie_flow_proj"):
        return x
    grad = x - _causal_lag(x, 1)
    flow = torch.tanh(_safe_linear(x, module.lie_flow_proj))
    update = _safe_linear(flow * grad, module.lie_value_proj)
    gate = torch.sigmoid(module.lie_gate.to(dtype=x.dtype, device=x.device)).view(
        1, 1, -1
    )
    return x + gate * update


def _op_dct_spectral_mix(module, inputs, _):
    x = inputs[0]
    if not hasattr(module, "dct_scale"):
        return x
    basis = _dct_basis(
        x.shape[1], module.dct_scale.shape[0], x.device, torch.float32
    )
    y = _basis_reconstruct(x, basis, module.dct_scale)
    return x + _safe_linear(y, module.dct_out_proj)


def _op_graph_eigbasis_mix(module, inputs, _):
    x = inputs[0]
    if not hasattr(module, "graph_eig_scale"):
        return x
    S = x.shape[1]
    decay = F.softplus(module.graph_eig_log_decay.float()) + 0.5
    if torch.is_grad_enabled() and decay.requires_grad:
        basis = _graph_eigbasis(S, x.device, decay)
    else:
        basis = _cached_graph_eigbasis(S, x.device, decay)
    y = _basis_reconstruct(x, basis, module.graph_eig_scale)
    return x + _safe_linear(y, module.graph_eig_out_proj)


def _op_legendre_basis_mix(module, inputs, _):
    x = inputs[0]
    if not hasattr(module, "legendre_scale"):
        return x
    basis = _legendre_basis(
        x.shape[1], module.legendre_scale.shape[0], x.device, torch.float32
    )
    y = _basis_reconstruct(x, basis, module.legendre_scale)
    return x + _safe_linear(y, module.legendre_out_proj)


def _op_cp_tensor_mix(module, inputs, _):
    x = inputs[0]
    if not hasattr(module, "cp_factor_a"):
        return x
    a = torch.einsum("bsd,rd->bsr", x, module.cp_factor_a.to(x.dtype))
    b = torch.einsum("bsd,rd->bsr", x, module.cp_factor_b.to(x.dtype))
    coeff = torch.tanh(a * b)
    y = torch.einsum("bsr,rd->bsd", coeff, module.cp_factor_c.to(x.dtype))
    return x + _safe_linear(y, module.cp_out_proj)


def _op_tensor_train_mix(module, inputs, _):
    x = inputs[0]
    if not hasattr(module, "tt_down"):
        return x
    h = _safe_linear(x, module.tt_down)
    h = _safe_linear(h, module.tt_core)
    y = _safe_linear(torch.tanh(h), module.tt_up)
    gate = torch.sigmoid(module.tt_gate.to(dtype=x.dtype, device=x.device)).view(
        1, 1, -1
    )
    return x + gate * y


def _op_tensor_ring_mix(module, inputs, _):
    x = inputs[0]
    if not hasattr(module, "tensor_ring_out_proj"):
        return x
    fwd = torch.roll(x, shifts=1, dims=-1) * module.tensor_ring_forward.to(x.dtype)
    back = torch.roll(x, shifts=-1, dims=-1) * module.tensor_ring_backward.to(x.dtype)
    return x + _safe_linear(fwd + back, module.tensor_ring_out_proj)


def _op_block_term_tensor_mix(module, inputs, _):
    x = inputs[0]
    if not hasattr(module, "block_term_gates"):
        return x
    chunks = torch.chunk(x, chunks=4, dim=-1)
    gates = module.block_term_gates.to(dtype=x.dtype, device=x.device)
    mixed_chunks = []
    start = 0
    for idx, chunk in enumerate(chunks):
        width = chunk.shape[-1]
        gate = torch.sigmoid(gates[idx, start : start + width]).view(1, 1, -1)
        mixed_chunks.append(chunk * gate)
        start += width
    y = torch.cat(mixed_chunks, dim=-1)
    return x + _safe_linear(y, module.block_term_out_proj)


def _op_alpha_divergence_mix(module, inputs, _):
    x = inputs[0]
    if not hasattr(module, "alpha_proj"):
        return x
    p = F.softplus(x.float()) + 1e-4
    q = F.softplus(_causal_lag(x, 1).float()) + 1e-4
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    alpha = 0.5 + 1.5 * torch.sigmoid(module.alpha_divergence_logit.float())
    moment = (p.pow(alpha) * q.pow(1.0 - alpha)).sum(dim=-1, keepdim=True)
    div = (moment - 1.0) / (alpha * (alpha - 1.0)).clamp_min(1e-4)
    signal = ((p - q) * torch.tanh(div)).to(x.dtype)
    update = _safe_linear(signal, module.alpha_proj)
    gate = torch.sigmoid(module.alpha_gate.to(dtype=x.dtype, device=x.device)).view(
        1, 1, -1
    )
    return x + gate * update


def _op_renyi_attention_mix(module, inputs, _):
    x = inputs[0]
    if not hasattr(module, "renyi_q_proj"):
        return x
    q = _safe_linear(x, module.renyi_q_proj)
    k = _safe_linear(x, module.renyi_k_proj)
    v = _safe_linear(x, module.renyi_v_proj)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(max(1, x.shape[-1]))
    S = x.shape[1]
    mask = _cached_tril(S, x.device, torch.bool)
    positive = F.softplus(scores.float()) + 1e-4
    positive = positive.masked_fill(~mask, 0.0)
    power = 1.0 + F.softplus(module.renyi_q_delta.float())
    weights = positive.pow(power)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    y = torch.matmul(weights.to(x.dtype), v)
    return x + _safe_linear(y, module.renyi_o_proj)


def _op_natural_gradient_mixer(module, inputs, _):
    x = inputs[0]
    if not hasattr(module, "natural_grad_proj"):
        return x
    h = _safe_linear(x, module.natural_grad_proj).float()
    centered = h - h.mean(dim=1, keepdim=True)
    B, S, D = centered.shape
    cov = torch.matmul(centered.transpose(1, 2), centered) / float(max(1, S))
    eye = torch.eye(D, device=x.device, dtype=torch.float32).expand(B, D, D)
    damping = F.softplus(module.natural_grad_log_damping.float()) + 1e-3
    solved = torch.linalg.solve(cov + damping * eye, centered.transpose(1, 2))
    y = solved.transpose(1, 2).to(x.dtype)
    return x + _safe_linear(y, module.natural_grad_out_proj)


def _op_dyadic_diff_mix(module, inputs, _):
    x = inputs[0]
    if not hasattr(module, "dyadic_scales"):
        return x
    acc = torch.zeros_like(x)
    for idx, lag in enumerate((1, 2, 4)):
        diff = x - _causal_lag(x, lag)
        scale = module.dyadic_scales[idx].to(dtype=x.dtype, device=x.device).view(
            1, 1, -1
        )
        acc = acc + diff * scale
    return x + _safe_linear(acc, module.dyadic_out_proj)


def _op_laplacian_pyramid_mix(module, inputs, _):
    x = inputs[0]
    if not hasattr(module, "pyramid_scales"):
        return x
    smooth = x
    acc = torch.zeros_like(x)
    for idx in range(3):
        next_smooth = 0.5 * (smooth + _causal_lag(smooth, 1))
        detail = smooth - next_smooth
        scale = module.pyramid_scales[idx].to(dtype=x.dtype, device=x.device).view(
            1, 1, -1
        )
        acc = acc + detail * scale
        smooth = next_smooth
    return x + _safe_linear(acc, module.pyramid_out_proj)


# ── Exotic Ops (Phase 4) ─────────────────────────────────────────────


def _op_difficulty_blend_3way(module, inputs, config):
    """Routes tokens to 'fast' vs 'deep' lanes based on learned difficulty."""
    x = inputs[0]
    B, S, D = x.shape

    if not hasattr(module, "gate_proj"):
        return x

    # Compute 3-way gate: [Fast, Medium, Hard]
    dt = x.dtype
    logits = _safe_linear(x, module.gate_proj.to(dt))
    weights = F.softmax(logits, dim=-1)

    _record_routing_telemetry(module, 3, weights.argmax(dim=-1), logits=logits)

    # Experts: 0=Identity(Fast), 1=LowRank(Medium), 2=MLP(Hard)
    out = x * weights[..., 0:1]  # Fast lane: direct skip

    # Medium lane: Low-rank
    if hasattr(module, "U_mid"):
        mid = _safe_linear(_safe_linear(x, module.U_mid.to(dt)), module.V_mid.to(dt))
        out = out + mid * weights[..., 1:2]

    # Hard lane: MLP
    if hasattr(module, "heavy_mlp"):
        hard = module.heavy_mlp(x)
        out = out + hard * weights[..., 2:3]

    return out


def _op_score_depth_blend(module, inputs, config):
    """Tokens re-enter block with different parameters each recursion.
    Depth is conditional on input difficulty score (inputs[1]).
    """
    x, scores = inputs[0], inputs[1]
    configured_depth = int(config.get("max_depth", 3))
    max_depth = scheduled_int(
        module,
        config,
        key="active_depth",
        default=configured_depth,
        minimum=1,
        maximum=max(1, configured_depth),
    )

    if not hasattr(module, "step_projs"):
        return x

    # Determine depth per token from scores
    depths = scores.argmax(dim=-1)  # [B, S] in range [0, max_depth-1]

    out = x
    dt = out.dtype
    # Pre-fetch and cast projection weights outside loop
    step_weights = [module.step_projs[i].to(dt) for i in range(max_depth)]
    # Precompute all depth masks at once: (max_depth, B, S, 1)
    depth_thresholds = torch.arange(max_depth, device=x.device).view(-1, 1, 1)
    all_masks = (
        (depths.unsqueeze(0) >= depth_thresholds).float().unsqueeze(-1)
    )  # (K, B, S, 1)
    # Sequential application (each step depends on previous out)
    for i in range(max_depth):
        step_out = _safe_linear(out, step_weights[i])
        out = out + all_masks[i] * (step_out - out) * 0.5

    _record_routing_telemetry(module, max_depth, depths, logits=scores)
    return out


def _op_calibrated_branch_merge(module, inputs, config):
    if len(inputs) != 2:
        logger.warning(
            "calibrated_branch_merge got %d inputs, expected 2 — falling back to identity",
            len(inputs),
        )
        return inputs[0]
    branches = [inputs[0], inputs[1]]
    dt = branches[0].dtype
    normalize_inputs = bool(config.get("normalize_inputs", True))
    merge_temperature = max(1e-4, float(config.get("merge_temperature", 1.0)))

    branch_norms = [branch_rms(branch) for branch in branches]
    if normalize_inputs:
        normalized = [branch / norm for branch, norm in zip(branches, branch_norms)]
    else:
        normalized = branches

    score_proj = getattr(module, "branch_score_proj", None)
    branch_bias = getattr(module, "branch_bias", None)
    scores = []
    for idx, branch in enumerate(normalized):
        if score_proj is not None and idx < score_proj.shape[0]:
            score = _safe_linear(branch, score_proj[idx : idx + 1].to(dt)).squeeze(-1)
        else:
            score = branch.float().mean(dim=-1).to(dt)
        if branch_bias is not None and idx < branch_bias.numel():
            score = score + branch_bias[idx].to(dt)
        scores.append(score)
    logits = torch.stack(scores, dim=-1) / merge_temperature
    weights = F.softmax(logits, dim=-1)

    min_secondary_share = float(config.get("min_secondary_share", 0.15))
    max_secondary_share = float(config.get("max_secondary_share", 0.5))
    if bool(config.get("curriculum_enabled", False)):
        min_secondary_share = scheduled_scalar(
            module,
            config,
            key="min_secondary_share",
            default=min_secondary_share,
        )
        max_secondary_share = scheduled_scalar(
            module,
            config,
            key="max_secondary_share",
            default=max_secondary_share,
        )
    secondary_weight = torch.clamp(
        weights[..., 1],
        min=min_secondary_share,
        max=max_secondary_share,
    )
    primary_weight = 1.0 - secondary_weight
    weights = torch.stack([primary_weight, secondary_weight], dim=-1)

    gains = getattr(module, "branch_gain", None)
    if gains is not None:
        gains_t = 0.5 + 1.0 * torch.sigmoid(gains.to(dt))
    else:
        gains_t = torch.ones(2, device=branches[0].device, dtype=dt)
    anchor = branch_norms[0]
    out = torch.zeros_like(branches[0])
    for idx, branch in enumerate(normalized):
        out = out + branch * weights[..., idx : idx + 1] * gains_t[idx]
    out = out * anchor

    weight_mean = weights.mean(dim=(0, 1))
    dominance = weights.max(dim=-1).values
    primary_role = str(config.get("primary_role", "primary"))
    secondary_role = str(config.get("secondary_role", "secondary"))
    _record_routing_telemetry(
        module,
        2,
        weights.argmax(dim=-1),
        logits=logits,
        routing_mode="calibrated_branch_merge",
        gate_type="branch_merge",
        branch_weights=weight_mean.detach(),
        branch_dominance=dominance,
        routed_branch_share=weights[..., 0],
        medium_branch_share=weights[..., 0],
        hard_branch_share=weights[..., 1],
        trace_payload=(
            {
                "branch_names": [primary_role, secondary_role],
                "primary_role": primary_role,
                "secondary_role": secondary_role,
                "curriculum_stage": stage_name(
                    get_routing_progress(module),
                    float(config.get("curriculum_warmup_frac", 0.25)),
                    float(config.get("curriculum_mid_frac", 0.65)),
                ),
            }
            if _capture_routing_trace(module)
            else None
        ),
    )
    return out


def _op_token_entropy(module, inputs, config):
    """Compute Shannon entropy of input scores as a difficulty signal (B,S,K) → (B,S,1).

    Uses log_softmax for numerical stability and temperature scaling
    to prevent gradient vanishing from softmax saturation.
    """
    scores = inputs[0]
    temperature = max(scores.shape[-1] ** 0.5, 1.0)
    log_probs = F.log_softmax(scores / temperature, dim=-1)
    probs = log_probs.exp()
    entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
    return entropy


def _op_relu_gated_moe(module, inputs, config):
    """ReLU-gated MoE (ReMoE): learned gate activates variable expert count per token."""
    x = inputs[0]
    if not hasattr(module, "gate_proj"):
        return x
    dt = x.dtype
    n_experts = module.gate_proj.shape[0]

    # Learned ReLU gate with load balancing: sparse activation
    raw_logits = _safe_linear(x, module.gate_proj.to(dt))
    raw_logits = _apply_moe_load_balance(module, raw_logits, n_experts)
    gate_scores = F.relu(raw_logits)  # (B, S, n_experts)
    # Normalize non-zero gates to sum to 1
    gate_sum = gate_scores.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    gate_weights = gate_scores / gate_sum  # (B, S, n_experts)

    _record_routing_telemetry(
        module, n_experts, gate_scores.argmax(dim=-1), logits=gate_scores
    )

    # Dispatch to learned expert projections — batched via einsum
    if hasattr(module, "expert_weights"):
        W_all = _get_stacked_params(module, "expert_weights", n_experts, dt)
        expert_outs = torch.einsum("bsd,eod->bseo", x, W_all)
        return (gate_weights.unsqueeze(-1) * expert_outs).sum(dim=2)
    # Fallback for legacy models: scale by total gate activation
    return gate_scores.sum(dim=-1, keepdim=True).expand_as(x) * x


def _op_pq_embedding_moe_block(module, inputs, _):
    """Factorized Semantic Bottleneck MoE: PQ denoised routing."""
    x = inputs[0]
    if hasattr(module, "block"):
        return module.block(x)
    return x


OP_IMPLS: Dict[str, Callable] = {
    "topk_gate": _op_topk_gate,
    "moe_topk": _op_moe_topk,
    "pq_embedding_moe_block": _op_pq_embedding_moe_block,
    "moe_2expert": _op_moe_2expert,
    "swiglu_mlp": _op_swiglu_mlp,
    "feature_sparsity": _op_feature_sparsity,
    "gated_lane_blend": _op_gated_lane_blend,
    "depth_gated_transform": _op_depth_gated_transform,
    "adjacent_token_merge": _op_adjacent_token_merge,
    "depth_token_mask": _op_depth_token_mask,
    "confidence_token_gate": _op_confidence_token_gate,
    "learned_token_gate": _op_learned_token_gate,
    "cheap_verify_blend": _op_cheap_verify_blend,
    "calibrated_branch_merge": _op_calibrated_branch_merge,
    "hybrid_token_gate": _op_hybrid_token_gate,
    "sparse_span_builder": _op_sparse_span_builder,
    "hybrid_sparse_router": _op_hybrid_sparse_router,
    "lane_conditioned_block": _op_lane_conditioned_block,
    "default_path": _op_default_path,
    "depth_weighted_proj": _op_depth_weighted_proj,
    "padic_depth_route": _op_padic_depth_route,
    "padic_gated_mixer": _op_padic_gated_mixer,
    "sinkhorn_ot_mix": _op_sinkhorn_ot_mix,
    "ultrametric_tree_mix": _op_ultrametric_tree_mix,
    "fno_spectral_mix": _op_fno_spectral_mix,
    "causal_gradient_mix": _op_causal_gradient_mix,
    "causal_laplacian_mix": _op_causal_laplacian_mix,
    "lie_derivative_flow_mix": _op_lie_derivative_flow_mix,
    "dct_spectral_mix": _op_dct_spectral_mix,
    "graph_eigbasis_mix": _op_graph_eigbasis_mix,
    "legendre_basis_mix": _op_legendre_basis_mix,
    "cp_tensor_mix": _op_cp_tensor_mix,
    "tensor_train_mix": _op_tensor_train_mix,
    "tensor_ring_mix": _op_tensor_ring_mix,
    "block_term_tensor_mix": _op_block_term_tensor_mix,
    "alpha_divergence_mix": _op_alpha_divergence_mix,
    "renyi_attention_mix": _op_renyi_attention_mix,
    "natural_gradient_mixer": _op_natural_gradient_mixer,
    "dyadic_diff_mix": _op_dyadic_diff_mix,
    "laplacian_pyramid_mix": _op_laplacian_pyramid_mix,
    "token_entropy": _op_token_entropy,
    "relu_gated_moe": _op_relu_gated_moe,
    "difficulty_blend_3way": _op_difficulty_blend_3way,
    "score_depth_blend": _op_score_depth_blend,
    # Backward-compatible aliases for old op names
    "route_topk": _op_feature_sparsity,
    "route_lanes": _op_gated_lane_blend,
    "route_recursion": _op_depth_gated_transform,
    "relu_gate_routing": _op_relu_gated_moe,
    "token_merge": _op_adjacent_token_merge,
    "mod_topk": _op_depth_token_mask,
    "early_exit": _op_confidence_token_gate,
    "cascade": _op_learned_token_gate,
    "speculative": _op_cheap_verify_blend,
    "adaptive_recursion": _op_depth_weighted_proj,
    "entropy_score": _op_token_entropy,
    "adaptive_lane_mixer": _op_difficulty_blend_3way,
    "mixed_recursion_gate": _op_score_depth_blend,
}
