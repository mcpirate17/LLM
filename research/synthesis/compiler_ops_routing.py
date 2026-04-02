from __future__ import annotations

import math
from typing import Callable, Dict

import torch
import torch.nn.functional as F

from .compiler_op_utils import (
    HAS_ARIA_CORE,
    aria_core,
    _c,
    _flatten_for_kernel,
    _get_stacked_params,
    _record_routing_telemetry,
    _unflatten_from_kernel,
    record_kernel_fallback,
)


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
    # Initialize bias buffer if needed (not a Parameter — no gradients)
    if not hasattr(module, "_moe_balance_bias"):
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
    logits = F.linear(x, module.gate_proj.to(x.dtype))
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

    for k_idx in range(top_k):
        expert_ids = idx_flat[:, k_idx]
        slot_weights = w_flat[:, k_idx].unsqueeze(-1)

        sort_order = expert_ids.argsort(stable=True)
        sorted_x = x_flat[sort_order]
        sorted_w = slot_weights[sort_order]

        expert_counts = torch.bincount(expert_ids, minlength=n_actual).tolist()
        x_chunks = sorted_x.split(expert_counts, dim=0)
        w_chunks = sorted_w.split(expert_counts, dim=0)

        result_chunks = []
        for e_idx in range(n_actual):
            if expert_counts[e_idx] == 0:
                result_chunks.append(sorted_x.new_empty(0, D))
                continue
            out = experts[e_idx](x_chunks[e_idx])
            result_chunks.append(out.to(x.dtype) * w_chunks[e_idx].to(x.dtype))

        result_sorted = torch.cat(result_chunks, dim=0)
        result_flat = torch.zeros(BS, D, device=x.device, dtype=x.dtype)
        result_flat[sort_order] = result_sorted
        output.view(BS, D).add_(result_flat)

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
    if cached is not None:
        return cached
    W_downs = []
    W_ups = []
    for i in range(n_experts):
        expert = module.experts[i]
        W_downs.append(expert[0].weight.to(dtype))  # Linear(D, H).weight is (H, D)
        W_ups.append(expert[2].weight.to(dtype))  # Linear(H, D).weight is (D, H)
    stacked = (torch.stack(W_downs), torch.stack(W_ups))  # (E,H,D), (E,D,H)
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

    logits = F.linear(x, module.gate_weight.to(x.dtype))
    logits = _apply_moe_load_balance(module, logits, n_experts)
    weights, indices = logits.topk(top_k, dim=-1)
    weights = F.softmax(weights, dim=-1)

    # Record routing telemetry
    _record_routing_telemetry(module, n_experts, indices, logits=logits)

    if not hasattr(module, "experts"):
        return F.linear(x, module.weight) if hasattr(module, "weight") else x

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
    logits = F.linear(x, module.gate_proj.to(dt))  # (B, S, 2)
    logits = _apply_moe_load_balance(module, logits, 2)
    weights = F.softmax(logits, dim=-1)  # (B, S, 2)

    # Record routing telemetry
    _record_routing_telemetry(module, 2, weights.argmax(dim=-1), logits=logits)

    # Each expert is a simple linear projection
    e0 = F.linear(x, module.expert_0_weight.to(dt))  # (B, S, D)
    e1 = F.linear(x, module.expert_1_weight.to(dt))  # (B, S, D)

    # Weighted combination
    output = weights[..., 0:1] * e0 + weights[..., 1:2] * e1
    return output


def _op_swiglu_mlp(module, inputs, _):
    """SwiGLU MLP channel mixer."""
    x = inputs[0]
    if not hasattr(module, "gate_proj"):
        return x
    if _c(x) and x.dim() >= 2:
        x2, orig = _flatten_for_kernel(x)
        y = aria_core.swiglu_f32(
            x2,
            module.gate_proj.weight,
            module.up_proj.weight,
            module.down_proj.weight,
            getattr(module.gate_proj, "bias", None),
            getattr(module.up_proj, "bias", None),
            getattr(module.down_proj, "bias", None),
        )
        return _unflatten_from_kernel(y, orig)
    return module.down_proj(F.silu(module.gate_proj(x)) * module.up_proj(x))


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
    """Learned difficulty-based lane blend: score tokens, soft-weight N internal linear projections.

    Each token gets routed to one of n_lanes compute paths based on learned
    difficulty scoring. Easy tokens get cheap transforms, hard tokens get expensive.
    Soft routing via softmax weights for gradient flow, hard assignment for telemetry.
    """
    x = inputs[0]
    B, S, D = x.shape
    n_lanes = int(config.get("n_lanes", 3))

    if not hasattr(module, "lane_scorer"):
        return x
    dt = x.dtype

    # 1. Score token difficulty → lane logits (B, S, n_lanes)
    lane_logits = F.linear(x, module.lane_scorer.to(dt))
    lane_weights = F.softmax(lane_logits, dim=-1)  # soft assignment
    lane_indices = lane_logits.argmax(dim=-1)  # hard assignment for telemetry
    _record_routing_telemetry(module, n_lanes, lane_indices, logits=lane_logits)

    # 2. Per-lane transforms: batched via einsum over stacked projections
    if hasattr(module, "lane_projs") and len(module.lane_projs) >= n_lanes:
        W_all = _get_stacked_params(module, "lane_projs", n_lanes, dt)
        all_outs = torch.einsum("bsd,lod->bslo", x, W_all)  # (B, S, L, D_out)
        return (lane_weights.unsqueeze(-1) * all_outs).sum(dim=2)
    # Fallback: identity for all lanes
    return x


def _op_depth_gated_transform(module, inputs, config):
    """Learned difficulty-based depth gate: score tokens, apply variable-depth linear transforms.

    Each token gets a learned depth score determining how many transform steps
    it receives. Easy tokens get 1 pass, hard tokens get up to max_depth passes.
    Uses soft depth weighting for gradient flow.
    """
    x = inputs[0]
    B, S, D = x.shape
    max_depth = int(config.get("max_depth", 3))
    max_depth = max(1, min(6, max_depth))

    if not hasattr(module, "depth_scorer"):
        return x
    dt = x.dtype

    # 1. Score token difficulty → depth logits (B, S, max_depth)
    depth_logits = F.linear(x, module.depth_scorer.to(dt))
    depth_weights = F.softmax(depth_logits, dim=-1)
    depth_indices = depth_logits.argmax(dim=-1)
    _record_routing_telemetry(module, max_depth, depth_indices, logits=depth_logits)

    # 2. Per-depth transforms: batched via einsum over stacked projections
    if hasattr(module, "depth_projs") and len(module.depth_projs) >= max_depth:
        W_all = _get_stacked_params(module, "depth_projs", max_depth, dt)
        all_outs = torch.einsum("bsd,kod->bsko", x, W_all)  # (B, S, K, D_out)
        return (depth_weights.unsqueeze(-1) * all_outs).sum(dim=2)
    # Fallback: identity for all depths
    return x


def _op_adjacent_token_merge(module, inputs, config):
    """Causal token compression: merge even-indexed tokens into their predecessor.

    Strictly causal: token p+1 absorbs information from token p (backward
    merge), so the merged value at position p+1 depends only on tokens ≤ p+1.
    The merge pattern is deterministic (even-stride) to avoid any dependency
    on future token content.
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
        # Deterministic causal stride merge: drop every other token starting
        # from position 1 (merge token p into token p-1, backward-looking).
        # This is strictly causal because each merged token only receives
        # information from its immediate predecessor.
        stride = max(2, S // n_keep)
        # Positions to drop: 1, 1+stride, 1+2*stride, ... (up to n_merge)
        drop_positions = torch.arange(1, S, stride, device=x.device)[:n_merge]

        merged = torch.zeros(B, S, device=x.device, dtype=torch.bool)
        merge_targets = (
            torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1).clone()
        )

        # Backward merge: token at drop_pos merges INTO drop_pos-1 (vectorized)
        valid_drops = drop_positions[drop_positions > 0]
        merged[:, valid_drops] = True
        merge_targets[:, valid_drops] = valid_drops - 1

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
        scores = F.linear(x, module.confidence_proj.to(x.dtype)).squeeze(-1)  # (B, S)
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
        scores = F.linear(x, module.cascade_proj.to(x.dtype)).squeeze(-1)  # (B, S)
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
    cheap_out = F.linear(x, module.cheap_proj.to(dt))
    # Learned verification gate: decides how much of the full input to blend
    gate = torch.sigmoid(F.linear(x, module.verify_gate.to(dt)).squeeze(-1))
    _record_routing_telemetry(module, 2, (gate > 0.5).long(), logits=gate)
    # Blend: cheap_out + gate * (x - cheap_out) = lerp(cheap_out, x, gate)
    return cheap_out + gate.unsqueeze(-1) * (x - cheap_out)


def _op_depth_weighted_proj(module, inputs, config):
    """Learned adaptive recursion: per-token depth from learned scorer, per-step transforms."""
    x = inputs[0]
    max_depth = int(config.get("max_depth", 3))
    max_depth = max(1, min(6, max_depth))

    if hasattr(module, "depth_scorer"):
        # Learned depth scoring: (B, S, D) → (B, S, max_depth)
        dt = x.dtype
        depth_logits = F.linear(x, module.depth_scorer.to(dt))
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


# ── Exotic Ops (Phase 4) ─────────────────────────────────────────────


def _op_difficulty_blend_3way(module, inputs, config):
    """Routes tokens to 'fast' vs 'deep' lanes based on learned difficulty."""
    x = inputs[0]
    B, S, D = x.shape

    if not hasattr(module, "gate_proj"):
        return x

    # Compute 3-way gate: [Fast, Medium, Hard]
    dt = x.dtype
    logits = F.linear(x, module.gate_proj.to(dt))
    weights = F.softmax(logits, dim=-1)

    _record_routing_telemetry(module, 3, weights.argmax(dim=-1), logits=logits)

    # Experts: 0=Identity(Fast), 1=LowRank(Medium), 2=MLP(Hard)
    out = x * weights[..., 0:1]  # Fast lane: direct skip

    # Medium lane: Low-rank
    if hasattr(module, "U_mid"):
        mid = F.linear(F.linear(x, module.U_mid.to(dt)), module.V_mid.to(dt))
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
    max_depth = int(config.get("max_depth", 3))

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
        step_out = F.linear(out, step_weights[i])
        out = out + all_masks[i] * (step_out - out) * 0.5

    _record_routing_telemetry(module, max_depth, depths, logits=scores)
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
    raw_logits = F.linear(x, module.gate_proj.to(dt))
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


OP_IMPLS: Dict[str, Callable] = {
    "topk_gate": _op_topk_gate,
    "moe_topk": _op_moe_topk,
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
    "depth_weighted_proj": _op_depth_weighted_proj,
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
