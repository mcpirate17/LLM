from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn.functional as F

from .compiler_op_utils import (
    HAS_ARIA_CORE,
    aria_core,
    _c,
    _flatten_for_kernel,
    _record_routing_telemetry,
    _unflatten_from_kernel,
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
            counts = torch.zeros(n_experts, device=logits.device)
            for i in range(n_experts):
                counts[i] = (selected == i).float().sum()
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
        except Exception:
            pass
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

    # Recorded weights for expert contribution
    output = torch.zeros_like(x)
    if hasattr(module, "experts"):
        for i, expert in enumerate(module.experts):
            # Find tokens that selected this expert
            # indices shape: (B, S, top_k)
            # mask: (B, S) - tokens that have expert i in their top-k
            mask = (indices == i).any(dim=-1)
            if mask.any():
                expert_input = x[mask]
                # expert_weight: find the weight assigned to expert i for these tokens
                # We need to extract the specific weight from 'weights' (B, S, top_k)
                # where indices (B, S, top_k) == i
                exp_mask = indices == i
                expert_weight = weights[exp_mask].reshape(-1, 1)
                output[mask] = output[mask] + expert(expert_input).to(
                    output.dtype
                ) * expert_weight.to(output.dtype)
    else:
        # Fallback to a learned projection if experts sub-modules aren't ready
        output = F.linear(x, module.weight) if hasattr(module, "weight") else x

    return output


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


def _op_route_topk(module, inputs, config):
    """Top-k routing: zero out all but top-k positions along last dim.

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
    # Sqrt scaling compensates for sparsity without amplifying gradients excessively
    scale = (D / k) ** 0.5
    return x * (mask.detach() - x.detach() + x) * scale


def _op_route_lanes(module, inputs, config):
    """Learned difficulty-based lane routing: score tokens, assign to lanes, per-lane transforms.

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

    # 2. Per-lane transforms: each lane has its own learned projection
    out = torch.zeros_like(x)
    for i in range(n_lanes):
        if hasattr(module, "lane_projs") and i < len(module.lane_projs):
            lane_out = F.linear(x, module.lane_projs[i].to(dt))
        else:
            lane_out = x
        out = out + lane_weights[..., i : i + 1] * lane_out

    return out


def _op_route_recursion(module, inputs, config):
    """Learned difficulty-based recursion depth: score tokens, apply variable-depth transforms.

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

    # 2. Per-depth transforms: cumulative application weighted by depth probability
    out = torch.zeros_like(x)
    for i in range(max_depth):
        if hasattr(module, "depth_projs") and i < len(module.depth_projs):
            step_out = F.linear(x, module.depth_projs[i].to(dt))
        else:
            step_out = x
        out = out + depth_weights[..., i : i + 1] * step_out

    return out


def _op_token_merge(module, inputs, config):
    """Similarity-based token merging (ToMe-style): merge most similar adjacent pairs."""
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
        y, restore_map = aria_core.token_merge_simple_f32(x, n_keep)
    else:
        # Cosine similarity between adjacent token pairs (causal-safe)
        x_norm = F.normalize(x.detach(), dim=-1)
        # Similarity of token i with token i+1 (only look backward/adjacent, not ahead)
        sim = (x_norm[:, :-1, :] * x_norm[:, 1:, :]).sum(dim=-1)  # (B, S-1)
        # Find pairs to merge: pick top n_merge most similar adjacent pairs
        # Greedy: mark merged positions, skip already-merged neighbors
        merged = torch.zeros(B, S, device=x.device, dtype=torch.bool)
        merge_targets = torch.zeros(B, S, device=x.device, dtype=torch.long)
        merge_targets[:] = torch.arange(S, device=x.device).unsqueeze(0)

        # Sort similarities descending per batch, greedily merge non-overlapping pairs
        sim_sorted, sim_idx = sim.sort(dim=-1, descending=True)
        for b in range(B):
            count = 0
            for j in range(S - 1):
                idx = sim_idx[b, j].item()
                if count >= n_merge:
                    break
                if not merged[b, idx] and not merged[b, idx + 1]:
                    merged[b, idx + 1] = True
                    merge_targets[b, idx + 1] = idx
                    count += 1

        # Build merged output: average merged pairs, keep unmerged tokens
        kept_mask = ~merged  # (B, S)
        # Use scatter_add for autograd-safe merging instead of in-place mutation
        # Start with a copy, then blend merged tokens via differentiable indexing
        weights = torch.ones(B, S, 1, device=x.device, dtype=x.dtype)
        target_idx = merge_targets.unsqueeze(-1).expand(-1, -1, D)  # (B, S, D)
        # Accumulate: for merged positions, add their values to the target position
        out = x.scatter_add(1, target_idx, x * merged.unsqueeze(-1).float())
        # Count how many tokens map to each position (1 for kept, 2 for merge targets)
        count_map = weights.scatter_add(
            1, merge_targets.unsqueeze(-1), merged.unsqueeze(-1).float()
        )
        out = out / count_map.clamp(min=1)

        # Gather only kept tokens
        # Build gather indices for kept positions
        kept_indices = []
        for b in range(B):
            idx = kept_mask[b].nonzero(as_tuple=False).squeeze(-1)
            # Pad to n_keep if needed
            if idx.shape[0] < n_keep:
                pad = idx[-1:].expand(n_keep - idx.shape[0])
                idx = torch.cat([idx, pad])
            kept_indices.append(idx[:n_keep])
        kept_indices = torch.stack(kept_indices)  # (B, n_keep)
        y = out.gather(1, kept_indices.unsqueeze(-1).expand(-1, -1, D))

    # Record merge telemetry
    merge_telem = getattr(
        module,
        "routing_telemetry",
        {
            "tokens_total": 0,
            "tokens_processed": 0,
            "merge_kept": 0,
            "merge_dropped": 0,
            "expert_counts": torch.zeros(1, device=x.device),
            "entropy_sum": 0.0,
            "count": 0,
            "heatmap": None,
        },
    )
    merge_telem["tokens_total"] += B * S
    merge_telem["tokens_processed"] += B * n_keep
    merge_telem["merge_kept"] = merge_telem.get("merge_kept", 0) + B * n_keep
    merge_telem["merge_dropped"] = merge_telem.get("merge_dropped", 0) + B * n_merge
    merge_telem["count"] = merge_telem.get("count", 0) + 1
    setattr(module, "routing_telemetry", merge_telem)

    # Restore to original seq length via causal-safe nearest-kept mapping.
    # For position i in the original sequence, map to the nearest kept
    # position <= i (causal: never look ahead). This is better than
    # broadcasting the last kept token into all dropped positions.
    if use_c_kernel:
        # C kernel path: kept tokens are first n_keep (simple truncation)
        restore_map = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        restore_map = restore_map.clamp(max=n_keep - 1)
    else:
        # Python path: kept positions are scattered, build per-batch map
        restore_map = torch.zeros(B, S, device=x.device, dtype=torch.long)
        for b in range(B):
            ki = kept_indices[b]  # (n_keep,) — sorted original positions
            ptr = 0
            for s in range(S):
                # Advance pointer while next kept position <= s
                while ptr < n_keep - 1 and ki[ptr + 1] <= s:
                    ptr += 1
                restore_map[b, s] = ptr
    return y.gather(1, restore_map.unsqueeze(-1).expand(-1, -1, D))


# ── Routing Control Ops (Phase 2) ────────────────────────────────────


def _op_mod_topk(module, inputs, config):
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


def _op_early_exit(module, inputs, config):
    """Learned early-exit: tokens with low confidence are attenuated."""
    x = inputs[0]
    threshold = float(config.get("threshold", 0.5))
    if hasattr(module, "confidence_proj"):
        scores = F.linear(x, module.confidence_proj.to(x.dtype)).squeeze(-1)  # (B, S)
    else:
        scores = _routing_scores_from_x(x)
    gate = torch.sigmoid(scores)
    keep = (gate > threshold).float()
    # STE: hard gate forward, soft gate backward
    gate_ste = keep - gate.detach() + gate
    _record_routing_telemetry(module, 2, keep.long(), logits=gate)
    return x * gate_ste.unsqueeze(-1)


def _op_cascade(module, inputs, config):
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


def _op_speculative(module, inputs, config):
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


def _op_adaptive_recursion(module, inputs, config):
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
        # Apply per-step transforms weighted by depth probability
        out = torch.zeros_like(x)
        for i in range(max_depth):
            if hasattr(module, "step_projs") and i < len(module.step_projs):
                step_out = F.linear(x, module.step_projs[i].to(dt))
            else:
                step_out = x
            out = out + depth_weights[..., i : i + 1] * step_out
        return out
    # Fallback for legacy models without learned params
    scores = _routing_scores_from_x(x)
    depth_scores = torch.stack([scores + (i * 0.1) for i in range(max_depth)], dim=-1)
    depth = depth_scores.argmax(dim=-1) + 1
    scale = 1.0 + 0.05 * depth.float()
    return x * scale.unsqueeze(-1)


# ── Exotic Ops (Phase 4) ─────────────────────────────────────────────


def _op_adaptive_lane_mixer(module, inputs, config):
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


def _op_mixed_recursion_gate(module, inputs, config):
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
    # Sequential application up to max_depth with residual per step.
    # Tokens whose depth >= current step get the update; others are unchanged.
    for i in range(max_depth):
        mask = (depths >= i).float().unsqueeze(-1)
        step_out = F.linear(out, module.step_projs[i].to(out.dtype))
        # Residual per step: prevents vanishing gradients through deep recursion
        out = out + mask * (step_out - out) * 0.5

    _record_routing_telemetry(module, max_depth, depths, logits=scores)
    return out


def _op_entropy_score(module, inputs, config):
    """Compute Shannon entropy of input scores as a difficulty signal (B,S,K) → (B,S,1)."""
    scores = inputs[0]
    probs = F.softmax(scores, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1, keepdim=True)
    return entropy


def _op_relu_gate_routing(module, inputs, config):
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

    # Dispatch to learned expert projections
    if hasattr(module, "expert_weights"):
        output = torch.zeros_like(x)
        for i in range(n_experts):
            w = gate_weights[..., i : i + 1]  # (B, S, 1)
            expert_out = F.linear(x, module.expert_weights[i].to(dt))
            output = output + w * expert_out
        return output
    # Fallback for legacy models: scale by total gate activation
    return gate_scores.sum(dim=-1, keepdim=True).expand_as(x) * x


OP_IMPLS: Dict[str, Callable] = {
    "topk_gate": _op_topk_gate,
    "moe_topk": _op_moe_topk,
    "moe_2expert": _op_moe_2expert,
    "swiglu_mlp": _op_swiglu_mlp,
    "route_topk": _op_route_topk,
    "route_lanes": _op_route_lanes,
    "route_recursion": _op_route_recursion,
    "token_merge": _op_token_merge,
    "mod_topk": _op_mod_topk,
    "early_exit": _op_early_exit,
    "cascade": _op_cascade,
    "speculative": _op_speculative,
    "adaptive_recursion": _op_adaptive_recursion,
    "entropy_score": _op_entropy_score,
    "relu_gate_routing": _op_relu_gate_routing,
    "adaptive_lane_mixer": _op_adaptive_lane_mixer,
    "mixed_recursion_gate": _op_mixed_recursion_gate,
}
