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

def _op_topk_gate(module, inputs, _):
    if not hasattr(module, 'gate_proj'): return inputs[0]
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
    logits = F.linear(x, module.gate_proj)
    gate_weights = F.softmax(logits, dim=-1)
    
    # Record routing telemetry
    _record_routing_telemetry(module, 2, gate_weights.argmax(dim=-1), logits=logits)
    
    half = D // 2
    out = torch.cat([x[..., :half] * gate_weights[..., 0:1], 
                     x[..., half:2*half] * gate_weights[..., 1:2]], dim=-1)
    if D > 2 * half:
        out = torch.cat([out, x[..., 2*half:]], dim=-1)
    return out

def _op_moe_topk(module, inputs, config):
    """Sparse Mixture-of-Experts channel mixer."""
    x = inputs[0]
    B, S, D = x.shape
    
    n_experts = int(config.get("num_experts", 4))
    top_k = int(config.get("top_k", 2))
    
    if not hasattr(module, 'gate_weight'):
        return x
        
    logits = F.linear(x, module.gate_weight)
    weights, indices = logits.topk(top_k, dim=-1)
    weights = F.softmax(weights, dim=-1)
    
    # Record routing telemetry
    _record_routing_telemetry(module, n_experts, indices, logits=logits)
    
    # Recorded weights for expert contribution
    if hasattr(module, 'experts'):
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
                exp_mask = (indices == i)
                expert_weight = weights[exp_mask].reshape(-1, 1)
                output[mask] = output[mask] + expert(expert_input) * expert_weight
    else:
        # Fallback to a learned projection if experts sub-modules aren't ready
        output = F.linear(x, module.weight) if hasattr(module, 'weight') else x

    return output

def _op_moe_2expert(module, inputs, config):
    """Lightweight 2-expert MoE with learned gating."""
    x = inputs[0]
    B, S, D = x.shape

    if not hasattr(module, 'gate_proj'):
        return x

    # Compute gate scores
    logits = F.linear(x, module.gate_proj)  # (B, S, 2)
    weights = F.softmax(logits, dim=-1)     # (B, S, 2)

    # Record routing telemetry
    _record_routing_telemetry(module, 2, weights.argmax(dim=-1), logits=logits)

    # Each expert is a simple linear projection
    e0 = F.linear(x, module.expert_0_weight)  # (B, S, D)
    e1 = F.linear(x, module.expert_1_weight)  # (B, S, D)

    # Weighted combination
    output = weights[..., 0:1] * e0 + weights[..., 1:2] * e1
    return output

def _op_swiglu_mlp(module, inputs, _):
    """SwiGLU MLP channel mixer."""
    x = inputs[0]
    if not hasattr(module, 'gate_proj'):
        return x
    if _c(x) and x.dim() >= 2:
        x2, orig = _flatten_for_kernel(x)
        y = aria_core.swiglu_f32(
            x2, module.gate_proj.weight, module.up_proj.weight, module.down_proj.weight,
            getattr(module.gate_proj, 'bias', None),
            getattr(module.up_proj, 'bias', None),
            getattr(module.down_proj, 'bias', None),
        )
        return _unflatten_from_kernel(y, orig)
    return module.down_proj(F.silu(module.gate_proj(x)) * module.up_proj(x))

def _op_route_topk(module, inputs, config):
    """Top-k routing: zero out all but top-k positions along last dim.

    Input:  (B, S, D)
    Output: (B, S, D) with only the top-k values per (B, S) slice kept.
    """
    x = inputs[0]
    k = min(int(config.get("k", 1)), x.shape[-1])
    topk_vals, topk_idx = x.topk(k, dim=-1)  # (B, S, k)
    _record_routing_telemetry(module, x.shape[-1], topk_idx, logits=x)
    # Build sparse mask and scatter top-k values back
    mask = torch.zeros_like(x)
    mask.scatter_(-1, topk_idx, 1.0)
    # STE: forward uses hard mask, backward passes through
    return x * (mask.detach() - x.detach() + x)

def _op_route_lanes(module, inputs, config):
    scores = inputs[0] # (B, S, L)
    if HAS_ARIA_CORE and scores.device.type == "cpu" and scores.dtype == torch.float32:
        lane_indices = aria_core.route_lane_argmax_f32(scores)
        _record_routing_telemetry(module, scores.shape[2], lane_indices, logits=scores)
        return lane_indices
    # Fallback
    lane_indices = scores.argmax(dim=-1)
    _record_routing_telemetry(module, scores.shape[2], lane_indices, logits=scores)
    return lane_indices

def _op_route_recursion(module, inputs, config):
    scores = inputs[0] # (B, S, Dp)
    max_depth = scores.shape[-1]
    if HAS_ARIA_CORE and scores.device.type == "cpu" and scores.dtype == torch.float32:
        depth = aria_core.route_recursion_depth_f32(scores)
    else:
        # Fallback
        depth = scores.argmax(dim=-1) + 1
    _record_routing_telemetry(module, max_depth, depth, logits=scores)
    return depth

def _op_token_merge(module, inputs, config):
    x = inputs[0]
    n_keep = int(config.get("n_keep", x.shape[1] // 2))
    seq_len = x.shape[1]
    if HAS_ARIA_CORE and x.device.type == "cpu" and x.dtype == torch.float32:
        y, restore_map = aria_core.token_merge_simple_f32(x, n_keep)
    else:
        # Fallback: simple truncation
        y = x[:, :n_keep, :]
        restore_map = torch.arange(seq_len, device=x.device).expand(x.shape[0], -1)
    # Record merge telemetry — tokens_processed = n_keep, tokens_total = seq_len
    merge_telem = getattr(module, "routing_telemetry", {
        "tokens_total": 0, "tokens_processed": 0,
        "merge_kept": 0, "merge_dropped": 0,
        "expert_counts": torch.zeros(1, device=x.device),
        "entropy_sum": 0.0, "count": 0, "heatmap": None,
    })
    B = x.shape[0]
    merge_telem["tokens_total"] += B * seq_len
    merge_telem["tokens_processed"] += B * n_keep
    merge_telem["merge_kept"] = merge_telem.get("merge_kept", 0) + B * n_keep
    merge_telem["merge_dropped"] = merge_telem.get("merge_dropped", 0) + B * (seq_len - n_keep)
    merge_telem["count"] = merge_telem.get("count", 0) + 1
    setattr(module, "routing_telemetry", merge_telem)
    # Restore to original length with causal-safe indexing:
    # Position i can only map to kept positions <= i (never look ahead)
    B_size, S_orig = restore_map.shape
    causal_limit = torch.arange(S_orig, device=restore_map.device).unsqueeze(0).expand(B_size, -1)
    causal_limit = causal_limit.clamp(max=y.shape[1] - 1)
    restore_map = restore_map.clamp(0, y.shape[1] - 1).minimum(causal_limit)
    return y.gather(1, restore_map.unsqueeze(-1).expand(-1, -1, x.shape[2]))

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
    _record_routing_telemetry(module, S,
                              (gate > 0.5).nonzero(as_tuple=False)[:, 1:],
                              logits=scores)
    return x * gate.unsqueeze(-1)

def _op_early_exit(module, inputs, config):
    x = inputs[0]
    threshold = float(config.get("threshold", 0.5))
    scores = _routing_scores_from_x(x)
    gate = torch.sigmoid(scores)
    keep = (gate > threshold).float()
    _record_routing_telemetry(module, 2, keep.long(), logits=gate)
    return x * keep.unsqueeze(-1)

def _op_cascade(module, inputs, config):
    x = inputs[0]
    threshold = float(config.get("threshold", 0.5))
    scores = _routing_scores_from_x(x)
    gate = torch.sigmoid(scores)
    _record_routing_telemetry(module, 2, (gate > threshold).long(), logits=gate)
    return x * gate.unsqueeze(-1)

def _op_speculative(module, inputs, config):
    x = inputs[0]
    threshold = float(config.get("threshold", 0.5))
    scores = _routing_scores_from_x(x)
    gate = torch.sigmoid(scores)
    keep = (gate > threshold).float()
    _record_routing_telemetry(module, 2, keep.long(), logits=gate)
    # Mild effect: scale rather than drop
    return x * (0.5 + 0.5 * gate).unsqueeze(-1)

def _op_adaptive_recursion(module, inputs, config):
    x = inputs[0]
    max_depth = int(config.get("max_depth", 3))
    max_depth = max(1, min(6, max_depth))
    scores = _routing_scores_from_x(x)
    depth_scores = torch.stack([scores + (i * 0.1) for i in range(max_depth)], dim=-1)
    if HAS_ARIA_CORE and depth_scores.device.type == "cpu" and depth_scores.dtype == torch.float32:
        depth = aria_core.route_recursion_depth_f32(depth_scores)
    else:
        depth = depth_scores.argmax(dim=-1) + 1
    scale = 1.0 + 0.05 * depth.float()
    return x * scale.unsqueeze(-1)

def _op_token_merging(module, inputs, config):
    x = inputs[0]
    n_keep = int(config.get("n_keep", x.shape[1] // 2))
    seq_len = x.shape[1]
    if HAS_ARIA_CORE and x.device.type == "cpu" and x.dtype == torch.float32:
        y, restore_map = aria_core.token_merge_simple_f32(x, n_keep)
    else:
        y = x[:, :n_keep, :]
        restore_map = torch.arange(seq_len, device=x.device).expand(x.shape[0], -1)
    # Ensure indices are within bounds for CUDA safety
    restore_map = restore_map.clamp(0, y.shape[1] - 1)
    return y.gather(1, restore_map.unsqueeze(-1).expand(-1, -1, x.shape[2]))

# ── Exotic Ops (Phase 4) ─────────────────────────────────────────────

def _op_adaptive_lane_mixer(module, inputs, config):
    """Routes tokens to 'fast' vs 'deep' lanes based on learned difficulty."""
    x = inputs[0]
    B, S, D = x.shape
    
    if not hasattr(module, 'gate_proj'): return x
    
    # Compute 3-way gate: [Fast, Medium, Hard]
    logits = F.linear(x, module.gate_proj)
    weights = F.softmax(logits, dim=-1)
    
    _record_routing_telemetry(module, 3, weights.argmax(dim=-1), logits=logits)
    
    # Experts: 0=Identity(Fast), 1=LowRank(Medium), 2=MLP(Hard)
    out = x * weights[..., 0:1] # Fast lane: direct skip
    
    # Medium lane: Low-rank
    if hasattr(module, 'U_mid'):
        mid = F.linear(F.linear(x, module.U_mid), module.V_mid)
        out = out + mid * weights[..., 1:2]
        
    # Hard lane: MLP
    if hasattr(module, 'heavy_mlp'):
        hard = module.heavy_mlp(x)
        out = out + hard * weights[..., 2:3]
        
    return out

def _op_mixed_recursion_gate(module, inputs, config):
    """Tokens re-enter block with different parameters each recursion.
    Depth is conditional on input difficulty score (inputs[1]).
    """
    x, scores = inputs[0], inputs[1]
    max_depth = int(config.get("max_depth", 3))
    
    if not hasattr(module, 'step_projs'): return x
    
    # Determine depth per token from scores
    depths = scores.argmax(dim=-1) # [B, S] in range [0, max_depth-1]
    
    out = x
    # Current implementation: sequential application up to max_depth
    # But only tokens whose depth >= current step get the update
    for i in range(max_depth):
        mask = (depths >= i).float().unsqueeze(-1)
        # Apply transformation for this step
        # proj: (D, D) or similar
        step_out = F.linear(out, module.step_projs[i])
        out = (1 - mask) * out + mask * step_out
        
    _record_routing_telemetry(module, max_depth, depths, logits=scores)
    return out

def _op_entropy_router(module, inputs, config):
    """Produces routing signal [B, S, 1] based on entropy of input scores (B,S,K)."""
    scores = inputs[0]
    probs = F.softmax(scores, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1, keepdim=True)
    return entropy

def _op_relu_gate_routing(module, inputs, config):
    """ReLU-based differentiable MoE gating (ReMoE).
    Unlike Top-K, this learns how many experts to activate per token.
    """
    x = inputs[0]
    if not hasattr(module, 'gate_proj'): return x
    
    # [B, S, n_experts]
    gate_scores = F.relu(F.linear(x, module.gate_proj))
    
    # Record telemetry on 'effective expert count' (sparsity)
    active_count = (gate_scores > 0).float().sum(dim=-1).mean().item()
    _record_routing_telemetry(module, gate_scores.shape[-1], gate_scores.argmax(dim=-1), logits=gate_scores)
    
    # Placeholder: In a real MoE this would dispatch to experts.
    # For micro-eval, we just return the weighted gate signal.
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
    "token_merging": _op_token_merging,
    "entropy_router": _op_entropy_router,
    "relu_gate_routing": _op_relu_gate_routing,
    "adaptive_lane_mixer": _op_adaptive_lane_mixer,
    "mixed_recursion_gate": _op_mixed_recursion_gate,
}
