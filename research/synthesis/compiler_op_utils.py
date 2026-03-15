from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from . import kernels
    HAS_KERNELS = True
except ImportError:
    HAS_KERNELS = False
    kernels = None

try:
    from . import cpu_ops
    HAS_CPU_OPS = True
except ImportError:
    HAS_CPU_OPS = False
    cpu_ops = None

try:
    from research.env import aria_core, HAS_ARIA_CORE
except ImportError:
    aria_core = None
    HAS_ARIA_CORE = False

def _record_sparse_telemetry(module: nn.Module, op_name: str, density: float,
                             fallback_reason: Optional[str] = None) -> None:
    telemetry = getattr(module, "sparse_telemetry", {})
    stats = telemetry.get(op_name, {
        "calls": 0,
        "fallback_calls": 0,
        "density_sum": 0.0,
        "last_density": 1.0,
        "last_fallback_reason": None,
    })
    stats["calls"] += 1
    stats["density_sum"] += float(density)
    stats["last_density"] = float(density)
    if fallback_reason is not None:
        stats["fallback_calls"] += 1
        stats["last_fallback_reason"] = fallback_reason
    telemetry[op_name] = stats
    setattr(module, "sparse_telemetry", telemetry)

def _record_routing_telemetry(module: nn.Module, n_experts: int, selected_experts: torch.Tensor,
                              logits: Optional[torch.Tensor] = None) -> None:
    """Record MoE routing statistics with lightweight sampling."""
    telemetry = getattr(module, "routing_telemetry", {
        "tokens_total": 0,
        "tokens_processed": 0,
        "expert_counts": torch.zeros(n_experts, device=selected_experts.device),
        "entropy_sum": 0.0,
        "count": 0,
        "heatmap": None,
        "_call_count": -1,
    })

    telemetry["_call_count"] += 1
    B, S = selected_experts.shape[:2]
    total_tokens = B * S
    telemetry["tokens_total"] += total_tokens
    telemetry["tokens_processed"] += total_tokens

    if telemetry["_call_count"] & 7 != 0:
        telemetry["count"] += 1
        setattr(module, "routing_telemetry", telemetry)
        return

    counts = torch.histc(selected_experts.float(), bins=n_experts, min=0, max=n_experts-1)
    telemetry["expert_counts"] += counts

    if logits is not None:
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
        telemetry["entropy_sum"] += entropy
        telemetry["count"] += 1

    if getattr(module, "_capture_heatmap", False) and telemetry["heatmap"] is None:
        telemetry["heatmap"] = selected_experts[0].detach().cpu().numpy().tolist()

    setattr(module, "routing_telemetry", telemetry)

def _build_nm_mask(weight: torch.Tensor, n: int, m: int) -> torch.Tensor:
    if n <= 0 or m <= 0 or n > m:
        return torch.ones_like(weight)
    
    if HAS_CPU_OPS and weight.device.type == "cpu" and weight.dtype == torch.float32:
        return cpu_ops.build_nm_mask_cpu(weight, n, m)

    rows, cols = weight.shape
    n_chunks = cols // m
    if n_chunks <= 0:
        return torch.ones_like(weight)

    usable = n_chunks * m
    core = weight[:, :usable].abs().reshape(rows, n_chunks, m)
    keep_idx = core.topk(k=n, dim=-1).indices
    mask_core = torch.zeros_like(core)
    mask_core.scatter_(-1, keep_idx, 1.0)
    mask = torch.ones_like(weight)
    mask[:, :usable] = mask_core.reshape(rows, usable)
    return mask

def _flatten_for_kernel(x: torch.Tensor):
    """Flatten >=3D tensor to 2D for C kernels that expect (batch, dim).

    Returns (x_2d, orig_shape) so the caller can reshape back via
    out.reshape(*orig_shape[:-1], -1).
    """
    if not isinstance(x, torch.Tensor):
        raise RuntimeError(f"_flatten_for_kernel expected Tensor, got {type(x).__name__}")
    if x.dim() < 1:
        raise RuntimeError(f"_flatten_for_kernel expected >=1D tensor, got {x.dim()}D")
    orig_shape = x.shape
    if x.dim() > 2:
        x = x.contiguous().reshape(-1, orig_shape[-1])
    elif not x.is_contiguous():
        x = x.contiguous()
    return x, orig_shape

def _unflatten_from_kernel(out: torch.Tensor, orig_shape):
    """Reshape 2D kernel output back to match the original input shape."""
    if len(orig_shape) > 2:
        return out.reshape(*orig_shape[:-1], -1)
    return out

def _build_block_sparse_mask(weight: torch.Tensor, block_size: int,
                             block_density: float) -> torch.Tensor:
    block_size = max(1, int(block_size))
    block_density = float(max(0.05, min(1.0, block_density)))

    rows, cols = weight.shape
    row_blocks = rows // block_size
    col_blocks = cols // block_size
    if row_blocks <= 0 or col_blocks <= 0:
        return torch.ones_like(weight)

    usable_rows = row_blocks * block_size
    usable_cols = col_blocks * block_size
    core = weight[:usable_rows, :usable_cols]
    blocks = core.view(row_blocks, block_size, col_blocks, block_size).permute(0, 2, 1, 3)
    scores = blocks.abs().mean(dim=(2, 3))

    keep_per_row = max(1, int(round(col_blocks * block_density)))
    keep_idx = scores.topk(k=keep_per_row, dim=1).indices

    block_mask = torch.zeros_like(scores)
    block_mask.scatter_(1, keep_idx, 1.0)
    block_mask = block_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, block_size, block_size)
    block_mask = block_mask.permute(0, 2, 1, 3).reshape(usable_rows, usable_cols)

    mask = torch.ones_like(weight)
    mask[:usable_rows, :usable_cols] = block_mask
    return mask


# ── Op Implementations ──────────────────────────────────────────────

def _c(x):
    """Check if tensor is eligible for aria_core C kernels.

    C kernels don't support autograd, so skip them when gradients are needed.
    Requires at least 2D tensor with reasonable dimensions.
    """
    return (HAS_ARIA_CORE and x.device.type == "cpu"
            and x.dtype == torch.float32 and not x.requires_grad
            and x.dim() >= 1 and x.numel() > 0)
