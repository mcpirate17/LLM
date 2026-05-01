from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from . import kernels

    HAS_KERNELS = True
except ImportError:
    HAS_KERNELS = False
    kernels = None

try:
    from research.env import aria_core, HAS_ARIA_CORE
except ImportError:
    aria_core = None
    HAS_ARIA_CORE = False

# Shared kernel fallback state for split op modules.
# compiler.py maintains its own copy; this one covers the compiler_ops_*.py files.
_kernel_fallback_occurred: bool = False
_kernel_fallback_logged: set = set()


def record_kernel_fallback(kernel_name: str, error: Exception) -> None:
    """Log a kernel fallback and set the module-level flag."""
    global _kernel_fallback_occurred
    _kernel_fallback_occurred = True
    if kernel_name not in _kernel_fallback_logged:
        _kernel_fallback_logged.add(kernel_name)
        logger.info(
            "kernel_fallback: kernel=%s reason=%s (further occurrences suppressed)",
            kernel_name,
            error,
        )


def kernel_fallback_occurred() -> bool:
    """Return whether any native kernel call fell back to Python this session."""
    return _kernel_fallback_occurred


def _safe_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """F.linear with automatic dtype casting to prevent bf16/f32 mismatch.

    During bf16 autocast, input x may be bf16 while module weights are f32
    (or vice versa after _cast_params_to). This wrapper ensures matching dtypes.
    Fixes 1,206 RuntimeError crashes in the training pipeline (71% of all RuntimeErrors).
    """
    if weight.dtype != x.dtype:
        weight = weight.to(x.dtype)
    if bias is not None and bias.dtype != x.dtype:
        bias = bias.to(x.dtype)
    return F.linear(x, weight, bias)


def _record_sparse_telemetry(
    module: nn.Module,
    op_name: str,
    density: float,
    fallback_reason: Optional[str] = None,
) -> None:
    telemetry = getattr(module, "sparse_telemetry", {})
    stats = telemetry.get(
        op_name,
        {
            "calls": 0,
            "fallback_calls": 0,
            "density_sum": 0.0,
            "last_density": 1.0,
            "last_fallback_reason": None,
        },
    )
    stats["calls"] += 1
    stats["density_sum"] += float(density)
    stats["last_density"] = float(density)
    if fallback_reason is not None:
        stats["fallback_calls"] += 1
        stats["last_fallback_reason"] = fallback_reason
    telemetry[op_name] = stats
    setattr(module, "sparse_telemetry", telemetry)


def _sparse_density_sampled(mask: torch.Tensor, module: nn.Module) -> float:
    """Compute mask density with 1-in-8 sampling to avoid GPU→CPU sync.

    Returns the cached density on non-sampled calls (no .item() sync).
    """
    counter = getattr(module, "_sparse_density_counter", -1) + 1
    object.__setattr__(module, "_sparse_density_counter", counter)
    if counter & 7 == 0:
        val = float(mask.mean().item())
        object.__setattr__(module, "_sparse_density_cached", val)
        return val
    return getattr(module, "_sparse_density_cached", 1.0)


def _telemetry_scalar(value, *, reduce: str = "sum") -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        data = value.detach()
        if reduce == "mean":
            return float(data.float().mean().item())
        return float(data.sum().item())
    return float(value)


def _record_routing_telemetry(
    module: nn.Module,
    n_experts: int,
    selected_experts: torch.Tensor,
    logits: Optional[torch.Tensor] = None,
    **extras,
) -> None:
    """Record MoE routing statistics with lightweight sampling."""
    telemetry = getattr(
        module,
        "routing_telemetry",
        {
            "tokens_total": 0,
            "tokens_processed": 0,
            "expert_counts": torch.zeros(n_experts, device=selected_experts.device),
            "entropy_sum": 0.0,
            "confidence_sum": 0.0,
            "confidence_sq_sum": 0.0,
            "confidence_count": 0,
            "count": 0,
            "heatmap": None,
            "_call_count": -1,
            "keep_count": 0,
            "drop_count": 0,
            "default_path_count": 0,
            "routed_token_count": 0,
            "sparse_span_count": 0,
            "sparse_span_width_sum": 0.0,
            "sparse_span_width_count": 0,
            "sparse_span_coverage_tokens": 0,
            "lane_histogram": torch.zeros(n_experts, device=selected_experts.device),
            "confidence_histogram": torch.zeros(10, device=selected_experts.device),
            "route_strength_sum": 0.0,
            "route_strength_count": 0,
            "branch_weight_sum": torch.zeros(n_experts, device=selected_experts.device),
            "branch_weight_count": 0,
            "branch_dominance_sum": 0.0,
            "routed_branch_share_sum": 0.0,
            "medium_branch_share_sum": 0.0,
            "hard_branch_share_sum": 0.0,
        },
    )

    def _resize_counter(name: str, size: int) -> None:
        current = telemetry.get(name)
        if not isinstance(current, torch.Tensor) or current.numel() != size:
            telemetry[name] = torch.zeros(
                size,
                device=selected_experts.device,
                dtype=torch.float32,
            )

    _resize_counter("expert_counts", n_experts)
    _resize_counter("lane_histogram", n_experts)
    _resize_counter("branch_weight_sum", n_experts)

    telemetry["_call_count"] += 1
    B, S = selected_experts.shape[:2]
    total_tokens = B * S
    telemetry["tokens_total"] += total_tokens
    telemetry["tokens_processed"] += total_tokens

    if telemetry["_call_count"] & 7 != 0:
        telemetry["count"] += 1
        setattr(module, "routing_telemetry", telemetry)
        return

    counts = torch.histc(
        selected_experts.float(), bins=n_experts, min=0, max=n_experts - 1
    )
    telemetry["expert_counts"] += counts
    telemetry["lane_histogram"] += counts

    if logits is not None:
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
        telemetry["entropy_sum"] += entropy
        # Confidence = max routing probability per token (1.0 = fully confident)
        conf = probs.max(dim=-1).values.mean().item()
        telemetry["confidence_sum"] += conf
        telemetry["confidence_sq_sum"] += conf * conf
        telemetry["confidence_count"] += 1
        telemetry["count"] += 1
        conf_hist = torch.histc(
            probs.max(dim=-1).values.float(), bins=10, min=0.0, max=1.0
        )
        telemetry["confidence_histogram"] += conf_hist

    if getattr(module, "_capture_heatmap", False) and telemetry["heatmap"] is None:
        telemetry["heatmap"] = selected_experts[0].detach().cpu().numpy().tolist()

    keep_mask = extras.get("keep_mask")
    if isinstance(keep_mask, torch.Tensor):
        keep_count = int(keep_mask.sum().item())
        telemetry["keep_count"] += keep_count
        telemetry["drop_count"] += int(keep_mask.numel()) - keep_count

    lane_histogram = extras.get("lane_histogram")
    if isinstance(lane_histogram, torch.Tensor):
        flat = lane_histogram.to(
            device=telemetry["lane_histogram"].device,
            dtype=telemetry["lane_histogram"].dtype,
        )
        if flat.numel() == telemetry["lane_histogram"].numel():
            telemetry["lane_histogram"] += flat.reshape_as(telemetry["lane_histogram"])

    sparse_span_count = extras.get("sparse_span_count")
    if sparse_span_count is not None:
        telemetry["sparse_span_count"] += int(_telemetry_scalar(sparse_span_count))
    sparse_span_width = extras.get("sparse_span_width")
    if sparse_span_width is not None:
        telemetry["sparse_span_width_sum"] += float(
            _telemetry_scalar(sparse_span_width)
        )
        telemetry["sparse_span_width_count"] += 1
    sparse_span_coverage_tokens = extras.get("sparse_span_coverage_tokens")
    if sparse_span_coverage_tokens is not None:
        telemetry["sparse_span_coverage_tokens"] += int(
            _telemetry_scalar(sparse_span_coverage_tokens)
        )
    default_path_count = extras.get("default_path_count")
    if default_path_count is not None:
        telemetry["default_path_count"] += int(_telemetry_scalar(default_path_count))
    routed_token_count = extras.get("routed_token_count")
    if routed_token_count is not None:
        telemetry["routed_token_count"] += int(_telemetry_scalar(routed_token_count))
    route_strength = extras.get("route_strength")
    if route_strength is not None:
        telemetry["route_strength_sum"] += float(
            _telemetry_scalar(route_strength, reduce="mean")
        )
        telemetry["route_strength_count"] += 1
    branch_weights = extras.get("branch_weights")
    if isinstance(branch_weights, torch.Tensor):
        flat = branch_weights.to(
            device=telemetry["branch_weight_sum"].device,
            dtype=telemetry["branch_weight_sum"].dtype,
        )
        if flat.numel() == telemetry["branch_weight_sum"].numel():
            telemetry["branch_weight_sum"] += flat.reshape_as(
                telemetry["branch_weight_sum"]
            )
            telemetry["branch_weight_count"] += 1
    dominance = extras.get("branch_dominance")
    if dominance is not None:
        telemetry["branch_dominance_sum"] += float(
            _telemetry_scalar(dominance, reduce="mean")
        )
    routed_share = extras.get("routed_branch_share")
    if routed_share is not None:
        telemetry["routed_branch_share_sum"] += float(
            _telemetry_scalar(routed_share, reduce="mean")
        )
    medium_share = extras.get("medium_branch_share")
    if medium_share is not None:
        telemetry["medium_branch_share_sum"] += float(
            _telemetry_scalar(medium_share, reduce="mean")
        )
    hard_share = extras.get("hard_branch_share")
    if hard_share is not None:
        telemetry["hard_branch_share_sum"] += float(
            _telemetry_scalar(hard_share, reduce="mean")
        )
    for key in (
        "routing_mode",
        "gate_type",
        "span_type",
        "fallback_mode",
        "lane_count",
    ):
        if extras.get(key) is not None:
            telemetry[key] = extras[key]
    trace_payload = extras.get("trace_payload")
    if trace_payload is not None:
        telemetry["trace_payload"] = trace_payload

    setattr(module, "routing_telemetry", telemetry)


def _build_nm_mask(weight: torch.Tensor, n: int, m: int) -> torch.Tensor:
    if n <= 0 or m <= 0 or n > m:
        return torch.ones_like(weight)

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
        raise RuntimeError(
            f"_flatten_for_kernel expected Tensor, got {type(x).__name__}"
        )
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


def _build_block_sparse_mask(
    weight: torch.Tensor, block_size: int, block_density: float
) -> torch.Tensor:
    block_size = max(1, int(block_size))
    # Floor at 0.25: below 25% density, too many gradient paths are dead
    # and convergence fails 41% of the time. Old floor was 0.05.
    block_density = float(max(0.25, min(1.0, block_density)))

    rows, cols = weight.shape
    row_blocks = rows // block_size
    col_blocks = cols // block_size
    if row_blocks <= 0 or col_blocks <= 0:
        return torch.ones_like(weight)

    usable_rows = row_blocks * block_size
    usable_cols = col_blocks * block_size
    core = weight[:usable_rows, :usable_cols]
    blocks = core.view(row_blocks, block_size, col_blocks, block_size).permute(
        0, 2, 1, 3
    )
    scores = blocks.abs().mean(dim=(2, 3))

    keep_per_row = max(1, int(round(col_blocks * block_density)))
    keep_idx = scores.topk(k=keep_per_row, dim=1).indices

    block_mask = torch.zeros_like(scores)
    block_mask.scatter_(1, keep_idx, 1.0)
    block_mask = (
        block_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, block_size, block_size)
    )
    block_mask = block_mask.permute(0, 2, 1, 3).reshape(usable_rows, usable_cols)

    mask = torch.ones_like(weight)
    mask[:usable_rows, :usable_cols] = block_mask
    return mask


# ── Op Implementations ──────────────────────────────────────────────


def _c(x):
    """Check if tensor is eligible for aria_core C kernels (fp32).

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


def _c16(x):
    """Check if tensor is eligible for aria_core fp16 C kernels."""
    return (
        HAS_ARIA_CORE
        and x.device.type == "cpu"
        and x.dtype == torch.float16
        and not x.requires_grad
        and x.dim() >= 1
        and x.numel() > 0
    )


def _t(x):
    """Check if tensor is eligible for Triton CUDA kernels.

    Triton kernels write into freshly-allocated output tensors and bypass
    autograd's tape, so skip them whenever the input requires gradients
    (sensitivity probes, second-order metrics) — the pure-PyTorch fallback
    in each op preserves the autograd graph.
    """
    return HAS_KERNELS and x.is_cuda and not x.requires_grad


def _is_inference_tensor(value: object) -> bool:
    """Return True when *value* is a tensor created inside inference_mode."""
    return isinstance(value, torch.Tensor) and bool(
        getattr(value, "is_inference", lambda: False)()
    )


def _get_stacked_params(
    module: nn.Module, attr_name: str, n: int, dtype: torch.dtype
) -> torch.Tensor:
    """Stack n ParameterList entries into a single tensor, cached by dtype.

    Avoids re-creating the stacked tensor on every forward pass. Cache is
    invalidated when dtype changes (e.g. during autocast transitions).
    """
    cache_key = f"_stacked_{attr_name}_{dtype}"
    cached = getattr(module, cache_key, None)
    if cached is not None and not _is_inference_tensor(cached):
        return cached
    params = getattr(module, attr_name)
    stacked = torch.stack([params[i].to(dtype) for i in range(n)])
    if not torch.is_inference_mode_enabled():
        object.__setattr__(module, cache_key, stacked)
    return stacked
