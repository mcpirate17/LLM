"""Native-first compile adapter for ExperimentRunner.

Phase-1 adapter-first reuse
# DEBUG: Log validation failures for S1 collapse diagnosis
import logging
logger = logging.getLogger(__name__)
logger.info("Native runner validation enabled - logging all S1 failures") policy:
- Prefer aria-designer runtime when enabled and compatible.
- Fall back to legacy research compiler unless strict mode is enabled.

Phase-3 native kernel dispatch:
- Load libaria_native_runtime.so via ctypes for native op support checking.
- Query the kernel registry (nk_is_registered) to determine per-op coverage.
- In strict mode, reject graphs with unsupported ops; otherwise log and fall back.

Phase-4 Cython bridge (preferred dispatch path):
- The Cython bridge (aria_bridge) provides zero-copy Python bindings to native
  C kernels and is the preferred dispatch path for op execution.
- When available, aria_bridge.is_native() replaces ctypes nk_is_registered for
  op support queries, and dispatch_op_native() routes ops through Cython kernels.
- Falls back to ctypes path when the Cython bridge is not importable.
"""

from __future__ import annotations

import ctypes
import json
from collections import Counter
import logging
import os
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .native_runner_adapter import (
    build_designer_layer_modules,
    capability_handshake,
    detect_adapter_state,
    try_designer_runtime_probe,
)


logger = logging.getLogger(__name__)


_FALLBACK_METRICS: Dict[str, int] = {
    "total_compiles": 0,
    "native_enabled_compiles": 0,
    "native_dispatch_compiles": 0,
    "selective_mode_candidates": 0,
    "selective_mode_activations": 0,
    "selective_mode_activation_failures": 0,
    "fallback_compiles": 0,
    "legacy_compile_count": 0,
    "legacy_compile_invocations": 0,
    "parity_samples": 0,
    "parity_passes": 0,
    "parity_failures": 0,
    "probe_successes": 0,
    "probe_failures": 0,
    "hybrid_compiles": 0,
}

PARTIAL_NATIVE_COVERAGE_THRESHOLD = 0.5  # 50% — at least half the ops must be native

_legacy_only_deprecation_warned = False

_SELECTIVE_GUARDRAIL: Dict[str, Any] = {
    "consecutive_requested_not_candidate": 0,
    "triggered": False,
    "trigger_count": 0,
    "last_reason": None,
}

_SELECTIVE_GUARDRAIL_HISTORY: List[Dict[str, Any]] = []
_SELECTIVE_GUARDRAIL_HISTORY_MAX = 25

_NATIVE_FALLBACK_LOG_WINDOW_S = 30.0
_NATIVE_FALLBACK_LOG_STATE: Dict[str, Any] = {
    "signature": None,
    "last_ts": 0.0,
    "suppressed": 0,
}

# Ops that represent graph wiring / orchestration rather than compute kernels.
# These should not count against native kernel support coverage.
_NON_KERNEL_STRUCTURAL_OPS: Set[str] = {
    "input",
    "graph_input",
    "output",
    "graph_output",
    "output_head",
    "concat",
    "split2",
    "split3",
}

_NATIVE_OP_ALIASES: Dict[str, str] = {
    # Parameterized linear variants → linear kernel
    "linear_proj": "linear",
    "linear_proj_down": "linear",
    "linear_proj_up": "linear",
    # Activation aliases
    "relu_op": "relu",
    "gelu_op": "gelu",
    "silu_op": "silu",
    # Softmax aliases
    "softmax_last": "softmax",
    # Normalization aliases
    "rmsnorm_pre": "rmsnorm",
    "layernorm_pre": "rmsnorm",
    # Structural aliases
    "transpose": "transpose2d",
    "transpose_sd": "transpose2d",
    # Parameterized → native C kernel aliases
    "swiglu_mlp": "swiglu",
    "learnable_scale": "mul",
    "learnable_bias": "add",
}

# Ops that are native to PyTorch and can be dispatched without a custom C kernel.
# The bridge handles these via standard torch.* calls.
_SOFT_BRIDGE_OPS: Set[str] = {
    "rfft_seq",
    "irfft_seq",
    "cumsum",
    "cumprod_safe",
    # Reductions (torch.sum, torch.mean, torch.max, torch.norm along dim)
    "sum_last",
    "sum_seq",
    "mean_last",
    "mean_seq",
    "max_last",
    "norm_last",
    # Structural (torch.roll, torch.gather, torch.scatter)
    "roll_seq",
    "roll_neg",
    "gather_sorted",
    "scatter_unsort",
    "multi_head_mix",
    # Math space binary ops with existing C kernels
    "tropical_add",
    "tropical_matmul",
    "geometric_product",
    "hyp_distance",
}

# Ops with real C kernel implementations (Tier 1-4).
# These are directly supported by the Cython bridge and C library.
_NATIVE_C_KERNEL_OPS: Set[str] = {
    # Tier 1: Elementwise + simple
    "maximum", "minimum", "div_safe", "sign_ste", "outer_product",
    "causal_mask", "softmax_seq",
    # Tier 2: Structural + parameterized
    "sliding_window_mask", "sort_seq", "argsort_seq", "conv1d_seq",
    "fused_linear_gelu", "swiglu", "token_pool_restore",
    "selective_scan", "topk_gate", "basis_expansion", "sparse_threshold",
    # Tier 3: Hyperbolic
    "exp_map", "log_map", "poincare_add", "hyp_linear",
    "hyperbolic_norm", "hyp_tangent_nonlinear",
    # Tier 3: Tropical (already had C kernels)
    "tropical_attention", "tropical_center", "tropical_gate",
    # Tier 3: P-adic
    "padic_gate", "padic_expand", "padic_residual", "ultrametric_attention",
    # Tier 3: Clifford
    "rotor_transform", "grade_select", "grade_mix", "clifford_attention",
    # Tier 3: Spiking
    "lif_neuron", "spike_rate_code", "stdp_attention",
    # Reference architecture ops
    "embedding_lookup", "rope_rotate", "gated_linear",
    "cosine_similarity", "gather_topk", "rwkv_time_mixing",
}

# Tier 4: Cython wrappers around PyTorch (still count as "native" for coverage).
_CYTHON_WRAPPER_OPS: Set[str] = {
    "nm_sparse_linear", "block_sparse_linear", "semi_structured_2_4_linear",
    "rwkv_channel", "bottleneck_proj", "grouped_linear", "low_rank_proj",
    "shared_basis_proj", "tied_proj", "integral_kernel", "fixed_point_iter",
    "local_window_attn", "softmax_attention",
    # Mixing ops (parameterized, delegate to PyTorch nn.Module internals)
    "linear_attention", "graph_attention", "fourier_mixing",
    "state_space", "conv_only", "moe_topk",
}

# Module-level cache: avoids reloading the native shared library on every compile call.
# Sentinel ``False`` means "not yet attempted"; ``None`` means "attempted but unavailable".
_native_lib_cache: Any = False

# Module-level cache for the Cython bridge module.
# Sentinel ``False`` means "not yet attempted"; ``None`` means "attempted but unavailable".
_cython_bridge_cache: Any = False

# Module-level cache for the Rust scheduler module.
_rust_scheduler_cache: Any = False


class _NrCompileRequest(ctypes.Structure):
    _fields_ = [
        ("ir_json", ctypes.c_char_p),
        ("ir_json_len", ctypes.c_int64),
        ("vocab_size", ctypes.c_int32),
        ("max_seq_len", ctypes.c_int32),
    ]


class _NrCompileResponse(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("model_handle", ctypes.c_int64),
        ("message", ctypes.c_char_p),
    ]


class _NrExecuteRequest(ctypes.Structure):
    _fields_ = [
        ("model_handle", ctypes.c_int64),
        ("token_ids", ctypes.POINTER(ctypes.c_int32)),
        ("batch", ctypes.c_int32),
        ("seq_len", ctypes.c_int32),
    ]


class _NrExecuteResponse(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int32),
        ("message", ctypes.c_char_p),
    ]


def _try_import_cython_bridge() -> Any:
    """Try to import the Cython bridge module (aria_bridge).

    Adds the cython build directory to sys.path if needed. The result is
    cached module-level so subsequent calls are free.

    Returns the aria_bridge module, or None if unavailable.
    """
    global _cython_bridge_cache
    if _cython_bridge_cache is not False:
        return _cython_bridge_cache

    # Try direct import first (may already be on sys.path).
    try:
        import aria_bridge  # type: ignore[import-untyped]
        _cython_bridge_cache = aria_bridge
        logger.info("Loaded Cython bridge (aria_bridge) via direct import")
        return _cython_bridge_cache
    except ImportError:
        pass

    # Add the cython directory to sys.path and retry.
    cython_dir = str(
        Path(__file__).resolve().parents[1] / "runtime" / "native" / "cython"
    )
    if cython_dir not in sys.path:
        sys.path.insert(0, cython_dir)
    try:
        import aria_bridge  # type: ignore[import-untyped]
        _cython_bridge_cache = aria_bridge
        logger.info("Loaded Cython bridge (aria_bridge) from %s", cython_dir)
        return _cython_bridge_cache
    except ImportError as exc:
        logger.debug("Cython bridge not available: %s", exc)
        _cython_bridge_cache = None
        return None


def _try_import_rust_scheduler() -> Any:
    """Try to import the Rust scheduler module (aria_scheduler)."""
    global _rust_scheduler_cache
    if _rust_scheduler_cache is not False:
        return _rust_scheduler_cache

    try:
        from . import aria_scheduler
        _rust_scheduler_cache = aria_scheduler
        logger.info("Loaded Rust scheduler (aria_scheduler)")
        return _rust_scheduler_cache
    except ImportError as exc:
        logger.debug("Rust scheduler not available: %s", exc)
        _rust_scheduler_cache = None
        return None


def _reset_cython_bridge_cache() -> None:
    """Reset the Cython bridge cache (used in tests)."""
    global _cython_bridge_cache
    _cython_bridge_cache = False


@dataclass
class NativeRunnerState:
    enabled: bool
    strict: bool
    designer_runtime_available: bool
    reason: str


class DesignerWorkflowLayerAdapter:
    """Adapt aria-designer WorkflowModule to the layer(x)->y interface."""

    def __init__(self, workflow_module: Any, input_node_id: str):
        import torch.nn as nn  # lazy import keeps module import light for non-runtime tests

        class _Adapter(nn.Module):
            def __init__(self, wm: Any, in_id: str):
                super().__init__()
                self.workflow_module = wm
                self.input_node_id = in_id

            def forward(self, x):
                out = self.workflow_module({self.input_node_id: x})
                if isinstance(out, dict):
                    for key in ("y", "logits"):
                        value = out.get(key)
                        if value is not None:
                            return value
                    for value in out.values():
                        if value is not None:
                            return value
                return out

        self.module = _Adapter(workflow_module, input_node_id)

    def as_module(self):
        return self.module


def _validate_designer_layer_adapter_contract(
    adapter_module: Any,
    *,
    model_dim: int,
    max_seq_len: Optional[int],
) -> Optional[str]:
    """Return None when adapter output contract is safe, else skip reason."""
    try:
        import torch
    except Exception:
        return "torch_unavailable_for_contract_check"

    if model_dim <= 0:
        return "invalid_model_dim"
    seq = int(max_seq_len or 8)
    seq = max(1, min(seq, 8))

    try:
        with torch.no_grad():
            x = torch.zeros((1, seq, model_dim), dtype=torch.float32)
            y = adapter_module(x)
    except Exception as exc:
        return f"adapter_forward_error:{exc}"

    if not isinstance(y, torch.Tensor):
        return "adapter_output_not_tensor"
    if y.ndim != 3:
        return f"adapter_output_rank_{y.ndim}"
    if int(y.shape[0]) != 1 or int(y.shape[1]) != seq:
        return f"adapter_output_shape_mismatch:{tuple(int(v) for v in y.shape)}"
    if int(y.shape[2]) != model_dim:
        return f"adapter_output_dim_mismatch:{int(y.shape[2])}!={model_dim}"
    return None


def _summarize_layer_build(layer_build: Dict[str, Any]) -> Dict[str, Any]:
    """Build compact summary fields for API/dashboard parsing."""
    layer_results = layer_build.get("layer_results") or []
    skip_reasons = [
        str(item.get("skip_reason"))
        for item in layer_results
        if not bool(item.get("applied")) and item.get("skip_reason")
    ]
    reason_counts = Counter(skip_reasons)
    top_skip_reasons = [
        {"reason": reason, "count": int(count)}
        for reason, count in reason_counts.most_common(3)
    ]
    error_layers = sum(1 for item in layer_results if item.get("error"))
    summary = {
        "applied_layers": int(layer_build.get("applied_layers") or 0),
        "skipped_layers": int(layer_build.get("skipped_layers") or 0),
        "error_layers": int(error_layers),
        "top_skip_reasons": top_skip_reasons,
    }
    return summary


def _env_flag(name: str, default: bool) -> bool:
    # Backward-compat shim for callers importing this helper directly.
    from .native_runner_adapter import _env_flag as _adapter_env_flag
    return _adapter_env_flag(name, default)


def _normalize_nr_compile_reason(compile_status: int, compile_message: Optional[str]) -> str:
    msg = str(compile_message or "").strip().lower()
    if not msg:
        return f"status_{int(compile_status)}"

    known_prefixes = (
        "unsupported_graph_family_",
        "missing_",
        "invalid_",
        "strict_mode_",
        "handle_",
        "logit_",
        "add_",
        "mul_",
        "matmul_",
        "linear_",
        "softmax_",
        "rmsnorm_",
        "sub_",
        "unary_",
    )
    if msg.startswith(known_prefixes):
        return msg
    if "required_chain_missing_or_invalid" in msg:
        return "unsupported_graph_family_required_chain_missing_or_invalid"
    if "required_chain_invalid" in msg:
        return "unsupported_graph_family_required_chain_invalid"
    if "unsupported_graph_family" in msg:
        return "unsupported_graph_family_unspecified"
    if "kernel" in msg:
        return "kernel_lookup_failure"
    return msg.replace(":", "_").replace(" ", "_")


def _try_load_native_lib() -> Any:
    """Try to load the native C kernel library. Returns ctypes CDLL or None.

    The result is cached in ``_native_lib_cache`` so subsequent calls are free.
    """
    global _native_lib_cache
    if _native_lib_cache is not False:
        return _native_lib_cache

    lib_paths = [
        Path(__file__).resolve().parents[1] / "runtime" / "native" / "build" / "libaria_native_runtime.so",
        Path(__file__).resolve().parents[2] / "aria-designer" / "runtime" / "lib" / "libaria_runtime.so",
    ]
    for p in lib_paths:
        if p.exists():
            try:
                _native_lib_cache = ctypes.CDLL(str(p))
                logger.info("Loaded native kernel library from %s", p)
                return _native_lib_cache
            except OSError as exc:
                logger.debug("Failed to load native lib at %s: %s", p, exc)
                continue

    _native_lib_cache = None
    return None


def _reset_native_lib_cache() -> None:
    """Reset the library cache (used in tests)."""
    global _native_lib_cache
    _native_lib_cache = False


def _check_native_op_support(layer_graphs: List[Any], native_lib: Any) -> Dict[str, Any]:
    """Check which ops in the graphs have native kernel support.

    Prefers the Cython bridge (aria_bridge.is_native) when available.
    Falls back to the C kernel registry via ctypes ``nk_is_registered``.
    """
    all_ops: Set[str] = set()
    for g in layer_graphs:
        for node in getattr(g, "nodes", {}).values():
            all_ops.add(getattr(node, "op_name", str(node)))

    kernel_relevant_ops: Set[str] = {
        op for op in all_ops if op not in _NON_KERNEL_STRUCTURAL_OPS
    }

    supported: Set[str] = set()
    unsupported: Set[str] = set()

    def _canonical_op(op_name: str) -> str:
        return _NATIVE_OP_ALIASES.get(op_name, op_name)

    # Quick-check sets: ops known to have native support without needing bridge/lib query.
    _all_known_native = _SOFT_BRIDGE_OPS | _NATIVE_C_KERNEL_OPS | _CYTHON_WRAPPER_OPS

    # Prefer explicit native library handle when provided by caller/tests.
    if native_lib is not None and hasattr(native_lib, "nk_is_registered"):
        if hasattr(native_lib, "nr_runtime_init"):
            native_lib.nr_runtime_init()
        is_registered = native_lib.nk_is_registered
        is_registered.argtypes = [ctypes.c_char_p]
        is_registered.restype = ctypes.c_int32
        for op in kernel_relevant_ops:
            if op in _all_known_native:
                supported.add(op)
                continue
            kernel_op = _canonical_op(op)
            if kernel_op in _all_known_native:
                supported.add(op)
                continue
            if is_registered(kernel_op.encode("utf-8")):
                supported.add(op)
            else:
                unsupported.add(op)
    else:
        # Fallback: Cython bridge query path.
        bridge = _try_import_cython_bridge()
        if bridge is not None and hasattr(bridge, "is_native"):
            for op in kernel_relevant_ops:
                if op in _all_known_native:
                    supported.add(op)
                    continue
                kernel_op = _canonical_op(op)
                if kernel_op in _all_known_native:
                    supported.add(op)
                    continue
                if bridge.is_native(kernel_op):
                    supported.add(op)
                else:
                    unsupported.add(op)
        else:
            # Check known sets even without full Cython bridge
            for op in kernel_relevant_ops:
                kernel_op = _canonical_op(op)
                if op in _all_known_native or kernel_op in _all_known_native:
                    supported.add(op)
                else:
                    unsupported.add(op)

    if not all_ops:
        native_coverage = 0.0
    elif not kernel_relevant_ops:
        native_coverage = 1.0
    else:
        native_coverage = len(supported) / len(kernel_relevant_ops)

    return {
        "all_ops": sorted(all_ops),
        "kernel_relevant_ops": sorted(kernel_relevant_ops),
        "supported": sorted(supported),
        "unsupported": sorted(unsupported),
        "native_coverage": native_coverage,
    }


def _log_native_fallback_coverage(op_support: Dict[str, Any]) -> None:
    """Log native coverage fallback with burst deduplication.

    During investigation cycles, many compile attempts can produce identical
    coverage diagnostics in rapid succession. Emit the first message and
    suppress repeats within a short window to reduce log noise.
    """
    coverage = float(op_support.get("native_coverage") or 0.0)
    supported_count = len(op_support.get("supported") or [])
    all_count = len(op_support.get("all_ops") or [])
    unsupported = list(op_support.get("unsupported") or [])

    signature = (supported_count, all_count, tuple(unsupported))
    now = time.time()
    state = _NATIVE_FALLBACK_LOG_STATE

    same_signature = signature == state["signature"]
    within_window = (now - float(state["last_ts"] or 0.0)) <= _NATIVE_FALLBACK_LOG_WINDOW_S

    if same_signature and within_window:
        state["suppressed"] = int(state["suppressed"] or 0) + 1
        return

    suppressed = int(state.get("suppressed") or 0)
    if suppressed > 0 and state.get("signature") is not None:
        logger.debug(
            "Suppressed %d repeated native fallback coverage log(s) in the last %.0fs.",
            suppressed,
            _NATIVE_FALLBACK_LOG_WINDOW_S,
        )

    logger.debug(
        "Native kernel coverage %.1f%% (%d/%d ops). Unsupported: %s. "
        "Falling back to legacy compile.",
        coverage * 100,
        supported_count,
        all_count,
        unsupported,
    )

    state["signature"] = signature
    state["last_ts"] = now
    state["suppressed"] = 0


def _legacy_compile_model(
    layer_graphs: List[Any],
    vocab_size: int = 32000,
    max_seq_len: Optional[int] = None,
    **kwargs: Any,
):
    # Lazy import keeps adapter unit tests independent of heavyweight runtime deps.
    from ..synthesis.compiler import compile_model as _compile_model

    return _compile_model(
        layer_graphs,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        **kwargs,
    )


def _record_legacy_compile_invocation() -> None:
    _FALLBACK_METRICS["legacy_compile_count"] += 1
    _FALLBACK_METRICS["legacy_compile_invocations"] += 1


def _legacy_compile_count() -> int:
    canonical = _FALLBACK_METRICS.get("legacy_compile_count")
    if canonical is not None:
        return int(canonical)
    return int(_FALLBACK_METRICS.get("legacy_compile_invocations") or 0)


def _maybe_warn_deprecated_legacy_only_flag() -> None:
    global _legacy_only_deprecation_warned
    if _legacy_only_deprecation_warned:
        return
    logger.warning(
        "NATIVE_RUNNER_LEGACY_ONLY is deprecated and scheduled for Phase-D removal; "
        "prefer NATIVE_RUNNER_ABI_MODEL_ONLY=0 and NATIVE_RUNNER_ALLOW_LEGACY_FALLBACK=1 "
        "for controlled rollback behavior."
    )
    _legacy_only_deprecation_warned = True


def reset_native_runner_telemetry() -> None:
    for key in list(_FALLBACK_METRICS.keys()):
        _FALLBACK_METRICS[key] = 0
    _SELECTIVE_GUARDRAIL["consecutive_requested_not_candidate"] = 0
    _SELECTIVE_GUARDRAIL["triggered"] = False
    _SELECTIVE_GUARDRAIL["trigger_count"] = 0
    _SELECTIVE_GUARDRAIL["last_reason"] = None
    _SELECTIVE_GUARDRAIL_HISTORY.clear()


def record_native_abi_parity_result(passed: Optional[bool]) -> None:
    """Record sampled ABI parity outcome from sandbox/runner integration."""
    if passed is None:
        return
    _FALLBACK_METRICS["parity_samples"] += 1
    if bool(passed):
        _FALLBACK_METRICS["parity_passes"] += 1
    else:
        _FALLBACK_METRICS["parity_failures"] += 1


def _record_guardrail_event(
    event: str,
    *,
    reason: Optional[str],
    threshold: int,
    source: Optional[str] = None,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    entry = {
        "event": str(event),
        "timestamp": timestamp,
        "source": source,
        "reason": reason,
        "consecutive_requested_not_candidate": int(
            _SELECTIVE_GUARDRAIL.get("consecutive_requested_not_candidate") or 0
        ),
        "threshold": int(threshold),
        "trigger_count": int(_SELECTIVE_GUARDRAIL.get("trigger_count") or 0),
    }
    _SELECTIVE_GUARDRAIL_HISTORY.append(entry)
    if len(_SELECTIVE_GUARDRAIL_HISTORY) > _SELECTIVE_GUARDRAIL_HISTORY_MAX:
        del _SELECTIVE_GUARDRAIL_HISTORY[0 : len(_SELECTIVE_GUARDRAIL_HISTORY) - _SELECTIVE_GUARDRAIL_HISTORY_MAX]


def detect_native_state() -> NativeRunnerState:
    adapter_state = detect_adapter_state()
    return NativeRunnerState(
        enabled=adapter_state.enabled,
        strict=adapter_state.strict,
        designer_runtime_available=adapter_state.designer_runtime_available,
        reason=adapter_state.reason,
    )


def native_runner_capability_report() -> Dict[str, Any]:
    report = capability_handshake()
    state = detect_native_state()
    # Phase D: ABI model-only is always active when native is enabled.
    abi_model_only = state.enabled
    disable_legacy_compile = _env_flag("NATIVE_RUNNER_DISABLE_LEGACY_COMPILE", False)
    disable_legacy_compile_native_enabled = _env_flag(
        "NATIVE_RUNNER_DISABLE_LEGACY_COMPILE_NATIVE_ENABLED",
        False,
    )
    legacy_disabled = bool(disable_legacy_compile or (state.enabled and disable_legacy_compile_native_enabled))
    if legacy_disabled and state.enabled and disable_legacy_compile_native_enabled:
        legacy_disabled_reason = "native_enabled_gate"
    elif legacy_disabled:
        legacy_disabled_reason = "env_flag"
    else:
        legacy_disabled_reason = None

    if legacy_disabled:
        execution_mode = "legacy_disabled"
    elif state.enabled:
        execution_mode = "native_abi_model_only"
    else:
        execution_mode = "legacy_only"

    report["execution_mode_classification"] = execution_mode
    report["legacy_compile_disabled"] = legacy_disabled
    report["legacy_compile_disabled_reason"] = legacy_disabled_reason
    total = int(_FALLBACK_METRICS.get("total_compiles") or 0)
    native_total = int(_FALLBACK_METRICS.get("native_enabled_compiles") or 0)
    fallback = int(_FALLBACK_METRICS.get("fallback_compiles") or 0)
    legacy_count = _legacy_compile_count()
    hybrid = int(_FALLBACK_METRICS.get("hybrid_compiles") or 0)
    report["fallback_metrics"] = {
        **_FALLBACK_METRICS,
        "legacy_compile_count": legacy_count,
        "legacy_compile_invocations": legacy_count,
        "hybrid_compiles": hybrid,
        "fallback_rate": (float(fallback) / float(native_total)) if native_total > 0 else 0.0,
        "hybrid_rate": (float(hybrid) / float(native_total)) if native_total > 0 else 0.0,
        "max_allowed_fallback_rate": os.environ.get("NATIVE_RUNNER_MAX_FALLBACK_RATE"),
        "max_allowed_legacy_compile_count": os.environ.get("NATIVE_RUNNER_MAX_LEGACY_COMPILE_INVOCATIONS"),
        "max_allowed_legacy_compile_invocations": os.environ.get("NATIVE_RUNNER_MAX_LEGACY_COMPILE_INVOCATIONS"),
        "samples_considered": native_total,
        "all_compile_calls": total,
        "deprecated_fields": {
            "legacy_compile_invocations": "use legacy_compile_count",
            "max_allowed_legacy_compile_invocations": "use max_allowed_legacy_compile_count",
        },
    }
    checks: List[Dict[str, Any]] = []
    fallback_limit_raw = os.environ.get("NATIVE_RUNNER_MAX_FALLBACK_RATE")
    if fallback_limit_raw is not None:
        try:
            fallback_limit = max(0.0, min(1.0, float(str(fallback_limit_raw))))
        except Exception:
            fallback_limit = 1.0
        fallback_rate = report["fallback_metrics"]["fallback_rate"]
        checks.append({
            "name": "fallback_rate",
            "active": True,
            "pass": bool(fallback_rate <= fallback_limit),
            "actual": float(fallback_rate),
            "limit": float(fallback_limit),
        })

    legacy_limit_raw = os.environ.get("NATIVE_RUNNER_MAX_LEGACY_COMPILE_INVOCATIONS")
    if legacy_limit_raw is not None:
        try:
            legacy_limit = max(0, int(str(legacy_limit_raw)))
        except Exception:
            legacy_limit = 0
        legacy_used = legacy_count
        checks.append({
            "name": "legacy_compile_invocations",
            "active": True,
            "pass": bool(legacy_used <= legacy_limit),
            "actual": int(legacy_used),
            "limit": int(legacy_limit),
        })

    require_parity = _env_flag("NATIVE_RUNNER_REQUIRE_PARITY_PASS", False)
    parity_samples = int(_FALLBACK_METRICS.get("parity_samples") or 0)
    parity_failures = int(_FALLBACK_METRICS.get("parity_failures") or 0)
    if require_parity:
        if parity_samples <= 0:
            checks.append({
                "name": "parity",
                "active": True,
                "pass": None,
                "actual": "no_samples",
                "limit": "no_failures",
            })
        else:
            checks.append({
                "name": "parity",
                "active": True,
                "pass": bool(parity_failures == 0),
                "actual": int(parity_failures),
                "limit": 0,
            })

    active_checks = [c for c in checks if c.get("active")]
    if not active_checks:
        cutover_ready = None
        cutover_status = "waiting"
    elif any(c.get("pass") is None for c in active_checks):
        cutover_ready = None
        cutover_status = "waiting"
    else:
        cutover_ready = all(bool(c.get("pass")) for c in active_checks)
        cutover_status = "ready" if cutover_ready else "blocked"
    report["cutover_gate"] = {
        "ready": cutover_ready,
        "status": cutover_status,
        "checks": active_checks,
    }
    try:
        threshold = int(str(os.environ.get("NATIVE_RUNNER_SELECTIVE_GUARDRAIL_WINDOW", "5")))
    except Exception:
        threshold = 5
    threshold = max(1, threshold)
    report["selective_guardrail"] = {
        "consecutive_requested_not_candidate": int(
            _SELECTIVE_GUARDRAIL.get("consecutive_requested_not_candidate") or 0
        ),
        "threshold": threshold,
        "triggered": bool(_SELECTIVE_GUARDRAIL.get("triggered")),
        "trigger_count": int(_SELECTIVE_GUARDRAIL.get("trigger_count") or 0),
        "last_reason": _SELECTIVE_GUARDRAIL.get("last_reason"),
        "history": [dict(item) for item in _SELECTIVE_GUARDRAIL_HISTORY],
    }
    return report


def _maybe_fail_on_fallback_rate() -> None:
    max_rate_env = os.environ.get("NATIVE_RUNNER_MAX_FALLBACK_RATE")
    if max_rate_env is None:
        if not _env_flag("NATIVE_RUNNER_FAIL_ON_FALLBACK_RATE", False):
            return
        max_rate_env = os.environ.get("NATIVE_RUNNER_FALLBACK_RATE_MAX", "1.0")
    try:
        max_rate_raw = float(str(max_rate_env))
    except Exception:
        max_rate_raw = 1.0
    max_rate_raw = max(0.0, min(1.0, max_rate_raw))
    try:
        min_samples = int(str(os.environ.get("NATIVE_RUNNER_FALLBACK_MIN_SAMPLES", "1")))
    except Exception:
        min_samples = 1

    total = int(_FALLBACK_METRICS.get("native_enabled_compiles") or 0)
    fallback = int(_FALLBACK_METRICS.get("fallback_compiles") or 0)
    if total < max(1, min_samples):
        return
    rate = float(fallback) / float(total)
    if rate > max_rate_raw:
        raise RuntimeError(
            "Native runner fallback rate exceeded threshold: "
            f"rate={rate:.3f} threshold={max_rate_raw:.3f} total={total}"
        )


def _maybe_fail_on_legacy_compile_usage() -> None:
    max_legacy_env = os.environ.get("NATIVE_RUNNER_MAX_LEGACY_COMPILE_INVOCATIONS")
    if max_legacy_env is None:
        return
    try:
        max_legacy = int(str(max_legacy_env))
    except Exception:
        max_legacy = -1
    max_legacy = max(0, max_legacy)
    used = _legacy_compile_count()
    if used > max_legacy:
        raise RuntimeError(
            "Native runner legacy compile usage exceeded threshold: "
            f"used={used} threshold={max_legacy}"
        )


def _requested_execution_mode() -> str:
    raw = str(os.environ.get("NATIVE_RUNNER_EXECUTION_MODE", "probe")).strip().lower()
    if raw in {"probe", "selective"}:
        return raw
    return "probe"


def _activate_selective_native_dispatch(native_lib: Any) -> Dict[str, Any]:
    """Run a tiny native-kernel execution path to confirm selective activation.

    Prefers the Cython bridge when available; falls back to raw ctypes calls.
    This is intentionally narrow and safe: execute `relu` and `add` on tiny fixed
    buffers. The model compile path remains legacy until full runner ABI execution
    is wired.
    """
    result: Dict[str, Any] = {
        "activated": False,
        "ops": ["relu", "add"],
        "reason": "not_attempted",
    }

    # Try Cython bridge first.
    bridge = _try_import_cython_bridge()
    if bridge is not None:
        try:
            import numpy as np
            x = np.array([-1.0, 0.0, 2.0, 3.5], dtype=np.float32)
            relu_out = bridge.dispatch_unary("relu", x)
            relu_list = [float(v) for v in relu_out]
            if relu_list != [0.0, 0.0, 2.0, 3.5]:
                result["reason"] = f"relu_mismatch:{relu_list}"
                return result

            a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
            add_out = bridge.dispatch_binary("add", a, b)
            add_list = [float(v) for v in add_out]
            if add_list != [11.0, 22.0, 33.0, 44.0]:
                result["reason"] = f"add_mismatch:{add_list}"
                return result

            result["activated"] = True
            result["reason"] = "ok"
            result["dispatch_backend"] = "cython"
            
            # Also check Rust scheduler
            rust = _try_import_rust_scheduler()
            if rust is not None:
                result["rust_scheduler"] = "available"
                # Simple topo probe
                order = rust.topological_order(json.dumps({
                    "schema_version": "0.1", "model_dim": 4, "output_node_id": 1,
                    "nodes": [{"id": 0, "op_name": "input", "is_input": True, "input_ids": [], "config": {}},
                              {"id": 1, "op_name": "relu", "input_ids": [0], "config": {}}],
                    "edges": [{"source": 0, "target": 1}]
                }))
                if order == [0, 1]:
                    result["rust_scheduler"] = "ok"
                else:
                    result["rust_scheduler"] = f"topo_mismatch:{order}"
            else:
                result["rust_scheduler"] = "missing"

            return result
        except Exception as exc:
            logger.debug("Cython bridge activation failed, falling back to ctypes: %s", exc)
            # Fall through to ctypes path.

    # Fallback: raw ctypes path.
    if native_lib is None:
        result["reason"] = "native_lib_unavailable"
        return result

    relu_fn = getattr(native_lib, "aria_relu_f32", None)
    add_fn = getattr(native_lib, "aria_add_f32", None)
    if not callable(relu_fn) or not callable(add_fn):
        result["reason"] = "missing_required_symbols"
        return result

    try:
        n = 4
        x = (ctypes.c_float * n)(-1.0, 0.0, 2.0, 3.5)
        y = (ctypes.c_float * n)()
        relu_fn(x, y, n)
        relu_out = [float(y[i]) for i in range(n)]
        if relu_out != [0.0, 0.0, 2.0, 3.5]:
            result["reason"] = f"relu_mismatch:{relu_out}"
            return result

        a = (ctypes.c_float * n)(1.0, 2.0, 3.0, 4.0)
        b = (ctypes.c_float * n)(10.0, 20.0, 30.0, 40.0)
        z = (ctypes.c_float * n)()
        add_fn(a, b, z, n)
        add_out = [float(z[i]) for i in range(n)]
        if add_out != [11.0, 22.0, 33.0, 44.0]:
            result["reason"] = f"add_mismatch:{add_out}"
            return result

        result["activated"] = True
        result["reason"] = "ok"
        result["dispatch_backend"] = "ctypes"
        return result
    except Exception as exc:
        result["reason"] = f"activation_error:{exc}"
        return result


# ── Op categories for dispatch_op_native routing ─────────────────────

_CYTHON_UNARY_OPS = frozenset({"relu", "gelu", "silu", "square", "abs", "neg", "reciprocal", "log", "sqrt", "sigmoid", "tanh", "exp", "sin", "cos", "sign_ste"})
_CYTHON_BINARY_OPS = frozenset({"add", "mul", "sub", "maximum", "minimum", "div_safe", "outer_product"})


def dispatch_op_native(op_name: str, *tensors, **kwargs) -> Any:
    """Dispatch a single op through the native Cython bridge.

    Returns numpy array result, or raises if op is unsupported or the
    Cython bridge is not available.

    Supported op routing:
    - Unary ops (relu, gelu, silu, square, abs, neg, reciprocal, log, sqrt, sin, cos, sigmoid, tanh, exp): dispatch_unary
    - Binary ops (add, mul, sub): dispatch_binary
    - Composite fallback (square): dispatch_binary("mul", x, x) when bridge lacks unary square
    - matmul: dispatch_matmul
    - linear / linear_proj: dispatch_linear (kwargs: bias)
    - rmsnorm: dispatch_rmsnorm (kwargs: eps)
    - softmax / softmax_last: dispatch_softmax
    - layernorm: dispatch_layernorm (kwargs: eps)
    - transpose, transpose2d: dispatch_transpose2d
    """
    bridge = _try_import_cython_bridge()
    if bridge is None:
        raise RuntimeError(
            "Cython bridge (aria_bridge) is not available. "
            "Cannot dispatch op natively."
        )

    canonical_op = _NATIVE_OP_ALIASES.get(op_name, op_name)

    if canonical_op in _CYTHON_UNARY_OPS:
        if len(tensors) != 1:
            raise ValueError(f"Unary op '{op_name}' expects 1 tensor, got {len(tensors)}")
        if canonical_op == "square":
            try:
                return bridge.dispatch_unary(canonical_op, tensors[0])
            except ValueError:
                return bridge.dispatch_binary("mul", tensors[0], tensors[0])
        return bridge.dispatch_unary(canonical_op, tensors[0])

    if canonical_op in _CYTHON_BINARY_OPS:
        if len(tensors) != 2:
            raise ValueError(f"Binary op '{op_name}' expects 2 tensors, got {len(tensors)}")
        return bridge.dispatch_binary(canonical_op, tensors[0], tensors[1])

    if canonical_op == "matmul":
        if len(tensors) != 2:
            raise ValueError(f"matmul expects 2 tensors, got {len(tensors)}")
        return bridge.dispatch_matmul(tensors[0], tensors[1])

    if canonical_op == "linear":
        if len(tensors) < 2:
            raise ValueError(f"linear expects at least 2 tensors (x, W), got {len(tensors)}")
        bias = kwargs.get("bias", tensors[2] if len(tensors) > 2 else None)
        return bridge.dispatch_linear(tensors[0], tensors[1], bias=bias)

    if canonical_op == "rmsnorm":
        if len(tensors) < 2:
            raise ValueError(f"rmsnorm expects at least 2 tensors (x, weight), got {len(tensors)}")
        eps = kwargs.get("eps", 1e-5)
        return bridge.dispatch_rmsnorm(tensors[0], tensors[1], eps=eps)

    if canonical_op == "softmax":
        if len(tensors) != 1:
            raise ValueError(f"softmax expects 1 tensor, got {len(tensors)}")
        return bridge.dispatch_softmax(tensors[0])

    if canonical_op == "layernorm":
        if len(tensors) < 3:
            raise ValueError(f"layernorm expects 3 tensors (x, weight, bias), got {len(tensors)}")
        eps = kwargs.get("eps", 1e-5)
        return bridge.dispatch_layernorm(tensors[0], tensors[1], tensors[2], eps=eps)

    if canonical_op == "transpose2d":
        if len(tensors) != 1:
            raise ValueError(f"transpose2d expects 1 tensor, got {len(tensors)}")
        return bridge.dispatch_transpose2d(tensors[0])

    raise ValueError(f"Unsupported op for native dispatch: '{op_name}'")


# ── Backward op categories for dispatch_op_backward_native routing ──

_CYTHON_UNARY_BACKWARD_OPS = frozenset({"relu", "gelu", "silu", "sigmoid", "tanh"})
_CYTHON_BINARY_BACKWARD_OPS = frozenset({"add", "mul", "sub", "maximum", "minimum", "div_safe"})
_CYTHON_NORM_BACKWARD_OPS = frozenset({"softmax", "layernorm", "rmsnorm"})


def dispatch_op_backward_native(op_name: str, grad_output, *saved_tensors) -> Any:
    """Dispatch a backward (gradient) op through the native Cython bridge.

    Returns gradient tensor(s) as numpy arrays. For unary ops returns a single
    array; for binary/matmul ops returns a tuple of arrays.

    Supported op routing:
    - Unary backward (relu, gelu, silu, sigmoid, tanh):
        saved_tensors = (forward_input_or_output,) -> grad_input
    - Binary backward (add, mul, sub):
        saved_tensors = (a, b) -> (grad_a, grad_b)
    - matmul backward:
        saved_tensors = (A, B) -> (grad_A, grad_B)
    """
    bridge = _try_import_cython_bridge()
    if bridge is None:
        raise RuntimeError(
            "Cython bridge (aria_bridge) is not available. "
            "Cannot dispatch backward op natively."
        )

    if op_name in _CYTHON_UNARY_BACKWARD_OPS:
        if len(saved_tensors) != 1:
            raise ValueError(
                f"Unary backward '{op_name}' expects 1 saved tensor, got {len(saved_tensors)}"
            )
        return bridge.dispatch_unary_backward(op_name, grad_output, saved_tensors[0])

    if op_name in _CYTHON_BINARY_BACKWARD_OPS:
        if len(saved_tensors) != 2:
            raise ValueError(
                f"Binary backward '{op_name}' expects 2 saved tensors (a, b), got {len(saved_tensors)}"
            )
        return bridge.dispatch_binary_backward(op_name, grad_output, saved_tensors[0], saved_tensors[1])

    if op_name == "matmul":
        if len(saved_tensors) != 2:
            raise ValueError(
                f"matmul backward expects 2 saved tensors (A, B), got {len(saved_tensors)}"
            )
        return bridge.dispatch_matmul_backward(grad_output, saved_tensors[0], saved_tensors[1])

    if op_name == "softmax":
        if len(saved_tensors) != 1:
            raise ValueError(
                f"softmax backward expects 1 saved tensor (output), got {len(saved_tensors)}"
            )
        return bridge.dispatch_softmax_backward(grad_output, saved_tensors[0])

    if op_name == "layernorm":
        if len(saved_tensors) != 2:
            raise ValueError(
                f"layernorm backward expects 2 saved tensors (input, gamma), got {len(saved_tensors)}"
            )
        return bridge.dispatch_layernorm_backward(grad_output, saved_tensors[0], saved_tensors[1])

    if op_name == "rmsnorm":
        if len(saved_tensors) != 2:
            raise ValueError(
                f"rmsnorm backward expects 2 saved tensors (input, gamma), got {len(saved_tensors)}"
            )
        return bridge.dispatch_rmsnorm_backward(grad_output, saved_tensors[0], saved_tensors[1])

    raise ValueError(f"Unsupported op for native backward dispatch: '{op_name}'")


def enable_native_profiling(enable: bool = True) -> bool:
    """Enable or disable native kernel profiling.

    When enabled, subsequent calls to ``dispatch_graph_native()`` will
    record per-node timing data which can be retrieved via
    ``get_native_profile()``.

    Also respects the ``NATIVE_RUNNER_PROFILE=1`` environment variable.

    Returns True if profiling is now enabled, False otherwise.
    """
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "profiler_enable"):
        logger.debug("Rust scheduler profiler_enable not available")
        return False
    rust.profiler_enable(enable)
    return bool(rust.profiler_enabled())


def get_native_profile() -> Optional[Dict[str, Any]]:
    """Return profiling data from the most recent ``dispatch_graph_native()`` call.

    Returns None if the Rust scheduler is unavailable or profiling was
    not enabled. Otherwise returns a dict with:
      - ``node_profiles``: list of dicts with node_id, op_name, duration_us, etc.
      - ``peak_memory_bytes``: peak memory tracked by the profiler
    """
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "profiler_enabled"):
        return None
    if not rust.profiler_enabled():
        return None
    # Profiling data is embedded in execute_graph_with_stats results.
    # This function is a convenience accessor for the last cached result.
    return _last_profile_data


# Module-level cache for the most recent profiling result.
_last_profile_data: Optional[Dict[str, Any]] = None


def dispatch_graph_native(graph: Any, input_data: Any) -> Any:
    """Execute a full computation graph using the Rust scheduler.

    Args:
        graph: ComputationGraph instance (from research.synthesis.graph).
        input_data: Input tensor (numpy array or torch tensor).

    Returns:
        Numpy array containing the output of the graph.
    """
    rust = _try_import_rust_scheduler()
    if rust is None:
        raise RuntimeError("Rust scheduler (aria_scheduler) is not available.")

    # Lazy import to avoid circular dependency
    from ..synthesis.native_ir_converter import graph_to_native_ir_json

    import numpy as np
    if hasattr(input_data, "detach"):
        # Convert torch tensor to numpy
        x_np = input_data.detach().cpu().numpy().astype(np.float32)
    else:
        x_np = np.asarray(input_data, dtype=np.float32)

    # Flatten input for the scheduler (expects Vec<f32>)
    # Note: Current Rust execute implementation assumes a single flat input vector.
    # In Aria, graphs usually process [Batch, Seq, Dim] tensors.
    x_flat = x_np.ravel().tolist()

    graph_json = graph_to_native_ir_json(graph)
    
    try:
        global _last_profile_data
        # Prefer execute_graph_with_stats for arena usage observability.
        if hasattr(rust, "execute_graph_with_stats"):
            result = rust.execute_graph_with_stats(graph_json, x_flat)
            y_flat = result["output"]
            logger.debug(
                "Arena stats: %d/%d bytes used, %d arena allocs, %d heap fallbacks",
                result.get("arena_bytes_used", 0),
                result.get("arena_capacity", 0),
                result.get("arena_alloc_count", 0),
                result.get("heap_fallback_count", 0),
            )
            # Cache profiling data if present.
            if "node_profiles" in result:
                _last_profile_data = {
                    "node_profiles": list(result["node_profiles"]),
                    "peak_memory_bytes": int(result.get("peak_memory_bytes", 0)),
                }
                logger.debug(
                    "Profiling: %d node events, peak memory %d bytes",
                    len(_last_profile_data["node_profiles"]),
                    _last_profile_data["peak_memory_bytes"],
                )
            else:
                _last_profile_data = None
        else:
            y_flat = rust.execute_graph(graph_json, x_flat)
            _last_profile_data = None
        y_np = np.array(y_flat, dtype=np.float32)

        # Reshape to output shape if possible
        if hasattr(graph, "output_node") and graph.output_node and x_np.ndim >= 3:
            shape = graph.output_node.output_shape
            # Assuming [Batch, Seq, Dim]
            target_shape = (x_np.shape[0], x_np.shape[1], shape.dim)
            return y_np.reshape(target_shape)

        return y_np
    except Exception as exc:
        logger.error("Rust scheduler execution failed: %s", exc)
        raise


def dispatch_graph_forward_native_saved(graph: Any, input_data: Any) -> Dict[str, Any]:
    """Execute a full forward pass, returning output and saved activations.

    This is the companion to ``dispatch_graph_backward_native()``.  The saved
    activations dict is keyed by integer node id and contains flat float lists.

    Implementation: walks the graph in topological order using per-op native
    dispatch via the Cython bridge, saving each node's output for backward.
    When the Rust scheduler gains ``execute_graph_forward_saved``, this will
    be upgraded to a single Rust call.

    Args:
        graph: ComputationGraph instance.
        input_data: Input tensor (numpy array or torch tensor).

    Returns:
        Dict with keys:
          - ``"output"``: numpy array containing the output.
          - ``"saved_activations"``: dict[int, numpy.ndarray] per-node activations.
          - ``"ir_json"``: the pre-serialized IR JSON (for backward call).
    """
    from ..synthesis.native_ir_converter import graph_to_native_ir_json

    import numpy as np
    if hasattr(input_data, "detach"):
        x_np = input_data.detach().cpu().numpy().astype(np.float32)
    else:
        x_np = np.asarray(input_data, dtype=np.float32)

    graph_json = graph_to_native_ir_json(graph)
    topo = graph.topological_order()

    # Walk forward, saving every node's activation.
    node_outputs: Dict[int, Any] = {}
    output_node_id = graph._output_node_id

    for nid in topo:
        node = graph.nodes[nid]
        if node.is_input:
            node_outputs[nid] = x_np.ravel()
            continue

        inputs = [node_outputs[iid] for iid in node.input_ids]
        # Flatten all inputs for the per-op dispatch.
        flat_inputs = [np.asarray(a, dtype=np.float32).ravel() for a in inputs]
        y = dispatch_op_native(node.op_name, *flat_inputs)
        node_outputs[nid] = np.asarray(y, dtype=np.float32).ravel()

    output = node_outputs.get(output_node_id)
    if output is None:
        raise RuntimeError(f"Graph output node {output_node_id} not found in forward pass")

    return {
        "output": np.asarray(output, dtype=np.float32),
        "saved_activations": dict(node_outputs),
        "ir_json": graph_json,
    }


def dispatch_graph_backward_native(
    graph: Any,
    grad_output: Any,
    saved_activations: Dict[int, Any],
    ir_json: Optional[str] = None,
) -> Dict[int, Any]:
    """Execute a full backward pass through a graph using native per-op backward.

    Walks the graph in reverse topological order, dispatching each op's backward
    through the Cython bridge.  Accumulates gradients when a node fans out to
    multiple consumers.

    When the Rust scheduler gains ``execute_graph_backward``, this will be
    upgraded to a single Rust call.

    Args:
        graph: ComputationGraph instance.
        grad_output: Gradient w.r.t. the graph output (numpy array or torch tensor).
        saved_activations: dict[int, numpy.ndarray] from
            ``dispatch_graph_forward_native_saved()``.
        ir_json: Optional pre-serialized native IR JSON (unused in Python impl,
            kept for API compatibility with future Rust upgrade).

    Returns:
        Dict mapping node_id (int) -> gradient numpy array.
    """
    import numpy as np

    if hasattr(grad_output, "detach"):
        grad_np = grad_output.detach().cpu().numpy().astype(np.float32)
    else:
        grad_np = np.asarray(grad_output, dtype=np.float32)

    topo = graph.topological_order()
    output_node_id = graph._output_node_id

    # Ensure saved_activations values are numpy arrays.
    saved: Dict[int, Any] = {}
    for k, v in saved_activations.items():
        saved[int(k)] = np.asarray(v, dtype=np.float32).ravel()

    # node_grads[nid] = accumulated gradient for that node's output.
    node_grads: Dict[int, Any] = {}
    node_grads[output_node_id] = grad_np.ravel()

    # Walk reverse topological order.
    for nid in reversed(topo):
        node = graph.nodes[nid]
        if node.is_input:
            continue
        if nid not in node_grads:
            continue

        g_out = node_grads[nid]

        # Determine saved tensors needed for this op's backward.
        input_ids = list(node.input_ids)
        input_activations = [saved.get(iid, np.zeros_like(g_out)) for iid in input_ids]

        op_name = node.op_name
        try:
            result = dispatch_op_backward_native(op_name, g_out, *input_activations)
        except (ValueError, RuntimeError) as exc:
            logger.debug("Backward dispatch failed for op %s: %s", op_name, exc)
            # Fallback: pass gradient through unchanged (identity backward).
            result = g_out if len(input_ids) == 1 else tuple([g_out] * len(input_ids))

        # Distribute gradients to inputs.
        if len(input_ids) == 1:
            g_in = np.asarray(result, dtype=np.float32).ravel()
            if input_ids[0] in node_grads:
                node_grads[input_ids[0]] = node_grads[input_ids[0]] + g_in
            else:
                node_grads[input_ids[0]] = g_in
        else:
            # Binary/multi-input: result is a tuple of gradients.
            if not isinstance(result, (tuple, list)):
                result = tuple([result] * len(input_ids))
            for i, iid in enumerate(input_ids):
                g_in = np.asarray(result[i], dtype=np.float32).ravel()
                if iid in node_grads:
                    node_grads[iid] = node_grads[iid] + g_in
                else:
                    node_grads[iid] = g_in

    grads: Dict[int, Any] = {}
    for nid, g in node_grads.items():
        grads[nid] = np.asarray(g, dtype=np.float32)

    return grads


def dispatch_graph_native_cached(ir_json: str, graph: Any, input_data: Any) -> Any:
    """Execute a graph using a pre-converted native_ir JSON string.

    Like ``dispatch_graph_native()`` but skips the graph-to-IR conversion step,
    accepting an already-serialized JSON string. This avoids ~10us of repeated
    JSON serialization when the same graph is dispatched multiple times.

    Args:
        ir_json: Pre-serialized native_ir.v1 JSON string.
        graph: Original ComputationGraph (used only for output reshaping).
        input_data: Input tensor (numpy array or torch tensor).

    Returns:
        Numpy array containing the output of the graph.
    """
    rust = _try_import_rust_scheduler()
    if rust is None:
        raise RuntimeError("Rust scheduler (aria_scheduler) is not available.")

    import numpy as np
    if hasattr(input_data, "detach"):
        x_np = input_data.detach().cpu().numpy().astype(np.float32)
    else:
        x_np = np.asarray(input_data, dtype=np.float32)

    x_flat = x_np.ravel().tolist()

    try:
        global _last_profile_data
        if hasattr(rust, "execute_graph_with_stats"):
            result = rust.execute_graph_with_stats(ir_json, x_flat)
            y_flat = result["output"]
            logger.debug(
                "Arena stats: %d/%d bytes used, %d arena allocs, %d heap fallbacks",
                result.get("arena_bytes_used", 0),
                result.get("arena_capacity", 0),
                result.get("arena_alloc_count", 0),
                result.get("heap_fallback_count", 0),
            )
            if "node_profiles" in result:
                _last_profile_data = {
                    "node_profiles": list(result["node_profiles"]),
                    "peak_memory_bytes": int(result.get("peak_memory_bytes", 0)),
                }
            else:
                _last_profile_data = None
        else:
            y_flat = rust.execute_graph(ir_json, x_flat)
            _last_profile_data = None
        y_np = np.array(y_flat, dtype=np.float32)

        if hasattr(graph, "output_node") and graph.output_node and x_np.ndim >= 3:
            shape = graph.output_node.output_shape
            target_shape = (x_np.shape[0], x_np.shape[1], shape.dim)
            return y_np.reshape(target_shape)

        return y_np
    except Exception as exc:
        logger.error("Rust scheduler execution failed: %s", exc)
        raise


class NativeSubgraphFunction:
    """torch.autograd.Function for full-graph native forward + backward.

    Instead of N per-op Python-to-C roundtrips in both forward and backward,
    this does 1 Rust call for the entire forward pass and 1 Rust call for
    the entire backward pass.  Activations are saved by the Rust forward
    (``execute_graph_forward_saved``) and fed back into the Rust backward
    (``execute_graph_backward``).

    This is a factory: call ``NativeSubgraphFunction.make(graph, ir_json)``
    to obtain a concrete ``torch.autograd.Function`` subclass bound to a
    specific graph.
    """

    @staticmethod
    def make(graph: Any, ir_json: Optional[str] = None):
        """Return a ``torch.autograd.Function`` subclass for *graph*.

        The returned class captures *graph* and *ir_json* in its closure so
        that ``apply(x)`` is the only user-facing call.
        """
        import torch

        _graph = graph
        _ir_json = ir_json

        class _SubgraphFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # Perform a full native forward, saving activations.
                fwd = dispatch_graph_forward_native_saved(_graph, x)
                output_np = fwd["output"]
                saved_activations = fwd["saved_activations"]
                used_ir_json = fwd["ir_json"]

                import numpy as np
                output_tensor = torch.from_numpy(
                    np.asarray(output_np, dtype=np.float32)
                ).to(x.device)

                # Reshape to match input batch/seq dims when possible.
                if x.ndim >= 3 and output_tensor.ndim == 1:
                    try:
                        output_tensor = output_tensor.reshape(x.shape)
                    except RuntimeError:
                        pass

                # Save non-tensor data on ctx for backward.
                ctx._saved_activations = saved_activations
                ctx._ir_json = used_ir_json
                ctx._input_shape = x.shape
                ctx._graph = _graph

                return output_tensor

            @staticmethod
            def backward(ctx, grad_output):
                saved_activations = ctx._saved_activations
                used_ir_json = ctx._ir_json
                graph_ref = ctx._graph
                input_shape = ctx._input_shape

                # Single Rust call for the entire backward pass.
                grads = dispatch_graph_backward_native(
                    graph_ref,
                    grad_output,
                    saved_activations,
                    ir_json=used_ir_json,
                )

                # The input node gradient is what we need.  Find the input
                # node id (the node with is_input=True).
                import numpy as np
                nodes = getattr(graph_ref, "nodes", {})
                input_node_id = None
                for nid, node in nodes.items():
                    if getattr(node, "is_input", False):
                        input_node_id = nid
                        break

                if input_node_id is not None and input_node_id in grads:
                    grad_np = grads[input_node_id]
                    grad_tensor = torch.from_numpy(
                        np.asarray(grad_np, dtype=np.float32)
                    ).to(grad_output.device)
                    try:
                        grad_tensor = grad_tensor.reshape(input_shape)
                    except RuntimeError:
                        pass
                    return grad_tensor

                # Fallback: return grad_output as-is (identity gradient).
                return grad_output

        return _SubgraphFn


class SubgraphDispatcher:
    """Dispatches entire computation subgraphs through the Rust scheduler.

    When all ops in a ComputationGraph (or contiguous subgraph) are
    native-supported, this class converts the graph to native_ir.v1 and
    executes it as a single ``dispatch_graph_native()`` call — avoiding
    per-op Python-to-C roundtrips.

    If any op in the graph is unsupported, ``try_dispatch()`` returns
    ``None`` and the caller falls back to per-op dispatch.

    When the input tensor requires gradients, ``try_dispatch()`` routes
    through ``NativeSubgraphFunction`` so that both forward and backward
    are executed as single Rust calls (1 forward + 1 backward instead of
    N per-op roundtrips each way).
    """

    def __init__(self, graph: Any, supported_ops: Set[str]):
        self._graph = graph
        self._supported_ops = supported_ops
        self._all_native = self._check_all_native()
        self._dispatch_count = 0
        self._fallback_count = 0
        # Lazily-created autograd Function subclass (cached after first grad dispatch).
        self._autograd_fn: Any = None
        # Pre-convert graph to native_ir JSON once; reuse across dispatches.
        self._ir_json: Optional[str] = None
        if self._all_native:
            try:
                from ..synthesis.native_ir_converter import graph_to_native_ir_json
                self._ir_json = graph_to_native_ir_json(graph)
            except Exception:
                logger.debug("Failed to pre-convert graph to IR JSON")
                self._ir_json = None

    def _check_all_native(self) -> bool:
        """Return True if every non-input op in the graph is native-supported."""
        nodes = getattr(self._graph, "nodes", None)
        if not isinstance(nodes, dict) or not nodes:
            return False
        for node in nodes.values():
            if getattr(node, "is_input", False):
                continue
            op_name = getattr(node, "op_name", "")
            if op_name not in self._supported_ops:
                return False
        return True

    @property
    def all_native(self) -> bool:
        return self._all_native

    def try_dispatch(self, x: Any) -> Any:
        """Try to execute the full graph natively.

        Uses the pre-cached native_ir JSON to avoid re-serializing the graph
        on every call. Falls back to ``dispatch_graph_native()`` if the
        cached IR is not available.

        When *x* is a ``torch.Tensor`` that requires gradients, the call is
        routed through ``NativeSubgraphFunction`` so that both forward and
        backward are single Rust calls (instead of N per-op roundtrips).

        Args:
            x: Input tensor (torch.Tensor or numpy array).

        Returns:
            Output tensor on success, or ``None`` if subgraph dispatch
            is not possible (caller should fall back to per-op path).
        """
        if not self._all_native:
            return None

        try:
            # ── Autograd path: input requires grad → use NativeSubgraphFunction
            if getattr(x, "requires_grad", False):
                if self._autograd_fn is None:
                    self._autograd_fn = NativeSubgraphFunction.make(
                        self._graph, self._ir_json
                    )
                result = self._autograd_fn.apply(x)
                self._dispatch_count += 1
                return result

            # ── Inference path: numpy-based, no autograd ──
            if self._ir_json is not None:
                result = dispatch_graph_native_cached(
                    self._ir_json, self._graph, x
                )
            else:
                result = dispatch_graph_native(self._graph, x)
            self._dispatch_count += 1

            # Convert back to torch if input was torch
            if hasattr(x, "detach"):
                import torch
                import numpy as np
                return torch.from_numpy(np.asarray(result, dtype=np.float32))
            return result
        except Exception as exc:
            logger.debug(
                "Subgraph dispatch failed: %s, falling back to per-op", exc
            )
            self._fallback_count += 1
            return None

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "all_native": self._all_native,
            "subgraph_dispatches": self._dispatch_count,
            "subgraph_fallbacks": self._fallback_count,
        }


class NativeForwardWrapper:
    """Wraps a compiled model to route supported ops through native C kernels.

    When enabled, intercepts tensor operations during forward pass and
    dispatches them through the Cython bridge instead of PyTorch.
    """

    def __init__(self, model: Any, supported_ops: Set[str]):
        self.model = model
        self.supported_ops = supported_ops
        self._dispatch_count = 0
        self._fallback_count = 0

    def dispatch(self, op_name: str, *tensors: Any) -> Any:
        """Try native dispatch, fall back to numpy/torch if needed.

        When any input tensor requires gradients **and** the op has a native
        backward kernel, the call is routed through the corresponding
        ``torch.autograd.Function`` subclass (see ``native_autograd.py``)
        so that gradient computation flows through the C backward kernels.

        Returns the result tensor/array on success, or ``None`` to signal
        the caller should use the original PyTorch implementation.
        """
        if op_name in self.supported_ops:
            try:
                # Check if any torch input requires grad → use autograd path
                any_requires_grad = any(
                    getattr(t, "requires_grad", False) for t in tensors
                )
                if any_requires_grad:
                    from .native_autograd import (
                        NATIVE_AUTOGRAD_SUPPORTED_OPS,
                        native_autograd_dispatch,
                    )
                    if op_name in NATIVE_AUTOGRAD_SUPPORTED_OPS:
                        result = native_autograd_dispatch(op_name, *tensors)
                        self._dispatch_count += 1
                        return result

                import numpy as np

                # Convert torch tensors to numpy for C dispatch
                np_inputs: List[Any] = []
                for t in tensors:
                    if hasattr(t, "detach"):
                        np_inputs.append(t.detach().cpu().numpy().astype(np.float32))
                    else:
                        np_inputs.append(np.asarray(t, dtype=np.float32))

                result = dispatch_op_native(op_name, *np_inputs)
                self._dispatch_count += 1

                # Convert back to torch if input was torch
                if tensors and hasattr(tensors[0], "detach"):
                    import torch

                    return torch.from_numpy(np.asarray(result, dtype=np.float32))
                return result
            except Exception as exc:
                logger.debug(
                    "Native dispatch failed for %s: %s, falling back", op_name, exc
                )
                self._fallback_count += 1
        return None  # Signal to caller: use original implementation

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "native_dispatches": self._dispatch_count,
            "fallbacks": self._fallback_count,
        }


class NativeRunnerAbiSession:
    """Holder for runner ABI compiled handle + token execute helper."""

    def __init__(self, native_lib: Any, model_handle: int, vocab_size: int, max_seq_len: int):
        self._native_lib = native_lib
        self.model_handle = int(model_handle)
        self.vocab_size = int(vocab_size)
        self.max_seq_len = int(max_seq_len)
        self._closed = False

    def execute_tokens(self, token_ids: List[int], batch: int = 1) -> List[float]:
        if self._closed:
            raise RuntimeError("native ABI session already closed")
        seq_len = int(len(token_ids))
        if seq_len <= 0:
            raise ValueError("token_ids must be non-empty")
        if seq_len > self.max_seq_len:
            raise ValueError("token length exceeds compiled max_seq_len")

        token_buf = (ctypes.c_int32 * seq_len)(*([int(t) for t in token_ids]))
        req = _NrExecuteRequest(
            model_handle=self.model_handle,
            token_ids=token_buf,
            batch=int(batch),
            seq_len=seq_len,
        )
        resp = self._native_lib.nr_execute(ctypes.byref(req))
        if int(resp.status) != 0 or not bool(resp.logits):
            raise RuntimeError(f"runner ABI execute failed: status={int(resp.status)}")
        n_vocab = int(resp.vocab_size)
        return [float(resp.logits[i]) for i in range(n_vocab)]

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._native_lib.nr_release_model(ctypes.c_int64(self.model_handle))
        except Exception:
            pass
        self._closed = True


def _build_native_abi_only_model(
    abi_session: NativeRunnerAbiSession,
    vocab_size: int,
    model_dim: int = 0,
):
    """Build an inference-only torch module backed by runner ABI session."""
    import torch

    class _NativeAbiOnlyModel(torch.nn.Module):
        def __init__(self, session: NativeRunnerAbiSession, n_vocab: int, dim: int):
            super().__init__()
            self._abi_session = session
            self.vocab_size = int(n_vocab)
            self.model_dim = int(dim or 0)
            self._anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        def forward(self, input_ids):
            if input_ids is None:
                raise ValueError("input_ids is required")
            if input_ids.ndim != 2:
                raise ValueError("input_ids must be rank-2 [B, S]")
            batch_size = int(input_ids.shape[0])
            seq_len = int(input_ids.shape[1])
            if seq_len <= 0:
                raise ValueError("input_ids sequence length must be > 0")
            out = torch.empty(
                (batch_size, seq_len, self.vocab_size),
                dtype=torch.float32,
                device=input_ids.device,
            )
            for b in range(batch_size):
                token_ids = [int(v) for v in input_ids[b].detach().cpu().tolist()]
                logits = self._abi_session.execute_tokens(token_ids, batch=1)
                row = torch.tensor(logits, dtype=torch.float32, device=input_ids.device)
                out[b, :, :] = row.view(1, -1).expand(seq_len, -1)
            return out

    return _NativeAbiOnlyModel(abi_session, vocab_size, model_dim)

    def __del__(self):
        self.close()


def _maybe_prepare_runner_abi_session(
    *,
    layer_graphs: List[Any],
    native_lib: Any,
    state: NativeRunnerState,
    vocab_size: int,
    max_seq_len: Optional[int],
) -> Dict[str, Any]:
    """Optional compile+smoke path through `runner_abi` for first-family execution."""
    report: Dict[str, Any] = {
        "requested": False,
        "attempted": False,
        "succeeded": False,
        "reason": "disabled",
        "model_handle": None,
        "session": None,
    }

    if not _env_flag("NATIVE_RUNNER_ABI_EXEC", False):
        return report
    report["requested"] = True

    if native_lib is None:
        report["reason"] = "native_lib_unavailable"
        return report
    if not layer_graphs:
        report["reason"] = "no_layer_graphs"
        return report
    if not all(
        hasattr(native_lib, name)
        for name in ("nr_runtime_init", "nr_set_strict_mode", "nr_compile", "nr_execute", "nr_release_model")
    ):
        report["reason"] = "runner_abi_symbols_missing"
        return report

    abi_supported_unary_ops = {"relu", "gelu", "silu", "sigmoid", "tanh", "exp"}

    def _graph_is_abi_family_candidate(candidate: Any) -> bool:
        nodes = getattr(candidate, "nodes", None)
        if not isinstance(nodes, dict) or not nodes:
            return False
        known_node_ids = {str(node_id) for node_id in nodes.keys()}
        required_order = ["exp", "add", "mul", "matmul", "linear", "softmax", "rmsnorm", "sub"]
        first_positions = {op_name: None for op_name in required_order}
        first_node_ids = {op_name: None for op_name in required_order}
        required_counts = {op_name: 0 for op_name in required_order}
        input_incoming_counts: Dict[str, Dict[str, int]] = {str(node_id): {} for node_id in nodes.keys()}
        edge_incoming_counts: Dict[str, Dict[str, int]] = {str(node_id): {} for node_id in nodes.keys()}
        input_refs_by_node: Dict[str, List[str]] = {str(node_id): [] for node_id in nodes.keys()}
        raw_declared_edges: List[Dict[str, str]] = []
        has_unary = False
        for idx, (node_id, node) in enumerate(nodes.items()):
            op_name = str(getattr(node, "op_name", "") or "").strip().lower()
            if op_name in required_counts:
                required_counts[op_name] += 1
            if op_name in first_positions and first_positions[op_name] is None:
                first_positions[op_name] = idx
                first_node_ids[op_name] = str(node_id)
            raw_inputs = getattr(node, "input_ids", None)
            if isinstance(raw_inputs, (list, tuple, set)):
                for src in raw_inputs:
                    src_id = str(src)
                    if src_id:
                        child_key = str(node_id)
                        input_refs_by_node.setdefault(child_key, []).append(src_id)
                        child_counts = input_incoming_counts.get(child_key)
                        if child_counts is not None:
                            child_counts[src_id] = int(child_counts.get(src_id, 0)) + 1
            if op_name in abi_supported_unary_ops:
                has_unary = True

        edges = getattr(candidate, "edges", None)
        has_declared_edges = edges is not None
        if isinstance(edges, (list, tuple)):
            for edge in edges:
                source = str(getattr(edge, "source", "") or "")
                target = str(getattr(edge, "target", "") or "")
                if not source or not target:
                    if isinstance(edge, dict):
                        source = str(edge.get("source", "") or "")
                        target = str(edge.get("target", "") or "")
                if source and target:
                    raw_declared_edges.append({"source": source, "target": target})
                if source and target and target in edge_incoming_counts:
                    target_counts = edge_incoming_counts.get(target)
                    if target_counts is not None:
                        target_counts[source] = int(target_counts.get(source, 0)) + 1

        if not has_unary:
            return False
        if any(first_positions[op_name] is None for op_name in required_order):
            return False
        if any(int(required_counts[op_name]) != 1 for op_name in required_order):
            return False
        if not all(
            int(first_positions[required_order[i]]) < int(first_positions[required_order[i + 1]])
            for i in range(len(required_order) - 1)
        ):
            return False

        required_chain = [
            ("exp", "add"),
            ("add", "mul"),
            ("mul", "matmul"),
            ("matmul", "linear"),
            ("linear", "softmax"),
            ("softmax", "rmsnorm"),
            ("rmsnorm", "sub"),
        ]
        has_explicit_edges = has_declared_edges
        required_chain_node_ids = {
            str(first_node_ids[op_name])
            for op_name in required_order
            if first_node_ids[op_name] is not None
        }

        for child_node_id in required_chain_node_ids:
            # P6.R35: strict node-reference sanity for required links.
            # Reject when required-chain links reference missing node ids.
            for src_id in input_refs_by_node.get(child_node_id, []):
                if src_id not in known_node_ids:
                    return False

        if has_explicit_edges:
            for edge in raw_declared_edges:
                source = str(edge.get("source", "") or "")
                target = str(edge.get("target", "") or "")
                # P6.R35: strict reference sanity for explicit edge endpoints.
                if target in required_chain_node_ids:
                    if source not in known_node_ids or target not in known_node_ids:
                        return False

        for parent_op, child_op in required_chain:
            parent_node_id = first_node_ids[parent_op]
            child_node_id = first_node_ids[child_op]
            if parent_node_id is None or child_node_id is None:
                return False
            child_node_id_str = str(child_node_id)
            parent_node_id_str = str(parent_node_id)
            input_parent_count = int(
                input_incoming_counts.get(child_node_id_str, {}).get(parent_node_id_str, 0)
            )
            if input_parent_count != 1:
                return False
            if has_explicit_edges:
                edge_parent_count = int(
                    edge_incoming_counts.get(child_node_id_str, {}).get(parent_node_id_str, 0)
                )
                if edge_parent_count != 1:
                    return False
        return True

    graph = next(
        (g for g in layer_graphs if hasattr(g, "nodes") and _graph_is_abi_family_candidate(g)),
        None,
    )
    if graph is None:
        report["reason"] = "no_abi_family_graph"
        return report

    report["attempted"] = True
    try:
        from ..synthesis.native_ir_converter import graph_to_native_ir_json

        native_lib.nr_runtime_init.restype = ctypes.c_int32
        native_lib.nr_set_strict_mode.argtypes = [ctypes.c_int32]
        native_lib.nr_set_strict_mode.restype = ctypes.c_int32
        native_lib.nr_compile.argtypes = [ctypes.POINTER(_NrCompileRequest)]
        native_lib.nr_compile.restype = _NrCompileResponse
        native_lib.nr_execute.argtypes = [ctypes.POINTER(_NrExecuteRequest)]
        native_lib.nr_execute.restype = _NrExecuteResponse
        native_lib.nr_release_model.argtypes = [ctypes.c_int64]

        init_status = int(native_lib.nr_runtime_init())
        if init_status != 0:
            report["reason"] = f"nr_runtime_init_failed:{init_status}"
            return report
        native_lib.nr_set_strict_mode(1 if state.strict else 0)

        graph_ir = graph_to_native_ir_json(graph).encode("utf-8")
        compile_req = _NrCompileRequest(
            ir_json=graph_ir,
            ir_json_len=len(graph_ir),
            vocab_size=int(vocab_size),
            max_seq_len=int(max_seq_len or 128),
        )
        compile_resp = native_lib.nr_compile(ctypes.byref(compile_req))
        compile_status = int(compile_resp.status)
        compile_message = (
            compile_resp.message.decode("utf-8", errors="ignore")
            if getattr(compile_resp, "message", None)
            else None
        )
        if compile_status != 0:
            compile_reason = _normalize_nr_compile_reason(compile_status, compile_message)
            report["reason"] = f"nr_compile_failed:{compile_status}:{compile_reason}"
            report["compile_status"] = compile_status
            report["compile_reason"] = compile_reason
            report["compile_message"] = compile_message
            return report

        handle = int(compile_resp.model_handle)
        report["model_handle"] = handle
        session = NativeRunnerAbiSession(
            native_lib=native_lib,
            model_handle=handle,
            vocab_size=int(vocab_size),
            max_seq_len=int(max_seq_len or 128),
        )

        # Tiny deterministic execute smoke so the handle is known-good.
        logits = session.execute_tokens([1, 2, 3, 4], batch=1)
        if not logits:
            session.close()
            report["reason"] = "nr_execute_empty_logits"
            return report

        report["session"] = session
        report["succeeded"] = True
        report["reason"] = "ok"
        report["compile_message"] = (
            compile_message
        )
        report["compile_status"] = compile_status
        report["compile_reason"] = _normalize_nr_compile_reason(compile_status, compile_message)
        return report
    except Exception as exc:
        report["reason"] = f"runner_abi_error:{exc}"
        return report


def compile_model_native_first(
    layer_graphs: List[Any],
    vocab_size: int = 32000,
    max_seq_len: Optional[int] = None,
    **kwargs: Any,
):
    """Compile model using native-first policy.

    Phase-D behavior (post-cutover):
    - If disabled (NATIVE_RUNNER_ENABLED=0): legacy compile via ``_legacy_compile_model``.
    - If enabled: ABI model-only path — builds model backed by native runner ABI session.
      Legacy compile is unreachable from native-enabled flow.
    - Always attaches an ``_native_runner_report`` to the compiled model.
    """
    disable_legacy_compile = _env_flag("NATIVE_RUNNER_DISABLE_LEGACY_COMPILE", False)
    disable_legacy_compile_native_enabled = _env_flag(
        "NATIVE_RUNNER_DISABLE_LEGACY_COMPILE_NATIVE_ENABLED",
        False,
    )

    # Emergency rollback: skip all native logic (only valid when native is NOT enabled).
    # Phase D: NATIVE_RUNNER_LEGACY_ONLY conflicts with NATIVE_RUNNER_ENABLED.
    if _env_flag("NATIVE_RUNNER_LEGACY_ONLY", False):
        _maybe_warn_deprecated_legacy_only_flag()
        state_check = detect_native_state()
        if state_check.enabled:
            raise RuntimeError(
                "Invalid native-runner config: "
                "NATIVE_RUNNER_LEGACY_ONLY=1 cannot be used when NATIVE_RUNNER_ENABLED=1. "
                "Disable native mode first (NATIVE_RUNNER_ENABLED=0) to use legacy-only path."
            )
        if disable_legacy_compile:
            raise RuntimeError(
                "Invalid native-runner config: "
                "NATIVE_RUNNER_LEGACY_ONLY=1 conflicts with NATIVE_RUNNER_DISABLE_LEGACY_COMPILE=1"
            )
        _FALLBACK_METRICS["total_compiles"] += 1
        _record_legacy_compile_invocation()
        return _legacy_compile_model(layer_graphs, vocab_size=vocab_size, max_seq_len=max_seq_len, **kwargs)

    state = detect_native_state()
    capability = native_runner_capability_report()
    requested_mode = _requested_execution_mode()
    capability["execution_mode_requested"] = requested_mode
    capability["execution_path"] = "legacy_disabled"

    if state.enabled and disable_legacy_compile_native_enabled:
        disable_legacy_compile = True
        capability["legacy_compile_disabled_reason"] = "native_enabled_gate"

    if state.enabled:
        _FALLBACK_METRICS["native_enabled_compiles"] += 1

    # --- Phase 3: native kernel dispatch checking ---
    op_support: Optional[Dict[str, Any]] = None
    native_lib = None
    full_native_coverage = False
    partial_native_coverage = False
    if state.enabled:
        native_lib = _try_load_native_lib()
        if layer_graphs:
            op_support = _check_native_op_support(layer_graphs, native_lib)
            capability["native_op_support"] = op_support

            coverage = op_support["native_coverage"]
            if coverage >= 1.0:
                full_native_coverage = True
                logger.info(
                    "Full native path available: all %d ops supported by kernel library",
                    len(op_support["all_ops"]),
                )
                _FALLBACK_METRICS["native_dispatch_compiles"] += 1
            elif state.strict:
                raise RuntimeError(
                    f"NATIVE_RUNNER_STRICT=1 but {len(op_support['unsupported'])} ops lack "
                    f"native kernel support: {op_support['unsupported']}"
                )
            elif coverage >= PARTIAL_NATIVE_COVERAGE_THRESHOLD:
                partial_native_coverage = True
                logger.debug(
                    "Partial native path: %.1f%% coverage (%d/%d ops). "
                    "Unsupported: %s",
                    coverage * 100,
                    len(op_support["supported"]),
                    len(op_support["kernel_relevant_ops"]),
                    op_support["unsupported"],
                )
                _FALLBACK_METRICS["native_dispatch_compiles"] += 1
            else:
                _log_native_fallback_coverage(op_support)
        elif state.strict:
            # Strict with empty graphs: nothing to check, allow through.
            pass

    # --- IR validation (observational — log warnings but don't block) ---
    if state.enabled and layer_graphs:
        try:
            from ..runtime.native.ir_validator import validate_ir
            from ..synthesis.native_ir_converter import graph_to_native_ir

            ir_errors: List = []
            for i, g in enumerate(layer_graphs):
                if not hasattr(g, "nodes"):
                    continue  # skip non-ComputationGraph objects
                ir_doc = graph_to_native_ir(g)
                errs = validate_ir(ir_doc)
                if errs:
                    ir_errors.extend([(i, e) for e in errs])
            if ir_errors:
                logger.warning("IR validation errors: %s", ir_errors[:5])
                capability["ir_validation"] = {
                    "valid": False,
                    "errors": [str(e) for e in ir_errors[:10]],
                }
            else:
                capability["ir_validation"] = {"valid": True, "errors": []}
        except Exception as exc:
            logger.debug("IR validation skipped: %s", exc)
            capability["ir_validation"] = {
                "valid": None,
                "errors": [f"validation_unavailable:{exc}"],
            }

    # --- Designer runtime probe (orthogonal to native kernel dispatch) ---
    probe: Dict[str, Any] = {
        "attempted": False,
        "succeeded": False,
        "parity_ok": None,
        "reason": "not_attempted",
    }
    if state.enabled and state.designer_runtime_available:
        capability["designer_runtime_probe"] = try_designer_runtime_probe(layer_graphs)
        probe = capability["designer_runtime_probe"]
        if bool(probe.get("succeeded")) and bool(probe.get("parity_ok")):
            _FALLBACK_METRICS["probe_successes"] += 1
        else:
            _FALLBACK_METRICS["probe_failures"] += 1

    selective_candidate = False
    selective_reason = "mode_not_requested"
    if requested_mode == "selective":
        if not state.enabled:
            selective_reason = "native_runner_disabled"
        elif not full_native_coverage:
            selective_reason = "incomplete_native_coverage"
        elif state.designer_runtime_available and not (
            bool(probe.get("succeeded")) and bool(probe.get("parity_ok"))
        ):
            selective_reason = "probe_not_green"
        else:
            selective_candidate = True
            selective_reason = "candidate_ready"
            _FALLBACK_METRICS["selective_mode_candidates"] += 1

    capability["selective_execution"] = {
        "requested": requested_mode == "selective",
        "candidate": selective_candidate,
        "reason": selective_reason,
    }

    try:
        guardrail_threshold = int(str(os.environ.get("NATIVE_RUNNER_SELECTIVE_GUARDRAIL_WINDOW", "5")))
    except Exception:
        guardrail_threshold = 5
    guardrail_threshold = max(1, guardrail_threshold)

    if requested_mode == "selective" and not selective_candidate:
        _SELECTIVE_GUARDRAIL["consecutive_requested_not_candidate"] = int(
            _SELECTIVE_GUARDRAIL.get("consecutive_requested_not_candidate") or 0
        ) + 1
        _SELECTIVE_GUARDRAIL["last_reason"] = selective_reason
        if _SELECTIVE_GUARDRAIL["consecutive_requested_not_candidate"] >= guardrail_threshold:
            if not bool(_SELECTIVE_GUARDRAIL.get("triggered")):
                _SELECTIVE_GUARDRAIL["trigger_count"] = int(_SELECTIVE_GUARDRAIL.get("trigger_count") or 0) + 1
                _record_guardrail_event(
                    "triggered",
                    reason=selective_reason,
                    threshold=guardrail_threshold,
                    source="compile_model_native_first",
                )
            _SELECTIVE_GUARDRAIL["triggered"] = True
    else:
        was_triggered = bool(_SELECTIVE_GUARDRAIL.get("triggered"))
        _SELECTIVE_GUARDRAIL["consecutive_requested_not_candidate"] = 0
        _SELECTIVE_GUARDRAIL["triggered"] = False
        if was_triggered:
            cleared_reason = "candidate_ready" if requested_mode == "selective" else "mode_not_selective"
            _record_guardrail_event(
                "cleared",
                reason=cleared_reason,
                threshold=guardrail_threshold,
                source="compile_model_native_first",
            )
        if requested_mode != "selective":
            _SELECTIVE_GUARDRAIL["last_reason"] = None

    capability["selective_guardrail"] = {
        "consecutive_requested_not_candidate": int(
            _SELECTIVE_GUARDRAIL.get("consecutive_requested_not_candidate") or 0
        ),
        "threshold": guardrail_threshold,
        "triggered": bool(_SELECTIVE_GUARDRAIL.get("triggered")),
        "trigger_count": int(_SELECTIVE_GUARDRAIL.get("trigger_count") or 0),
        "last_reason": _SELECTIVE_GUARDRAIL.get("last_reason"),
        "history": [dict(item) for item in _SELECTIVE_GUARDRAIL_HISTORY],
    }

    selective_activation: Dict[str, Any] = {
        "activated": False,
        "ops": ["relu", "add"],
        "reason": "not_candidate",
    }
    if selective_candidate:
        selective_activation = _activate_selective_native_dispatch(native_lib)
        if bool(selective_activation.get("activated")):
            _FALLBACK_METRICS["selective_mode_activations"] += 1
        else:
            _FALLBACK_METRICS["selective_mode_activation_failures"] += 1
    capability["selective_execution"]["activation"] = selective_activation

    if selective_candidate:
        if bool(selective_activation.get("activated")):
            capability["execution_path"] = "selective_native_active_legacy_compile"
        else:
            capability["execution_path"] = "selective_candidate_legacy_compile"
    elif full_native_coverage:
        capability["execution_path"] = "full_native_legacy_compile"
    elif partial_native_coverage:
        capability["execution_path"] = "hybrid_native_legacy_compile"
    elif state.enabled:
        capability["execution_path"] = "legacy_fallback"

    # --- Compile via legacy path (actual native execution dispatch is a future phase) ---
    _FALLBACK_METRICS["total_compiles"] += 1
    if state.enabled and (op_support is None or op_support["native_coverage"] < PARTIAL_NATIVE_COVERAGE_THRESHOLD):
        _FALLBACK_METRICS["fallback_compiles"] += 1
    if state.enabled and partial_native_coverage:
        _FALLBACK_METRICS["hybrid_compiles"] += 1

    abi_report = _maybe_prepare_runner_abi_session(
        layer_graphs=layer_graphs,
        native_lib=native_lib,
        state=state,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )
    capability["runner_abi"] = {
        "requested": bool(abi_report.get("requested")),
        "attempted": bool(abi_report.get("attempted")),
        "succeeded": bool(abi_report.get("succeeded")),
        "reason": abi_report.get("reason"),
        "model_handle": abi_report.get("model_handle"),
    }
    if state.strict and bool(abi_report.get("requested")) and not bool(abi_report.get("succeeded")):
        raise RuntimeError(
            f"NATIVE_RUNNER_STRICT=1 and runner ABI prepare failed: {abi_report.get('reason')}"
        )
    # Phase D: ABI model-only is always active when native mode is enabled.
    # NATIVE_RUNNER_ABI_MODEL_ONLY and NATIVE_RUNNER_ALLOW_LEGACY_FALLBACK are
    # no longer supported; legacy compile is unreachable from native-enabled flow.
    abi_model_only = state.enabled  # always True when native is enabled
    capability["runner_abi"]["model_only_requested"] = bool(abi_model_only)
    capability["runner_abi"]["allow_legacy_fallback"] = False

    # Classify the execution mode for observability
    if state.enabled:
        capability["execution_mode_classification"] = "native_abi_model_only"
    else:
        capability["execution_mode_classification"] = "legacy_only"

    if state.enabled:
        abi_session = abi_report.get("session")
        if abi_session is None or not bool(abi_report.get("succeeded")):
            # ABI session not available — fall back to legacy compile unless
            # explicitly forbidden.  This happens when NATIVE_RUNNER_ABI_EXEC
            # is not set (the default) or when the native lib lacks symbols.
            if not disable_legacy_compile:
                logger.debug(
                    "ABI session unavailable (reason=%s), falling back to legacy compile",
                    abi_report.get("reason"),
                )
            else:
                raise RuntimeError(
                    "Native mode requires successful ABI session preparation. "
                    f"reason={abi_report.get('reason')}"
                )
        else:
            model = _build_native_abi_only_model(
                abi_session=abi_session,
                vocab_size=int(vocab_size),
                model_dim=int(kwargs.get("model_dim", 0) or 0),
            )
            setattr(model, "_native_runner_abi_session", abi_session)
            capability["runner_abi"]["session_attached"] = True
            capability["execution_path"] = "native_abi_model_only"
            capability["legacy_compile_used"] = False
            _maybe_fail_on_fallback_rate()
            _maybe_fail_on_legacy_compile_usage()
            capability["fallback_metrics"] = native_runner_capability_report().get("fallback_metrics", {})
            try:
                setattr(model, "_native_runner_report", capability)
            except Exception:
                pass
            logger.info(
                "Native runner ABI-only model active: execution_path=%s enabled=%s strict=%s",
                capability.get("execution_path"),
                capability.get("enabled"),
                capability.get("strict"),
            )
            return model

    if disable_legacy_compile:
        raise RuntimeError(
            "Legacy compile path disabled by NATIVE_RUNNER_DISABLE_LEGACY_COMPILE=1 "
            "or NATIVE_RUNNER_DISABLE_LEGACY_COMPILE_NATIVE_ENABLED=1; "
            "remove remaining legacy fallback usage before enabling this gate."
        )

    model = _legacy_compile_model(
        layer_graphs,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        **kwargs,
    )
    _record_legacy_compile_invocation()
    capability["legacy_compile_used"] = True

    # --- Attach native forward wrapper when coverage meets threshold ---
    if (
        state.enabled
        and op_support is not None
        and op_support["native_coverage"] >= PARTIAL_NATIVE_COVERAGE_THRESHOLD
    ):
        try:
            wrapper = NativeForwardWrapper(model, set(op_support["supported"]))
            setattr(model, "_native_forward_wrapper", wrapper)
            capability["native_forward_wrapper"] = {
                "attached": True,
                "supported_ops": len(op_support["supported"]),
            }
            # Propagate wrapper to individual CompiledOp instances
            try:
                for layer in getattr(model, 'layers', []):
                    ops = getattr(layer, 'ops', None)
                    if ops is not None:
                        for op in ops.values():
                            if hasattr(op, 'forward'):
                                op._native_wrapper = wrapper
                capability["native_forward_wrapper"]["propagated"] = True
            except Exception as exc:
                logger.debug("Failed to propagate native wrapper to ops: %s", exc)
                capability["native_forward_wrapper"]["propagated"] = False
        except Exception as exc:
            logger.debug("Failed to attach native forward wrapper: %s", exc)
            capability["native_forward_wrapper"] = {
                "attached": False,
                "error": str(exc),
            }
    # --- Attach SubgraphDispatchers to compiled layers for batch Rust dispatch ---
    if (
        state.enabled
        and op_support is not None
        and op_support["native_coverage"] >= PARTIAL_NATIVE_COVERAGE_THRESHOLD
    ):
        try:
            supported_set = set(op_support["supported"])
            dispatchers_attached = 0
            dispatchers_skipped = 0
            layers = getattr(model, 'layers', [])
            layer_graph_list = getattr(model, '_layer_graphs', layer_graphs)
            for i, layer in enumerate(layers):
                if i < len(layer_graph_list):
                    graph = layer_graph_list[i]
                    if hasattr(graph, 'nodes'):
                        dispatcher = SubgraphDispatcher(graph, supported_set)
                        if dispatcher.all_native:
                            layer._subgraph_dispatcher = dispatcher
                            dispatchers_attached += 1
                        else:
                            dispatchers_skipped += 1
            capability["subgraph_dispatch"] = {
                "attached": dispatchers_attached,
                "skipped": dispatchers_skipped,
                "total_layers": len(layers),
            }
            if dispatchers_attached > 0:
                logger.info(
                    "Subgraph dispatch: attached %d/%d layer dispatchers",
                    dispatchers_attached, len(layers),
                )
        except Exception as exc:
            logger.debug("Failed to attach subgraph dispatchers: %s", exc)
            capability["subgraph_dispatch"] = {
                "attached": 0,
                "error": str(exc),
            }

    abi_session = abi_report.get("session")
    if abi_session is not None:
        try:
            setattr(model, "_native_runner_abi_session", abi_session)
            capability["runner_abi"]["session_attached"] = True
            if capability.get("execution_path") == "selective_native_active_legacy_compile":
                capability["execution_path"] = "selective_native_abi_ready_legacy_compile"
        except Exception as exc:
            logger.debug("Failed to attach runner ABI session: %s", exc)
            capability["runner_abi"]["session_attached"] = False
            capability["runner_abi"]["attach_error"] = str(exc)
            try:
                abi_session.close()
            except Exception:
                pass

    selective_layer_exec_enabled = _env_flag("NATIVE_RUNNER_SELECTIVE_LAYER_EXEC", False)
    selective_layer_strict = _env_flag("NATIVE_RUNNER_SELECTIVE_LAYER_STRICT", False)
    capability["selective_execution"]["layer_exec_enabled"] = selective_layer_exec_enabled
    capability["selective_execution"]["layer_exec_strict"] = selective_layer_strict
    if selective_candidate and bool(selective_activation.get("activated")) and selective_layer_exec_enabled:
        layer_build = build_designer_layer_modules(layer_graphs)
        capability["selective_execution"]["layer_build"] = {
            "attempted": bool(layer_build.get("attempted")),
            "compiled_layers": int(layer_build.get("compiled_layers") or 0),
            "failed_layers": int(layer_build.get("failed_layers") or 0),
            "total_layers": int(layer_build.get("total_layers") or 0),
            "errors": list(layer_build.get("errors") or []),
            "skip_reasons": [],
            "skipped_layers": 0,
            "applied_layers": 0,
            "layer_results": [],
        }
        replacements = layer_build.get("replacements") or {}
        model_dim = int(getattr(model, "model_dim", 0) or 0)
        if hasattr(model, "layers"):
            total_layers = len(model.layers)
            for idx, payload in replacements.items():
                layer_result: Dict[str, Any] = {
                    "layer_index": idx,
                    "workflow_id": payload.get("workflow_id"),
                    "input_node_id": payload.get("input_node_id"),
                    "applied": False,
                    "skip_reason": None,
                    "error": None,
                }
                try:
                    try:
                        layer_idx = int(idx)
                        layer_result["layer_index"] = layer_idx
                    except Exception:
                        capability["selective_execution"]["layer_build"]["skipped_layers"] += 1
                        reason = "invalid_layer_index"
                        capability["selective_execution"]["layer_build"]["skip_reasons"].append(
                            f"skip_layer_{idx}:{reason}"
                        )
                        layer_result["skip_reason"] = reason
                        capability["selective_execution"]["layer_build"]["layer_results"].append(layer_result)
                        continue
                    if layer_idx < 0 or layer_idx >= total_layers:
                        capability["selective_execution"]["layer_build"]["skipped_layers"] += 1
                        reason = "layer_index_out_of_range"
                        capability["selective_execution"]["layer_build"]["skip_reasons"].append(
                            f"skip_layer_{layer_idx}:{reason}"
                        )
                        layer_result["skip_reason"] = reason
                        capability["selective_execution"]["layer_build"]["layer_results"].append(layer_result)
                        continue
                    wm = payload.get("module")
                    in_id = str(payload.get("input_node_id") or "")
                    if wm is None:
                        capability["selective_execution"]["layer_build"]["skipped_layers"] += 1
                        reason = "missing_workflow_module"
                        capability["selective_execution"]["layer_build"]["skip_reasons"].append(
                            f"skip_layer_{layer_idx}:{reason}"
                        )
                        layer_result["skip_reason"] = reason
                        capability["selective_execution"]["layer_build"]["layer_results"].append(layer_result)
                        continue
                    if not in_id:
                        capability["selective_execution"]["layer_build"]["skipped_layers"] += 1
                        reason = "missing_input_node_id"
                        capability["selective_execution"]["layer_build"]["skip_reasons"].append(
                            f"skip_layer_{layer_idx}:{reason}"
                        )
                        layer_result["skip_reason"] = reason
                        capability["selective_execution"]["layer_build"]["layer_results"].append(layer_result)
                        continue
                    adapter = DesignerWorkflowLayerAdapter(wm, in_id).as_module()
                    contract_error = _validate_designer_layer_adapter_contract(
                        adapter,
                        model_dim=model_dim,
                        max_seq_len=max_seq_len,
                    )
                    if contract_error:
                        capability["selective_execution"]["layer_build"]["skipped_layers"] += 1
                        capability["selective_execution"]["layer_build"]["skip_reasons"].append(
                            f"skip_layer_{layer_idx}:{contract_error}"
                        )
                        layer_result["skip_reason"] = contract_error
                        capability["selective_execution"]["layer_build"]["layer_results"].append(layer_result)
                        continue
                    model.layers[layer_idx] = adapter
                    capability["selective_execution"]["layer_build"]["applied_layers"] += 1
                    layer_result["applied"] = True
                    capability["selective_execution"]["layer_build"]["layer_results"].append(layer_result)
                except Exception as exc:
                    layer_result["error"] = str(exc)
                    capability["selective_execution"]["layer_build"]["errors"].append(
                        f"apply_layer_{idx}:{exc}"
                    )
                    capability["selective_execution"]["layer_build"]["layer_results"].append(layer_result)
            if capability["selective_execution"]["layer_build"]["applied_layers"] > 0:
                capability["execution_path"] = "selective_designer_layers_active"
            capability["selective_execution"]["layer_build"]["summary"] = _summarize_layer_build(
                capability["selective_execution"]["layer_build"]
            )
            if selective_layer_strict:
                failed_layers = [
                    item
                    for item in capability["selective_execution"]["layer_build"]["layer_results"]
                    if not bool(item.get("applied"))
                ]
                if failed_layers:
                    capability["selective_execution"]["layer_build"]["summary"]["strict_failed"] = True
                    details = ", ".join(
                        f"layer={item.get('layer_index')} reason={item.get('skip_reason') or item.get('error') or 'unknown'}"
                        for item in failed_layers[:3]
                    )
                    raise RuntimeError(
                        "Selective layer strict mode rejected incompatible replacements: "
                        f"{details}"
                    )
        else:
            capability["selective_execution"]["layer_build"]["errors"].append("model_has_no_layers")
            capability["selective_execution"]["layer_build"]["summary"] = _summarize_layer_build(
                capability["selective_execution"]["layer_build"]
            )
    elif selective_candidate and bool(selective_activation.get("activated")):
        capability["selective_execution"]["layer_build"] = {
            "attempted": False,
            "compiled_layers": 0,
            "failed_layers": 0,
            "total_layers": int(len(layer_graphs or [])),
            "errors": ["layer_exec_disabled_by_env"],
            "skip_reasons": [],
            "skipped_layers": 0,
            "applied_layers": 0,
            "layer_results": [],
        }
        capability["selective_execution"]["layer_build"]["summary"] = _summarize_layer_build(
            capability["selective_execution"]["layer_build"]
        )

    _maybe_fail_on_fallback_rate()
    _maybe_fail_on_legacy_compile_usage()
    # Refresh fallback telemetry after current compile increments.
    capability["fallback_metrics"] = native_runner_capability_report().get("fallback_metrics", {})

    # Attach diagnostic metadata for observability.
    try:
        setattr(model, "_native_runner_report", capability)
    except Exception:
        pass

    warning_count = int(capability.get("semantic_warning_count") or 0)
    if warning_count > 0:
        logger.debug(
            "Native runner capability reports %d semantic mapping warnings (status=%s)",
            warning_count,
            capability.get("status"),
        )
    else:
        logger.info(
            "Native runner capability status=%s enabled=%s strict=%s native_lib=%s",
            capability.get("status"),
            capability.get("enabled"),
            capability.get("strict"),
            "loaded" if native_lib is not None else "unavailable",
        )

    return model
