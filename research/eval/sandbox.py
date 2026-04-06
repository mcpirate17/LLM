"""
Sandbox Execution

Safe evaluation of synthesized programs with:
- Timeout enforcement
- OOM catching
- CUDA fatal error detection (device-side assert, context corruption)
- NaN/Inf detection
- Gradient health checking
- Memory tracking
"""

from __future__ import annotations

import gc
import logging
import os
import random
import signal
import time
import traceback

logger = logging.getLogger(__name__)
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from ..scientist.perf import PerfTracer, OpKernelProfiler
from .sparsity import check_activation_sparsity
from .utils import compute_grad_norm, language_model_loss, make_adamw
from research.defaults import VOCAB_SIZE


def _env_bool(key: str, default: str = "0") -> bool:
    """Parse an environment variable as a boolean flag."""
    return os.getenv(key, default).strip().lower() in {"1", "true", "yes", "on"}


def _safe_eval_level() -> str:
    level = os.getenv("AI_SCI_SAFE_EVAL_LEVEL", "minimal").strip().lower()
    if level not in {"minimal", "full"}:
        return "minimal"
    return level


def _resolve_probe_flag(explicit: Optional[bool], env_key: str, default: bool) -> bool:
    if explicit is not None:
        return bool(explicit)
    return _env_bool(env_key, "1" if default else "0")


# Substrings in CUDA errors that indicate an unrecoverable (sticky) context
_CUDA_FATAL_MARKERS = (
    "device-side assert",
    "cudaErrorAssert",
    "CUDA error: an illegal memory access",
    "CUDA error: unspecified launch failure",
    "context is destroyed",
)
_SAFE_EVAL_CALL_COUNT = 0


def is_cuda_fatal(error: BaseException) -> bool:
    """Return True if the exception indicates a sticky/unrecoverable CUDA error."""
    msg = str(error).lower()
    return any(m.lower() in msg for m in _CUDA_FATAL_MARKERS)


from research.synthesis.result_schemas import SandboxResult


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


def _mapped_shared_token_ids(batch_size: int, seq_len: int, vocab_size: int):
    """Create token IDs using a zero-copy NumPy->Torch view without disk I/O.
    Uses pinned memory if possible for faster transfer to GPU.
    """
    arr = np.empty((batch_size, seq_len), dtype=np.int64)
    arr[:] = np.random.randint(
        0, vocab_size, size=(batch_size, seq_len), dtype=np.int64
    )
    tensor = torch.from_numpy(arr)
    # Z8: Pin memory for faster CPU->GPU transfer if we know we're going to GPU later
    if torch.cuda.is_available():
        try:
            tensor = tensor.pin_memory()
        except RuntimeError as exc:
            logger.debug("pin_memory unavailable for shared token buffer: %s", exc)
    return tensor, arr, None


def _install_timeout(timeout_seconds: int):
    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
    except (AttributeError, ValueError):
        pass
    return old_handler


def _reset_timeout(old_handler) -> None:
    try:
        signal.alarm(0)
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)
    except (AttributeError, ValueError):
        pass


def _resolve_safe_eval_runtime(
    device: str,
    run_training_dynamics_probe: Optional[bool],
    run_activation_sparsity_probe: Optional[bool],
):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    trace_enabled = _env_bool("AI_SCI_PERF_TRACE")
    tracer = PerfTracer() if trace_enabled else None
    kernel_profile_enabled = _env_bool("AI_SCI_KERNEL_PROFILE")
    op_profiler = OpKernelProfiler(enabled=kernel_profile_enabled, top_k=20)
    safe_eval_level = _safe_eval_level()
    run_training_dynamics = _resolve_probe_flag(
        run_training_dynamics_probe,
        "AI_SCI_SAFE_EVAL_TRAINING_DYNAMICS",
        safe_eval_level == "full",
    )
    run_activation_sparsity = _resolve_probe_flag(
        run_activation_sparsity_probe,
        "AI_SCI_SAFE_EVAL_ACTIVATION_SPARSITY",
        safe_eval_level == "full",
    )
    return (
        dev,
        tracer,
        op_profiler,
        safe_eval_level,
        run_training_dynamics,
        run_activation_sparsity,
    )


def _prepare_input_ids(
    dev: torch.device,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> torch.Tensor:
    if dev.type == "cuda":
        return torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)
    shared_ids, _mapped_array, _mapped_path = _mapped_shared_token_ids(
        batch_size, seq_len, vocab_size
    )
    return shared_ids


def _abi_probe_enabled(explicit: Optional[bool]) -> bool:
    if explicit is None:
        return _env_bool("NATIVE_RUNNER_ABI_INFER_PROBE", "1")
    return bool(explicit)


def _run_native_abi_probe(
    model: nn.Module,
    input_ids: torch.Tensor,
    batch_size: int,
    vocab_size: int,
    abi_infer_probe: Optional[bool],
):
    abi_session = getattr(model, "_native_runner_abi_session", None)
    if not (_abi_probe_enabled(abi_infer_probe) and abi_session is not None):
        return None, {
            "attempted": False,
            "succeeded": False,
            "reason": "disabled_or_missing_session",
            "primary_requested": False,
            "primary_used": False,
            "mode": "probe_only",
        }

    abi_probe_logits = None
    probe_payload = {
        "attempted": True,
        "succeeded": False,
        "reason": "unknown",
        "vocab_size": None,
        "max_logit": None,
        "primary_requested": False,
        "primary_used": False,
        "mode": "probe_only",
    }
    try:
        flat_tokens = input_ids.detach().reshape(-1)
        if hasattr(abi_session, "execute_tokens_tensor"):
            abi_logits = abi_session.execute_tokens_tensor(
                flat_tokens, batch=batch_size
            )
        else:
            abi_logits = abi_session.execute_tokens(
                flat_tokens.cpu().tolist(),
                batch=batch_size,
            )
        if int(len(abi_logits)) != int(vocab_size):
            probe_payload["reason"] = (
                f"vocab_mismatch:{len(abi_logits)}!={int(vocab_size)}"
            )
        else:
            probe_payload["succeeded"] = True
            probe_payload["reason"] = "ok"
            probe_payload["vocab_size"] = int(len(abi_logits))
            probe_payload["max_logit"] = float(max(abi_logits)) if abi_logits else None
            abi_probe_logits = abi_logits
    except Exception as exc:
        probe_payload["reason"] = f"execute_error:{exc}"
    return abi_probe_logits, probe_payload


def _resolve_native_primary_flags(
    abi_infer_primary: Optional[bool],
    abi_infer_primary_no_grad: Optional[bool],
) -> tuple[bool, bool]:
    if abi_infer_primary is None:
        native_primary_requested = _env_bool("NATIVE_RUNNER_ABI_INFER_PRIMARY", "0")
    else:
        native_primary_requested = bool(abi_infer_primary)
    if abi_infer_primary_no_grad is None:
        native_primary_no_grad = _env_bool(
            "NATIVE_RUNNER_ABI_INFER_PRIMARY_NO_GRAD", "1"
        )
    else:
        native_primary_no_grad = bool(abi_infer_primary_no_grad)
    return native_primary_requested, native_primary_no_grad


def _maybe_use_native_primary_logits(
    abi_probe_logits,
    native_primary_requested: bool,
    native_primary_no_grad: bool,
    dev: torch.device,
    batch_size: int,
    seq_len: int,
):
    if not (
        native_primary_requested
        and native_primary_no_grad
        and abi_probe_logits is not None
    ):
        return None, False
    logits = torch.tensor(abi_probe_logits, dtype=torch.float32, device=dev).view(
        1, 1, -1
    )
    return logits.expand(batch_size, seq_len, -1).contiguous(), True


def _run_forward_pass(
    model: nn.Module,
    input_ids: torch.Tensor,
    dev: torch.device,
    op_profiler: OpKernelProfiler,
):
    logits = None

    def _run_forward() -> None:
        nonlocal logits
        with torch.amp.autocast(
            device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")
        ):
            logits = model(input_ids)

    forward_kernel = op_profiler.profile_callable(_run_forward)
    if logits is None:
        _run_forward()
    return logits, forward_kernel


def _validate_logits_shape(
    logits: torch.Tensor,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> tuple[Optional[str], Optional[str]]:
    if logits.dim() != 3:
        return (
            f"Expected 3D logits (batch, seq, vocab), got shape {tuple(logits.shape)}",
            "shape_mismatch",
        )
    b_out, s_out, v_out = logits.shape
    if b_out != batch_size or s_out != seq_len:
        return (
            f"Logits shape mismatch: got ({b_out}, {s_out}, {v_out}), expected ({batch_size}, {seq_len}, *)",
            "shape_mismatch",
        )
    if v_out != vocab_size:
        return (
            f"Logits vocab dim mismatch: got {v_out}, expected {vocab_size}",
            "shape_mismatch",
        )
    return None, None


def _configure_parity_settings():
    try:
        parity_sample_rate = max(
            0.0,
            min(1.0, float(os.getenv("NATIVE_RUNNER_ABI_PARITY_SAMPLE_RATE", "0.0"))),
        )
    except ValueError:
        logger.debug("Invalid NATIVE_RUNNER_ABI_PARITY_SAMPLE_RATE; defaulting to 0.0")
        parity_sample_rate = 0.0
    try:
        parity_threshold = float(os.getenv("NATIVE_RUNNER_ABI_PARITY_MAX_ABS", "1.0"))
    except ValueError:
        logger.debug("Invalid NATIVE_RUNNER_ABI_PARITY_MAX_ABS; defaulting to 1.0")
        parity_threshold = 1.0
    parity_strict = _env_bool("NATIVE_RUNNER_ABI_PARITY_STRICT", "0")
    return parity_sample_rate, parity_threshold, parity_strict


def _run_native_parity_check(
    model: nn.Module,
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    dev: torch.device,
    result: SandboxResult,
) -> tuple[bool, Optional[str], Optional[float], Optional[float], float, bool]:
    parity_sample_rate, parity_threshold, parity_strict = _configure_parity_settings()
    parity_attempt = parity_sample_rate > 0.0 and random.random() < parity_sample_rate
    parity_max_abs = None
    parity_mean_abs = None
    parity_pass = None
    parity_reason = "not_sampled"
    if parity_attempt:
        parity_reason = "ok"
        try:
            with (
                torch.no_grad(),
                torch.amp.autocast(
                    device_type=dev.type,
                    dtype=torch.bfloat16,
                    enabled=(dev.type == "cuda"),
                ),
            ):
                shadow_logits = model(input_ids)
            if shadow_logits.dim() != 3 or tuple(shadow_logits.shape) != tuple(
                logits.shape
            ):
                parity_pass = False
                parity_reason = f"shape_mismatch:{tuple(shadow_logits.shape)}!={tuple(logits.shape)}"
            else:
                diff = torch.abs(shadow_logits.float() - logits.float())
                parity_max_abs = float(diff.max().item())
                parity_mean_abs = float(diff.mean().item())
                parity_pass = parity_max_abs <= parity_threshold
                if not parity_pass:
                    parity_reason = "max_abs_exceeded"
        except Exception as exc:
            parity_pass = False
            parity_reason = f"shadow_forward_error:{exc}"
    if isinstance(result.native_abi_probe, dict):
        result.native_abi_probe["parity_sample_rate"] = float(parity_sample_rate)
        result.native_abi_probe["parity_attempted"] = bool(parity_attempt)
        result.native_abi_probe["parity_pass"] = parity_pass
        result.native_abi_probe["parity_reason"] = parity_reason
        result.native_abi_probe["parity_max_abs_diff"] = parity_max_abs
        result.native_abi_probe["parity_mean_abs_diff"] = parity_mean_abs
        result.native_abi_probe["parity_max_abs_threshold"] = float(parity_threshold)
        result.native_abi_probe["parity_strict"] = bool(parity_strict)
    return (
        parity_attempt,
        parity_reason,
        parity_max_abs,
        parity_mean_abs,
        parity_threshold,
        parity_strict,
    )


def _run_backward_stage(
    model: nn.Module,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    vocab_size: int,
    dev: torch.device,
    op_profiler: OpKernelProfiler,
):
    logits_for_loss = logits
    logit_std = logits.detach().std()
    if logit_std > 10.0:
        logits_for_loss = logits / (logit_std / 2.0)
    loss = language_model_loss(logits_for_loss, input_ids, vocab_size)

    def _run_backward() -> None:
        loss.backward()

    backward_kernel = op_profiler.profile_callable(_run_backward)
    if backward_kernel is None:
        _run_backward()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return loss, backward_kernel


def _gradient_health(model: nn.Module):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    has_nan = False
    has_zero = True
    n_with_grad = len(grads)
    total_norm = 0.0
    if grads:
        total_norm = compute_grad_norm(model)
        try:
            norms = torch._foreach_norm(grads, 2)
            norm_vec = torch.stack([n.detach() for n in norms])
            has_nan = not bool(torch.isfinite(norm_vec).all().item())
            has_zero = not bool((norm_vec > 1e-10).any().item())
        except RuntimeError as exc:
            logger.debug(
                "torch._foreach_norm failed during sandbox grad check; using scalar fallback: %s",
                exc,
            )
            for grad in grads:
                if torch.isnan(grad).any():
                    has_nan = True
                pnorm = grad.data.float().norm().item()
                if pnorm > 1e-10:
                    has_zero = False
    return float(total_norm), has_nan, has_zero, n_with_grad


def _collect_routing_telemetry(model: nn.Module, capture_heatmaps: bool):
    def _accumulate_hist(acc, hist):
        if acc is None:
            return hist
        if acc.numel() == hist.numel():
            return acc + hist
        if acc.numel() < hist.numel():
            padded = torch.zeros_like(hist)
            padded[: acc.numel()] = acc
            return padded + hist
        padded = torch.zeros_like(acc)
        padded[: hist.numel()] = hist
        return acc + padded

    heatmaps = {}
    total_savings = 0.0
    total_depth_ratio = 0.0
    routing_op_count = 0
    tokens_total = 0
    keep_count = 0
    drop_count = 0
    default_path_count = 0
    routed_token_count = 0
    sparse_span_count = 0
    sparse_span_width_sum = 0.0
    sparse_span_width_count = 0
    sparse_span_coverage_tokens = 0
    lane_histogram = None
    confidence_histogram = None
    confidence_sum = 0.0
    confidence_sq_sum = 0.0
    confidence_count = 0
    route_strength_sum = 0.0
    route_strength_count = 0
    branch_weight_sum = None
    branch_weight_count = 0
    branch_dominance_sum = 0.0
    routed_branch_share_sum = 0.0
    medium_branch_share_sum = 0.0
    hard_branch_share_sum = 0.0
    routing_modes = set()
    gate_types = set()
    span_types = set()
    lane_count_max = 0
    trace_payloads = {}
    for name, module in model.named_modules():
        rt = getattr(module, "routing_telemetry", None)
        if not rt:
            continue
        if rt.get("heatmap") is not None:
            heatmaps[name] = rt["heatmap"]
        routing_op_count += 1
        tokens_total += int(rt.get("tokens_total", 0) or 0)
        keep_count += int(rt.get("keep_count", 0) or 0)
        drop_count += int(rt.get("drop_count", 0) or 0)
        default_path_count += int(rt.get("default_path_count", 0) or 0)
        routed_token_count += int(rt.get("routed_token_count", 0) or 0)
        sparse_span_count += int(rt.get("sparse_span_count", 0) or 0)
        sparse_span_width_sum += float(rt.get("sparse_span_width_sum", 0.0) or 0.0)
        sparse_span_width_count += int(rt.get("sparse_span_width_count", 0) or 0)
        sparse_span_coverage_tokens += int(
            rt.get("sparse_span_coverage_tokens", 0) or 0
        )
        confidence_sum += float(rt.get("confidence_sum", 0.0) or 0.0)
        confidence_sq_sum += float(rt.get("confidence_sq_sum", 0.0) or 0.0)
        confidence_count += int(rt.get("confidence_count", 0) or 0)
        route_strength_sum += float(rt.get("route_strength_sum", 0.0) or 0.0)
        route_strength_count += int(rt.get("route_strength_count", 0) or 0)
        branch_dominance_sum += float(rt.get("branch_dominance_sum", 0.0) or 0.0)
        routed_branch_share_sum += float(rt.get("routed_branch_share_sum", 0.0) or 0.0)
        medium_branch_share_sum += float(rt.get("medium_branch_share_sum", 0.0) or 0.0)
        hard_branch_share_sum += float(rt.get("hard_branch_share_sum", 0.0) or 0.0)
        branch_weight_count += int(rt.get("branch_weight_count", 0) or 0)
        if rt.get("routing_mode"):
            routing_modes.add(str(rt["routing_mode"]))
        if rt.get("gate_type"):
            gate_types.add(str(rt["gate_type"]))
        if rt.get("span_type"):
            span_types.add(str(rt["span_type"]))
        lane_count_max = max(lane_count_max, int(rt.get("lane_count", 0) or 0))
        total_savings += rt.get("savings_ratio", 0.0)
        total_depth_ratio += rt.get("depth_ratio", 1.0)
        if isinstance(rt.get("lane_histogram"), torch.Tensor):
            hist = rt["lane_histogram"].detach().to(torch.float32).cpu()
            lane_histogram = _accumulate_hist(lane_histogram, hist)
        if isinstance(rt.get("confidence_histogram"), torch.Tensor):
            hist = rt["confidence_histogram"].detach().to(torch.float32).cpu()
            confidence_histogram = _accumulate_hist(confidence_histogram, hist)
        if isinstance(rt.get("branch_weight_sum"), torch.Tensor):
            hist = rt["branch_weight_sum"].detach().to(torch.float32).cpu()
            branch_weight_sum = _accumulate_hist(branch_weight_sum, hist)
        if rt.get("trace_payload") is not None:
            trace_payloads[name] = rt["trace_payload"]
    payload = None
    if routing_op_count > 0:
        payload = {
            "routing_savings_ratio": round(total_savings / routing_op_count, 4),
            "routing_depth_ratio": round(total_depth_ratio / routing_op_count, 4),
        }
        if tokens_total > 0:
            payload["routing_keep_drop_ratio"] = {
                "keep": round(keep_count / tokens_total, 4),
                "drop": round(drop_count / tokens_total, 4),
            }
            payload["default_path_fraction"] = round(
                default_path_count / tokens_total, 4
            )
            payload["routed_compute_fraction"] = round(
                routed_token_count / tokens_total, 4
            )
        if sparse_span_width_count > 0:
            payload["sparse_span_count"] = int(sparse_span_count)
            payload["average_span_width"] = round(
                sparse_span_width_sum / sparse_span_width_count, 4
            )
        if tokens_total > 0:
            payload["sparse_span_coverage"] = round(
                sparse_span_coverage_tokens / tokens_total, 4
            )
        if lane_histogram is not None:
            lane_probs = lane_histogram / lane_histogram.sum().clamp(min=1.0)
            lane_entropy = float(
                -(lane_probs * torch.log(lane_probs.clamp(min=1e-10))).sum().item()
            )
            payload["lane_utilization_histogram"] = lane_histogram.int().tolist()
            payload["lane_entropy"] = round(lane_entropy, 4)
            payload["lane_utilization"] = payload["lane_utilization_histogram"]
            payload["active_lane_count"] = int((lane_histogram > 0).sum().item())
            payload["dead_lane_count"] = int((lane_histogram == 0).sum().item())
        if confidence_count > 0:
            conf_mean = confidence_sum / confidence_count
            conf_var = max(
                0.0, (confidence_sq_sum / confidence_count) - (conf_mean * conf_mean)
            )
            payload["route_confidence_mean"] = round(conf_mean, 4)
            payload["route_confidence_std"] = round(conf_var**0.5, 4)
        if route_strength_count > 0:
            payload["route_strength_mean"] = round(
                route_strength_sum / route_strength_count, 4
            )
        if branch_weight_sum is not None and branch_weight_count > 0:
            branch_means = (branch_weight_sum / max(branch_weight_count, 1)).tolist()
            payload["branch_weight_mean"] = [round(float(v), 4) for v in branch_means]
            payload["branch_dominance_mean"] = round(
                branch_dominance_sum / branch_weight_count, 4
            )
            payload["routed_branch_share"] = round(
                routed_branch_share_sum / branch_weight_count, 4
            )
            payload["medium_branch_share"] = round(
                medium_branch_share_sum / branch_weight_count, 4
            )
            payload["hard_branch_share"] = round(
                hard_branch_share_sum / branch_weight_count, 4
            )
        if confidence_histogram is not None:
            payload["confidence_histogram"] = confidence_histogram.int().tolist()
        if routing_modes:
            payload["routing_modes"] = sorted(routing_modes)
        if gate_types:
            payload["gate_types"] = sorted(gate_types)
        if span_types:
            payload["span_types"] = sorted(span_types)
        if lane_count_max > 0:
            payload["lane_count"] = lane_count_max
        if trace_payloads:
            payload["routing_traces"] = trace_payloads
        if capture_heatmaps and heatmaps:
            payload["routing_heatmaps"] = heatmaps
    return payload


def _set_result_error(
    result: SandboxResult, error: str, error_type: str
) -> SandboxResult:
    result.passed = False
    result.error = error
    result.error_type = error_type
    return result


def _extract_failure_op(tb: List[str], error_text: str) -> Optional[str]:
    failure_op = None
    for line in reversed(tb):
        if "_op_" in line and "in _op_" in line:
            import re as _re

            match = _re.search(r"in (_op_\w+)", line)
            if match:
                failure_op = match.group(1).removeprefix("_op_")
                break
        if "CompiledOp" in line and "forward" in line:
            import re as _re

            match = _re.search(r"CompiledOp\[(\w+)\]", line)
            if match:
                failure_op = match.group(1)
                break
    if failure_op is None:
        if "kv_compress" in error_text:
            failure_op = "latent_attention_compressor"
        elif "conv_weight" in error_text:
            failure_op = "conv1d_seq"
    return failure_op


def _sandbox_cleanup(dev: torch.device) -> None:
    empty_cache_every = int(os.getenv("AI_SCI_EMPTY_CACHE_EVERY", "0") or 0)
    force_gc_every = int(os.getenv("AI_SCI_FORCE_GC_EVERY", "0") or 0)
    global _SAFE_EVAL_CALL_COUNT
    _SAFE_EVAL_CALL_COUNT += 1
    if (
        dev.type == "cuda"
        and empty_cache_every > 0
        and (_SAFE_EVAL_CALL_COUNT % empty_cache_every == 0)
    ):
        try:
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            logger.debug(
                "torch.cuda.empty_cache() failed during sandbox cleanup: %s", exc
            )
    if force_gc_every > 0 and (_SAFE_EVAL_CALL_COUNT % force_gc_every == 0):
        gc.collect()


def _run_compile_stage(
    model: nn.Module,
    dev: torch.device,
    result: SandboxResult,
    tracer: Optional[PerfTracer],
) -> nn.Module:
    result.stage = "compile"
    if tracer is not None:
        tracer.start("compile", use_gpu=False)
    t0 = time.perf_counter()
    model = model.to(dev)
    result.param_count = sum(p.numel() for p in model.parameters())
    result.compile_time_ms = (time.perf_counter() - t0) * 1000
    if tracer is not None:
        tracer.stop("compile")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    return model


def _run_forward_stage(
    model: nn.Module,
    dev: torch.device,
    result: SandboxResult,
    tracer: Optional[PerfTracer],
    op_profiler: OpKernelProfiler,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    abi_infer_probe: Optional[bool],
    abi_infer_primary: Optional[bool],
    abi_infer_primary_no_grad: Optional[bool],
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    result.stage = "forward"
    if tracer is not None:
        tracer.start("forward", use_gpu=True)
    t1 = time.perf_counter()
    input_ids = _prepare_input_ids(dev, batch_size, seq_len, vocab_size)
    abi_probe_logits, result.native_abi_probe = _run_native_abi_probe(
        model, input_ids, batch_size, vocab_size, abi_infer_probe
    )
    native_primary_requested, native_primary_no_grad = _resolve_native_primary_flags(
        abi_infer_primary,
        abi_infer_primary_no_grad,
    )
    if isinstance(result.native_abi_probe, dict):
        result.native_abi_probe["primary_requested"] = bool(native_primary_requested)
    logits, native_primary_used = _maybe_use_native_primary_logits(
        abi_probe_logits,
        native_primary_requested,
        native_primary_no_grad,
        dev,
        batch_size,
        seq_len,
    )
    if native_primary_used and isinstance(result.native_abi_probe, dict):
        result.native_abi_probe["primary_used"] = True
        result.native_abi_probe["mode"] = "primary_forward_only"
    forward_kernel = None
    if logits is None:
        logits, forward_kernel = _run_forward_pass(model, input_ids, dev, op_profiler)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    result.forward_time_ms = (time.perf_counter() - t1) * 1000
    if tracer is not None:
        tracer.stop("forward")
    result.output_shape = str(tuple(logits.shape))
    shape_error, shape_error_type = _validate_logits_shape(
        logits, batch_size, seq_len, vocab_size
    )
    if shape_error is not None:
        raise RuntimeError(f"{shape_error_type}:{shape_error}")
    result.has_nan_output = bool(torch.isnan(logits).any())
    result.has_inf_output = bool(torch.isinf(logits).any())
    if result.has_nan_output or result.has_inf_output:
        raise FloatingPointError("NaN/Inf in forward output")
    if forward_kernel:
        result.kernel_timing = {"forward": forward_kernel}
    return input_ids, logits, native_primary_used


def _finalize_native_primary(
    model: nn.Module,
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    dev: torch.device,
    result: SandboxResult,
) -> bool:
    (
        parity_attempt,
        parity_reason,
        parity_max_abs,
        _parity_mean_abs,
        parity_threshold,
        parity_strict,
    ) = _run_native_parity_check(model, input_ids, logits, dev, result)
    if (
        parity_attempt
        and result.native_abi_probe.get("parity_pass") is False
        and parity_strict
    ):
        result.error = (
            "ABI parity regression in primary mode: "
            f"reason={parity_reason}, max_abs={parity_max_abs}, threshold={parity_threshold}"
        )
        result.error_type = "abi_parity_regression"
        return False
    if dev.type == "cuda":
        result.peak_memory_mb = torch.cuda.max_memory_allocated(dev) / (1024**2)
    result.passed = True
    return True


def _run_backward_and_health_stage(
    model: nn.Module,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    vocab_size: int,
    dev: torch.device,
    result: SandboxResult,
    tracer: Optional[PerfTracer],
    op_profiler: OpKernelProfiler,
) -> None:
    result.stage = "backward"
    if tracer is not None:
        tracer.start("backward", use_gpu=True)
    t2 = time.perf_counter()
    _loss, backward_kernel = _run_backward_stage(
        model,
        logits,
        input_ids,
        vocab_size,
        dev,
        op_profiler,
    )
    result.backward_time_ms = (time.perf_counter() - t2) * 1000
    if tracer is not None:
        tracer.stop("backward")
    total_norm, has_nan, has_zero, n_with_grad = _gradient_health(model)
    result.grad_norm = total_norm
    result.has_nan_grad = has_nan
    result.has_zero_grad = has_zero and n_with_grad > 0
    if has_nan:
        raise FloatingPointError("NaN in gradients")
    if has_zero:
        raise RuntimeError("All gradients are zero")
    if backward_kernel:
        kernel_timing = getattr(result, "kernel_timing", {}) or {}
        kernel_timing["backward"] = backward_kernel
        result.kernel_timing = kernel_timing


def _apply_stability_results(
    result: SandboxResult,
    stability: Dict,
) -> Optional[SandboxResult]:
    result.stability_score = stability["score"]
    result.extreme_input_passed = stability["extreme_passed"]
    result.random_input_passed = stability["random_passed"]
    result.causality_passed = stability["causality_passed"]
    result.output_range = stability.get("output_range")
    if not result.causality_passed:
        result.passed = False
        result.error = (
            "Strict Causality Gate Failed: Model looks ahead at future tokens."
        )
        result.error_type = "causality_violation"
        return result
    if stability.get("training_dynamics_passed") is False:
        result.passed = False
        _cv = stability.get("training_dynamics_cv", 0)
        _trend = stability.get("training_dynamics_trend", 0)
        result.error = f"Training dynamics unstable: CV={_cv:.3f}, trend={_trend:.3f} (10-step probe)"
        result.error_type = "unstable_dynamics"
        return result
    return None


def _run_activation_sparsity_stage(
    model: nn.Module,
    input_ids: torch.Tensor,
    result: SandboxResult,
) -> Optional[SandboxResult]:
    sparsity_report = check_activation_sparsity(model, [input_ids])
    result.activation_sparsity = sparsity_report.overall_sparsity
    result.dead_neuron_count = sparsity_report.total_dead_neurons
    result.sparsity_report = {
        "dead_neuron_ratio": sparsity_report.dead_neuron_ratio,
        "max_layer_collapse": sparsity_report.max_layer_collapse,
        "n_collapsed_layers": sum(1 for r in sparsity_report.layers if r.is_collapsed),
    }
    if any(r.is_collapsed for r in sparsity_report.layers):
        result.passed = False
        result.error = f"Activation collapse: {result.sparsity_report['n_collapsed_layers']} layers collapsed"
        result.error_type = "activation_collapse"
        return result
    return None


def _run_stability_stage(
    model: nn.Module,
    dev: torch.device,
    input_ids: torch.Tensor,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    result: SandboxResult,
    tracer: Optional[PerfTracer],
    safe_eval_level: str,
    run_training_dynamics_probe: bool,
    run_activation_sparsity_probe: bool,
) -> Optional[SandboxResult]:
    result.stage = "stability"
    if tracer is not None:
        tracer.start("stability", use_gpu=True)
    capture_heatmaps = safe_eval_level == "full"
    if capture_heatmaps and hasattr(model, "set_capture_heatmap"):
        model.set_capture_heatmap(True)
    try:
        stability = _stability_probe(
            model,
            dev,
            batch_size,
            seq_len,
            vocab_size,
            run_training_dynamics_probe=run_training_dynamics_probe,
        )
        result.routing_report = _collect_routing_telemetry(model, capture_heatmaps)
        result.sparsity_report = result.routing_report
        failed = _apply_stability_results(result, stability)
        if failed is not None:
            return failed
        if run_activation_sparsity_probe:
            failed = _run_activation_sparsity_stage(model, input_ids, result)
            if failed is not None:
                return failed
    finally:
        if capture_heatmaps and hasattr(model, "set_capture_heatmap"):
            model.set_capture_heatmap(False)
        if tracer is not None:
            tracer.stop("stability")
    return None


def _handle_sandbox_exception(
    result: SandboxResult,
    exc: BaseException,
    timeout_seconds: int,
    dev: torch.device,
) -> None:
    if isinstance(exc, TimeoutError):
        result.error = f"Timeout after {timeout_seconds}s in stage {result.stage}"
        result.error_type = "timeout"
        return
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        result.error = "CUDA out of memory"
        result.error_type = "oom"
        return
    if isinstance(exc, FloatingPointError):
        message = str(exc)
        result.error = message
        result.error_type = "nan_forward" if "forward output" in message else "nan_grad"
        return
    if isinstance(exc, RuntimeError) and str(exc).startswith("shape_mismatch:"):
        result.error = str(exc).split(":", 1)[1]
        result.error_type = "shape_mismatch"
        return
    if isinstance(exc, RuntimeError) and str(exc) == "All gradients are zero":
        result.error = str(exc)
        result.error_type = "zero_grad"
        return
    if is_cuda_fatal(exc):
        result.error = f"Fatal CUDA error in stage {result.stage}: {exc}"
        result.error_type = "cuda_fatal"
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                _probe = torch.zeros(1, device="cuda")
                del _probe
                torch.cuda.synchronize()
            except Exception as recovery_exc:
                logger.warning(
                    "CUDA context unrecoverable after fatal error: %s",
                    recovery_exc,
                )
        return
    tb = traceback.format_exc().strip().split("\n")
    result.error = "\n".join(tb[-3:])
    result.error_type = type(exc).__name__
    failure_op = _extract_failure_op(tb, result.error)
    if failure_op:
        result.failure_op = failure_op


def safe_eval(
    model: nn.Module,
    batch_size: int = 2,
    seq_len: int = 128,
    vocab_size: int = VOCAB_SIZE,
    device: str = "cuda",
    timeout_seconds: int = 30,
    run_stability_probe: bool = True,
    run_training_dynamics_probe: Optional[bool] = None,
    run_activation_sparsity_probe: Optional[bool] = None,
    abi_infer_probe: Optional[bool] = None,
    abi_infer_primary: Optional[bool] = None,
    abi_infer_primary_no_grad: Optional[bool] = None,
) -> SandboxResult:
    """Safely evaluate a model through Stage 0 and Stage 0.5.

    Stage 0: Compilation + forward + backward
    Stage 0.5: Numerical stability probe
    """
    result = SandboxResult()
    (
        dev,
        tracer,
        op_profiler,
        safe_eval_level,
        run_training_dynamics_probe,
        run_activation_sparsity_probe,
    ) = _resolve_safe_eval_runtime(
        device,
        run_training_dynamics_probe,
        run_activation_sparsity_probe,
    )
    old_handler = _install_timeout(timeout_seconds)

    try:
        model = _run_compile_stage(model, dev, result, tracer)
        input_ids, logits, native_primary_used = _run_forward_stage(
            model,
            dev,
            result,
            tracer,
            op_profiler,
            batch_size,
            seq_len,
            vocab_size,
            abi_infer_probe,
            abi_infer_primary,
            abi_infer_primary_no_grad,
        )
        if native_primary_used:
            if _finalize_native_primary(model, input_ids, logits, dev, result):
                return result
            return result
        _run_backward_and_health_stage(
            model,
            logits,
            input_ids,
            vocab_size,
            dev,
            result,
            tracer,
            op_profiler,
        )
        if run_stability_probe:
            failed = _run_stability_stage(
                model,
                dev,
                input_ids,
                batch_size,
                seq_len,
                vocab_size,
                result,
                tracer,
                safe_eval_level,
                run_training_dynamics_probe,
                run_activation_sparsity_probe,
            )
            if failed is not None:
                return failed
        if dev.type == "cuda":
            result.peak_memory_mb = torch.cuda.max_memory_allocated(dev) / (1024**2)
        result.passed = True
    except Exception as exc:
        _handle_sandbox_exception(result, exc, timeout_seconds, dev)
    finally:
        _reset_timeout(old_handler)
        _sandbox_cleanup(dev)

    return result


def _check_stability_ids(
    model: nn.Module,
    ids: torch.Tensor,
    dev: torch.device,
) -> Optional[torch.Tensor]:
    try:
        with (
            torch.no_grad(),
            torch.amp.autocast(
                device_type=dev.type,
                dtype=torch.bfloat16,
                enabled=(dev.type == "cuda"),
            ),
        ):
            out = model(ids)
        if not (torch.isnan(out).any() or torch.isinf(out).any()):
            return out
    except Exception as exc:
        logger.debug("Stability probe forward failed: %s", exc, exc_info=True)
    return None


def _run_random_input_stability(
    model: nn.Module,
    dev: torch.device,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    results: Dict,
) -> bool:
    outputs = []
    for _ in range(3):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)
        out = _check_stability_ids(model, ids, dev)
        if out is not None:
            outputs.append(out)
    if len(outputs) != 3:
        return False
    results["random_passed"] = True
    all_out = torch.cat([o.flatten() for o in outputs])
    results["output_range"] = (
        f"[{all_out.min().item():.2f}, {all_out.max().item():.2f}]"
    )
    return True


def _run_extreme_input_stability(
    model: nn.Module,
    dev: torch.device,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    results: Dict,
) -> int:
    checks_passed = 0
    ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=dev)
    if _check_stability_ids(model, ids, dev) is not None:
        checks_passed += 1
        results["extreme_passed"] = True
    ids = (
        torch.arange(seq_len, device=dev).unsqueeze(0).expand(batch_size, -1)
        % vocab_size
    )
    if _check_stability_ids(model, ids, dev) is not None:
        checks_passed += 1
    ids = torch.full(
        (batch_size, seq_len), vocab_size - 1, dtype=torch.long, device=dev
    )
    if _check_stability_ids(model, ids, dev) is not None:
        checks_passed += 1
    return checks_passed


def _run_causality_gate(
    model: nn.Module,
    dev: torch.device,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> bool:
    try:
        with (
            torch.no_grad(),
            torch.amp.autocast(
                device_type=dev.type,
                dtype=torch.bfloat16,
                enabled=(dev.type == "cuda"),
            ),
        ):
            ids_base = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)
            out_base = model(ids_base)
            ids_mod = ids_base.clone()
            midpoint = seq_len // 2
            ids_mod[:, midpoint:] = torch.randint(
                0,
                vocab_size,
                (batch_size, seq_len - midpoint),
                device=dev,
            )
            out_mod = model(ids_mod)
        diff = (
            torch.abs(
                out_base[:, :midpoint, :].float() - out_mod[:, :midpoint, :].float()
            )
            .max()
            .item()
        )
        return diff < 0.05
    except Exception:
        return False


def _run_training_dynamics_check(
    model: nn.Module,
    dev: torch.device,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    results: Dict,
) -> bool:
    try:
        model.train()
        probe_optimizer = make_adamw(
            model.parameters(), lr=1e-3, fused_if_available=False
        )
        probe_losses: List[float] = []
        use_amp = dev.type == "cuda"
        for _ in range(20):
            ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)
            probe_optimizer.zero_grad()
            with torch.amp.autocast(
                device_type=dev.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                logits = model(ids)
                loss = language_model_loss(logits, ids, logits.size(-1))
            if torch.isnan(loss) or torch.isinf(loss):
                return False
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            probe_optimizer.step()
            probe_losses.append(loss.item())
        mean_loss = sum(probe_losses) / len(probe_losses)
        if mean_loss <= 0:
            results["training_dynamics_passed"] = True
            return True
        var_loss = sum((x - mean_loss) ** 2 for x in probe_losses) / len(probe_losses)
        cv = (var_loss**0.5) / mean_loss
        sign_changes = sum(
            1
            for i in range(2, len(probe_losses))
            if (probe_losses[i] - probe_losses[i - 1])
            * (probe_losses[i - 1] - probe_losses[i - 2])
            < 0
        )
        reversal_rate = sign_changes / max(len(probe_losses) - 2, 1)
        first5 = sum(probe_losses[:5]) / 5
        last5 = sum(probe_losses[-5:]) / 5
        dynamics_bad = cv > 0.25 or (last5 > first5 * 1.05 and cv > 0.10)
        results["training_dynamics_passed"] = not dynamics_bad
        if dynamics_bad:
            results["training_dynamics_cv"] = round(cv, 4)
            results["training_dynamics_trend"] = round(last5 / max(first5, 1e-8), 4)
            results["training_dynamics_reversal_rate"] = round(reversal_rate, 4)
        return not dynamics_bad
    except Exception:
        results["training_dynamics_passed"] = False
        return False


def _stability_probe(
    model: nn.Module,
    dev: torch.device,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    *,
    run_training_dynamics_probe: bool,
) -> Dict:
    """Run numerical stability probes."""
    model.eval()
    results = {
        "score": 0.0,
        "extreme_passed": False,
        "random_passed": False,
        "causality_passed": True,
    }
    checks_passed = 0
    total_checks = 0
    total_checks += 1
    if _run_random_input_stability(
        model, dev, batch_size, seq_len, vocab_size, results
    ):
        checks_passed += 1
    total_checks += 3
    checks_passed += _run_extreme_input_stability(
        model,
        dev,
        batch_size,
        seq_len,
        vocab_size,
        results,
    )
    total_checks += 1
    results["causality_passed"] = _run_causality_gate(
        model,
        dev,
        batch_size,
        seq_len,
        vocab_size,
    )
    if results["causality_passed"]:
        checks_passed += 1
    if run_training_dynamics_probe:
        total_checks += 1
        if _run_training_dynamics_check(
            model, dev, batch_size, seq_len, vocab_size, results
        ):
            checks_passed += 1
    else:
        results["training_dynamics_passed"] = None

    results["score"] = checks_passed / max(total_checks, 1)
    model.train()
    return results
