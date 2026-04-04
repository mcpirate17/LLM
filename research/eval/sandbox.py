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
import torch.nn.functional as F
from ..scientist.perf import PerfTracer, OpKernelProfiler
from .sparsity import check_activation_sparsity
from .utils import compute_grad_norm
from research.defaults import VOCAB_SIZE


def _env_bool(key: str, default: str = "0") -> bool:
    """Parse an environment variable as a boolean flag."""
    return os.getenv(key, default).strip().lower() in {"1", "true", "yes", "on"}


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


def safe_eval(
    model: nn.Module,
    batch_size: int = 2,
    seq_len: int = 128,
    vocab_size: int = VOCAB_SIZE,
    device: str = "cuda",
    timeout_seconds: int = 30,
    run_stability_probe: bool = True,
    abi_infer_probe: Optional[bool] = None,
    abi_infer_primary: Optional[bool] = None,
    abi_infer_primary_no_grad: Optional[bool] = None,
) -> SandboxResult:
    """Safely evaluate a model through Stage 0 and Stage 0.5.

    Stage 0: Compilation + forward + backward
    Stage 0.5: Numerical stability probe
    """
    result = SandboxResult()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    trace_enabled = _env_bool("AI_SCI_PERF_TRACE")
    tracer = PerfTracer() if trace_enabled else None
    kernel_profile_enabled = _env_bool("AI_SCI_KERNEL_PROFILE")
    op_profiler = OpKernelProfiler(enabled=kernel_profile_enabled, top_k=20)
    mapped_array = None

    # Set timeout (Unix only)
    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
    except (AttributeError, ValueError):
        pass  # Windows or not main thread

    try:
        # ── Stage 0: Basic functionality ──
        result.stage = "compile"
        if tracer is not None:
            tracer.start("compile", use_gpu=False)
        t0 = time.perf_counter()

        model = model.to(dev)
        result.param_count = sum(p.numel() for p in model.parameters())

        t1 = time.perf_counter()
        result.compile_time_ms = (t1 - t0) * 1000
        if tracer is not None:
            tracer.stop("compile")

        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)

        # Forward pass
        result.stage = "forward"
        if tracer is not None:
            tracer.start("forward", use_gpu=True)
        if dev.type == "cuda":
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)
        else:
            shared_ids, mapped_array, _mapped_path = _mapped_shared_token_ids(
                batch_size, seq_len, vocab_size
            )
            input_ids = shared_ids

        # Optional inference-only probe through native runner ABI session.
        # This validates that compile-time ABI handles can execute real token payloads
        # without replacing training/backprop path yet.
        if abi_infer_probe is None:
            abi_probe_enabled = _env_bool("NATIVE_RUNNER_ABI_INFER_PROBE", "1")
        else:
            abi_probe_enabled = bool(abi_infer_probe)
        abi_session = getattr(model, "_native_runner_abi_session", None)
        if abi_probe_enabled and abi_session is not None:
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
                flat_tokens = input_ids.detach().cpu().reshape(-1).tolist()
                abi_logits = abi_session.execute_tokens(flat_tokens, batch=batch_size)
                if int(len(abi_logits)) != int(vocab_size):
                    probe_payload["reason"] = (
                        f"vocab_mismatch:{len(abi_logits)}!={int(vocab_size)}"
                    )
                else:
                    probe_payload["succeeded"] = True
                    probe_payload["reason"] = "ok"
                    probe_payload["vocab_size"] = int(len(abi_logits))
                    probe_payload["max_logit"] = (
                        float(max(abi_logits)) if abi_logits else None
                    )
                    abi_probe_logits = abi_logits
            except Exception as exc:
                probe_payload["reason"] = f"execute_error:{exc}"
            result.native_abi_probe = probe_payload
        else:
            abi_probe_logits = None
            result.native_abi_probe = {
                "attempted": False,
                "succeeded": False,
                "reason": "disabled_or_missing_session",
                "primary_requested": False,
                "primary_used": False,
                "mode": "probe_only",
            }
        logits = None
        if abi_infer_primary is None:
            native_primary_requested = os.getenv(
                "NATIVE_RUNNER_ABI_INFER_PRIMARY", "0"
            ).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            native_primary_requested = bool(abi_infer_primary)
        if abi_infer_primary_no_grad is None:
            native_primary_no_grad = os.getenv(
                "NATIVE_RUNNER_ABI_INFER_PRIMARY_NO_GRAD", "1"
            ).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            native_primary_no_grad = bool(abi_infer_primary_no_grad)
        if isinstance(result.native_abi_probe, dict):
            result.native_abi_probe["primary_requested"] = bool(
                native_primary_requested
            )
        native_primary_used = False
        if (
            native_primary_requested
            and native_primary_no_grad
            and abi_probe_logits is not None
        ):
            logits = torch.tensor(
                abi_probe_logits, dtype=torch.float32, device=dev
            ).view(1, 1, -1)
            logits = logits.expand(batch_size, seq_len, -1).contiguous()
            native_primary_used = True
            if isinstance(result.native_abi_probe, dict):
                result.native_abi_probe["primary_used"] = True
                result.native_abi_probe["mode"] = "primary_forward_only"

        def _run_forward() -> None:
            nonlocal logits
            with torch.amp.autocast(
                device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")
            ):
                logits = model(input_ids)

        forward_kernel = None
        if not native_primary_used:
            forward_kernel = op_profiler.profile_callable(_run_forward)
        if logits is None:
            _run_forward()

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t2 = time.perf_counter()
        result.forward_time_ms = (t2 - t1) * 1000
        if tracer is not None:
            tracer.stop("forward")

        result.output_shape = str(tuple(logits.shape))

        # Validate output shape: must be (batch, seq, vocab)
        if logits.dim() != 3:
            result.error = f"Expected 3D logits (batch, seq, vocab), got shape {tuple(logits.shape)}"
            result.error_type = "shape_mismatch"
            return result

        b_out, s_out, v_out = logits.shape
        if b_out != batch_size or s_out != seq_len:
            result.error = (
                f"Logits shape mismatch: got ({b_out}, {s_out}, {v_out}), "
                f"expected ({batch_size}, {seq_len}, *)"
            )
            result.error_type = "shape_mismatch"
            return result
        if v_out != vocab_size:
            result.error = (
                f"Logits vocab dim mismatch: got {v_out}, expected {vocab_size}"
            )
            result.error_type = "shape_mismatch"
            return result

        # Check output health
        result.has_nan_output = bool(torch.isnan(logits).any())
        result.has_inf_output = bool(torch.isinf(logits).any())
        if result.has_nan_output or result.has_inf_output:
            result.error = "NaN/Inf in forward output"
            result.error_type = "nan_forward"
            return result

        if native_primary_used and native_primary_no_grad:
            parity_sample_rate_raw = os.getenv(
                "NATIVE_RUNNER_ABI_PARITY_SAMPLE_RATE", "0.0"
            )
            try:
                parity_sample_rate = max(0.0, min(1.0, float(parity_sample_rate_raw)))
            except ValueError:
                logger.debug(
                    "Invalid NATIVE_RUNNER_ABI_PARITY_SAMPLE_RATE=%r; defaulting to 0.0",
                    parity_sample_rate_raw,
                )
                parity_sample_rate = 0.0
            parity_attempt = (
                parity_sample_rate > 0.0 and random.random() < parity_sample_rate
            )
            parity_max_abs = None
            parity_mean_abs = None
            parity_pass = None
            parity_reason = "not_sampled"
            parity_threshold_raw = os.getenv("NATIVE_RUNNER_ABI_PARITY_MAX_ABS", "1.0")
            try:
                parity_threshold = float(parity_threshold_raw)
            except ValueError:
                logger.debug(
                    "Invalid NATIVE_RUNNER_ABI_PARITY_MAX_ABS=%r; defaulting to 1.0",
                    parity_threshold_raw,
                )
                parity_threshold = 1.0
            parity_strict = os.getenv(
                "NATIVE_RUNNER_ABI_PARITY_STRICT", "0"
            ).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
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
                result.native_abi_probe["parity_sample_rate"] = float(
                    parity_sample_rate
                )
                result.native_abi_probe["parity_attempted"] = bool(parity_attempt)
                result.native_abi_probe["parity_pass"] = parity_pass
                result.native_abi_probe["parity_reason"] = parity_reason
                result.native_abi_probe["parity_max_abs_diff"] = parity_max_abs
                result.native_abi_probe["parity_mean_abs_diff"] = parity_mean_abs
                result.native_abi_probe["parity_max_abs_threshold"] = float(
                    parity_threshold
                )
                result.native_abi_probe["parity_strict"] = bool(parity_strict)

            if parity_attempt and parity_pass is False and parity_strict:
                result.error = (
                    "ABI parity regression in primary mode: "
                    f"reason={parity_reason}, max_abs={parity_max_abs}, threshold={parity_threshold}"
                )
                result.error_type = "abi_parity_regression"
                return result

            if dev.type == "cuda":
                result.peak_memory_mb = torch.cuda.max_memory_allocated(dev) / (1024**2)
            result.passed = True
            return result

        # Backward pass
        result.stage = "backward"
        if tracer is not None:
            tracer.start("backward", use_gpu=True)
        # Scale logits to prevent softmax saturation (which causes zero loss
        # and zero gradients).  We use the logits' own std as a dynamic
        # temperature — this keeps the cross-entropy in a numerically healthy
        # range regardless of the model's output magnitude.
        logits_for_loss = logits
        logit_std = logits.detach().std()
        if logit_std > 10.0:
            logits_for_loss = logits / (logit_std / 2.0)

        loss = F.cross_entropy(
            logits_for_loss.reshape(-1, logits_for_loss.shape[-1]),
            input_ids.reshape(-1),
        )

        def _run_backward() -> None:
            loss.backward()

        backward_kernel = op_profiler.profile_callable(_run_backward)
        if backward_kernel is None:
            _run_backward()

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t3 = time.perf_counter()
        result.backward_time_ms = (t3 - t2) * 1000
        if tracer is not None:
            tracer.stop("backward")

        # Check gradients
        has_nan = False
        has_zero = True
        n_with_grad = 0

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if grads:
            n_with_grad = len(grads)
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
        else:
            total_norm = 0.0

        result.grad_norm = float(total_norm)
        result.has_nan_grad = has_nan
        result.has_zero_grad = has_zero and n_with_grad > 0
        if has_nan:
            result.error = "NaN in gradients"
            result.error_type = "nan_grad"
            return result
        if has_zero:
            result.error = "All gradients are zero"
            result.error_type = "zero_grad"
            return result

        if forward_kernel or backward_kernel:
            result.kernel_timing = {
                "forward": forward_kernel,
                "backward": backward_kernel,
            }

        # ── Stage 0.5: Numerical stability probe ──
        if run_stability_probe:
            result.stage = "stability"
            if tracer is not None:
                tracer.start("stability", use_gpu=True)

            # Enable heatmap capture during stability probe
            if hasattr(model, "set_capture_heatmap"):
                model.set_capture_heatmap(True)

            stability = _stability_probe(model, dev, batch_size, seq_len, vocab_size)
            result.stability_score = stability["score"]
            result.extreme_input_passed = stability["extreme_passed"]
            result.random_input_passed = stability["random_passed"]
            result.causality_passed = stability["causality_passed"]
            result.output_range = stability.get("output_range")

            # Extract heatmaps if captured
            heatmaps = {}
            total_savings = 0.0
            total_depth_ratio = 0.0
            routing_op_count = 0

            for name, module in model.named_modules():
                rt = getattr(module, "routing_telemetry", None)
                if rt:
                    if rt.get("heatmap") is not None:
                        heatmaps[name] = rt["heatmap"]

                    # Z13: Aggregate routing efficiency
                    routing_op_count += 1
                    # Basic estimates for now
                    total_savings += rt.get("savings_ratio", 0.0)
                    total_depth_ratio += rt.get("depth_ratio", 1.0)

            if routing_op_count > 0:
                if getattr(result, "sparsity_report", None) is None:
                    result.sparsity_report = {}
                result.sparsity_report["routing_savings_ratio"] = round(
                    total_savings / routing_op_count, 4
                )
                result.sparsity_report["routing_depth_ratio"] = round(
                    total_depth_ratio / routing_op_count, 4
                )
                if heatmaps:
                    result.sparsity_report["routing_heatmaps"] = heatmaps

            if hasattr(model, "set_capture_heatmap"):
                model.set_capture_heatmap(False)

            if not result.causality_passed:
                result.passed = False
                result.error = (
                    "Strict Causality Gate Failed: Model looks ahead at future tokens."
                )
                result.error_type = "causality_violation"
                return result

            # Hard gate: reject architectures with chaotic training dynamics
            if stability.get("training_dynamics_passed") is False:
                result.passed = False
                _cv = stability.get("training_dynamics_cv", 0)
                _trend = stability.get("training_dynamics_trend", 0)
                result.error = (
                    f"Training dynamics unstable: CV={_cv:.3f}, "
                    f"trend={_trend:.3f} (10-step probe)"
                )
                result.error_type = "unstable_dynamics"
                return result

            if tracer is not None:
                tracer.stop("stability")

            # ── Activation Sparsity Check ──
            # Uses the last input_ids batch from forward pass
            sparsity_report = check_activation_sparsity(model, [input_ids])
            result.activation_sparsity = sparsity_report.overall_sparsity
            result.dead_neuron_count = sparsity_report.total_dead_neurons
            result.sparsity_report = {
                "dead_neuron_ratio": sparsity_report.dead_neuron_ratio,
                "max_layer_collapse": sparsity_report.max_layer_collapse,
                "n_collapsed_layers": sum(
                    1 for r in sparsity_report.layers if r.is_collapsed
                ),
            }

            if any(r.is_collapsed for r in sparsity_report.layers):
                result.passed = False
                result.error = f"Activation collapse: {result.sparsity_report['n_collapsed_layers']} layers collapsed"
                result.error_type = "activation_collapse"
                return result

        if dev.type == "cuda":
            result.peak_memory_mb = torch.cuda.max_memory_allocated(dev) / (1024**2)

        result.passed = True

    except TimeoutError:
        result.error = f"Timeout after {timeout_seconds}s in stage {result.stage}"
        result.error_type = "timeout"
    except torch.cuda.OutOfMemoryError:
        result.error = "CUDA out of memory"
        result.error_type = "oom"
    except Exception as e:
        if is_cuda_fatal(e):
            result.error = f"Fatal CUDA error in stage {result.stage}: {e}"
            result.error_type = "cuda_fatal"
            # Reset CUDA context after fatal error so subsequent evals can proceed
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    # Attempt a small allocation to verify recovery
                    _probe = torch.zeros(1, device="cuda")
                    del _probe
                    torch.cuda.synchronize()
                except Exception as recovery_exc:
                    logger.warning(
                        "CUDA context unrecoverable after fatal error: %s",
                        recovery_exc,
                    )
        else:
            tb = traceback.format_exc().strip().split("\n")
            result.error = "\n".join(tb[-3:])
            result.error_type = type(e).__name__
            # Op attribution: extract the failing op from the traceback.
            # Look for _op_<name> functions or CompiledOp._dispatch patterns.
            failure_op = None
            for line in reversed(tb):
                if "_op_" in line and "in _op_" in line:
                    # e.g. 'in _op_n_way_sparse_router'
                    import re as _re

                    m = _re.search(r"in (_op_\w+)", line)
                    if m:
                        failure_op = m.group(1).removeprefix("_op_")
                        break
                if "CompiledOp" in line and "forward" in line:
                    # Try to extract the op name from the CompiledOp repr
                    m = (
                        _re.search(r"CompiledOp\[(\w+)\]", line)
                        if "_re" in dir()
                        else None
                    )
                    if m:
                        failure_op = m.group(1)
                        break
            if failure_op is None:
                # Heuristic fallback from the error line itself
                if "kv_compress" in result.error:
                    failure_op = "latent_attention_compressor"
                elif "conv_weight" in result.error:
                    failure_op = "conv1d_seq"
            if failure_op:
                result.failure_op = failure_op
    finally:
        # Reset timeout
        try:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except (AttributeError, ValueError):
            pass

        # Cleanup — skip CUDA cache clear if context is dead
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

    return result


def _stability_probe(
    model: nn.Module,
    dev: torch.device,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
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

    def _check_ids(ids: torch.Tensor) -> Optional[torch.Tensor]:
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

    # Test 1: Multiple random inputs (check consistency)
    total_checks += 1
    outputs = []
    for _ in range(3):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)
        out = _check_ids(ids)
        if out is not None:
            outputs.append(out)

    if len(outputs) == 3:
        checks_passed += 1
        results["random_passed"] = True
        all_out = torch.cat([o.flatten() for o in outputs])
        results["output_range"] = (
            f"[{all_out.min().item():.2f}, {all_out.max().item():.2f}]"
        )

    # Test 2: Repeated tokens (stress test for attention patterns)
    total_checks += 1
    ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=dev)
    if _check_ids(ids) is not None:
        checks_passed += 1
        results["extreme_passed"] = True

    # Test 3: Sequential tokens (1, 2, 3, ...)
    total_checks += 1
    ids = (
        torch.arange(seq_len, device=dev).unsqueeze(0).expand(batch_size, -1)
        % vocab_size
    )
    if _check_ids(ids) is not None:
        checks_passed += 1

    # Test 4: High token IDs
    total_checks += 1
    ids = torch.full(
        (batch_size, seq_len), vocab_size - 1, dtype=torch.long, device=dev
    )
    if _check_ids(ids) is not None:
        checks_passed += 1

    # Test 5: Strict Causality Gate
    # Ensure that changing future tokens does not change past logits.
    # Tolerance 0.05: pointwise ops accumulate floating point drift through
    # deep graphs (4 layers × 7+ ops); real violations produce diff > 0.1.
    total_checks += 1
    try:
        with (
            torch.no_grad(),
            torch.amp.autocast(
                device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")
            ),
        ):
            ids_base = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)
            out_base = model(ids_base)

            ids_mod = ids_base.clone()
            midpoint = seq_len // 2
            ids_mod[:, midpoint:] = torch.randint(
                0, vocab_size, (batch_size, seq_len - midpoint), device=dev
            )
            out_mod = model(ids_mod)

            diff = (
                torch.abs(
                    out_base[:, :midpoint, :].float() - out_mod[:, :midpoint, :].float()
                )
                .max()
                .item()
            )
            if diff < 0.05:
                checks_passed += 1
                results["causality_passed"] = True
            else:
                results["causality_passed"] = False
    except Exception:
        results["causality_passed"] = False

    # Test 6: Training dynamics probe — fast gradient steps to detect chaotic loss
    # Uses higher LR (3e-3) to amplify instability faster in fewer steps
    total_checks += 1
    try:
        model.train()
        _probe_steps = 20
        _probe_lr = 1e-3
        _probe_optimizer = torch.optim.Adam(model.parameters(), lr=_probe_lr)
        _probe_losses: List[float] = []

        _use_amp = dev.type == "cuda"
        for _ in range(_probe_steps):
            ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)
            _probe_optimizer.zero_grad()
            with torch.amp.autocast(
                device_type=dev.type, dtype=torch.bfloat16, enabled=_use_amp
            ):
                logits = model(ids)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    ids[:, 1:].reshape(-1),
                )
            if torch.isnan(loss) or torch.isinf(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            _probe_optimizer.step()
            _probe_losses.append(loss.item())

        if len(_probe_losses) >= _probe_steps:
            _mean_l = sum(_probe_losses) / len(_probe_losses)
            if _mean_l > 0:
                _var_l = sum((x - _mean_l) ** 2 for x in _probe_losses) / len(
                    _probe_losses
                )
                _cv = (_var_l**0.5) / _mean_l
                # Check consecutive step-to-step sign changes (direction reversals)
                _sign_changes = sum(
                    1
                    for i in range(2, len(_probe_losses))
                    if (_probe_losses[i] - _probe_losses[i - 1])
                    * (_probe_losses[i - 1] - _probe_losses[i - 2])
                    < 0
                )
                _reversal_rate = _sign_changes / max(len(_probe_losses) - 2, 1)
                # Also check if loss decreased at all (last 5 vs first 5)
                _first5 = sum(_probe_losses[:5]) / 5
                _last5 = sum(_probe_losses[-5:]) / 5
                # Fail if: high volatility OR loss diverging
                # Reversal rate is noisy early in training, so only use CV + trend
                _dynamics_bad = (
                    _cv > 0.25  # moderate CV threshold
                    or (
                        _last5 > _first5 * 1.05 and _cv > 0.10
                    )  # loss increasing + unstable
                )
                if not _dynamics_bad:
                    checks_passed += 1
                    results["training_dynamics_passed"] = True
                else:
                    results["training_dynamics_passed"] = False
                    results["training_dynamics_cv"] = round(_cv, 4)
                    results["training_dynamics_trend"] = round(
                        _last5 / max(_first5, 1e-8), 4
                    )
                    results["training_dynamics_reversal_rate"] = round(
                        _reversal_rate, 4
                    )
            else:
                checks_passed += 1  # zero loss is fine
                results["training_dynamics_passed"] = True
        else:
            results["training_dynamics_passed"] = False  # NaN/Inf during probe
    except Exception:
        results["training_dynamics_passed"] = False

    results["score"] = checks_passed / max(total_checks, 1)
    model.train()
    return results
