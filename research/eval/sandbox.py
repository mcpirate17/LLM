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
import os
import signal
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..scientist.perf import PerfTracer, OpKernelProfiler

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


def cuda_health_check(device: str = "cuda") -> bool:
    """Probe whether CUDA is still functional.

    Attempts a tiny tensor allocation + sync on the given device.
    Returns True if healthy, False if the CUDA context is dead.
    """
    if not torch.cuda.is_available():
        return False
    try:
        dev = torch.device(device)
        t = torch.zeros(1, device=dev)
        del t
        torch.cuda.synchronize(dev)
        return True
    except Exception:
        return False


@dataclass
class SandboxResult:
    """Result of sandbox evaluation."""
    passed: bool = False
    stage: str = ""  # "compile", "forward", "backward", "stability"
    error: Optional[str] = None
    error_type: Optional[str] = None
    # Timing
    compile_time_ms: float = 0.0
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    # Metrics
    param_count: int = 0
    peak_memory_mb: float = 0.0
    output_shape: Optional[str] = None
    # Gradient health
    grad_norm: float = 0.0
    has_nan_grad: bool = False
    has_zero_grad: bool = False
    has_nan_output: bool = False
    has_inf_output: bool = False
    # Numerical stability (Stage 0.5)
    stability_score: float = 0.0  # 0-1, higher is more stable
    extreme_input_passed: bool = False
    random_input_passed: bool = False
    output_range: Optional[str] = None
    kernel_timing: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


def _mapped_shared_token_ids(batch_size: int, seq_len: int, vocab_size: int):
    """Create token IDs using a zero-copy NumPy->Torch view without disk I/O."""
    arr = np.empty((batch_size, seq_len), dtype=np.int64)
    arr[:] = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)
    tensor = torch.from_numpy(arr)
    return tensor, arr, None


def safe_eval(
    model: nn.Module,
    batch_size: int = 2,
    seq_len: int = 128,
    vocab_size: int = 32000,
    device: str = "cuda",
    timeout_seconds: int = 30,
    run_stability_probe: bool = True,
) -> SandboxResult:
    """Safely evaluate a model through Stage 0 and Stage 0.5.

    Stage 0: Compilation + forward + backward
    Stage 0.5: Numerical stability probe
    """
    result = SandboxResult()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    trace_enabled = os.getenv("AI_SCI_PERF_TRACE", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    tracer = PerfTracer() if trace_enabled else None
    kernel_profile_enabled = os.getenv("AI_SCI_KERNEL_PROFILE", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    op_profiler = OpKernelProfiler(enabled=kernel_profile_enabled, top_k=20)
    mapped_array = None
    mapped_path = None

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
            shared_ids, mapped_array, mapped_path = _mapped_shared_token_ids(batch_size, seq_len, vocab_size)
            input_ids = shared_ids
        logits = None

        def _run_forward() -> None:
            nonlocal logits
            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                    enabled=(dev.type == "cuda")):
                logits = model(input_ids)

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
            result.error = (
                f"Expected 3D logits (batch, seq, vocab), got shape {tuple(logits.shape)}"
            )
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

        # Backward pass
        result.stage = "backward"
        if tracer is not None:
            tracer.start("backward", use_gpu=True)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
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
        total_norm = 0.0
        has_nan = False
        has_zero = True
        n_with_grad = 0

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if grads:
            n_with_grad = len(grads)
            try:
                norms = torch._foreach_norm(grads, 2)
                norm_vec = torch.stack([n.detach() for n in norms])
                total_norm = float(torch.linalg.vector_norm(norm_vec, ord=2).item())
                has_nan = not bool(torch.isfinite(norm_vec).all().item())
                has_zero = not bool((norm_vec > 1e-10).any().item())
            except Exception:
                for grad in grads:
                    pnorm = grad.data.norm(2).item()
                    total_norm += pnorm ** 2
                    if torch.isnan(grad).any():
                        has_nan = True
                    if pnorm > 1e-10:
                        has_zero = False
                total_norm = total_norm ** 0.5

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
            stability = _stability_probe(model, dev, batch_size, seq_len, vocab_size)
            result.stability_score = stability["score"]
            result.extreme_input_passed = stability["extreme_passed"]
            result.random_input_passed = stability["random_passed"]
            result.output_range = stability.get("output_range")
            if tracer is not None:
                tracer.stop("stability")

        if dev.type == "cuda":
            result.peak_memory_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

        result.passed = True
        
        # Attach detailed perf info to result if needed
        # result.metadata["perf"] = tracer.get_summary()

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
        else:
            tb = traceback.format_exc().strip().split("\n")
            result.error = "\n".join(tb[-3:])
            result.error_type = type(e).__name__
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
        if dev.type == "cuda" and empty_cache_every > 0 and (_SAFE_EVAL_CALL_COUNT % empty_cache_every == 0):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass  # CUDA context may be corrupted
        # Cleanup mapped shared-memory buffer
        try:
            if mapped_array is not None:
                del mapped_array
            if mapped_path and os.path.exists(mapped_path):
                os.remove(mapped_path)
        except Exception:
            pass
        if force_gc_every > 0 and (_SAFE_EVAL_CALL_COUNT % force_gc_every == 0):
            gc.collect()

    return result


def _stability_probe(
    model: nn.Module, dev: torch.device,
    batch_size: int, seq_len: int, vocab_size: int,
) -> Dict:
    """Run numerical stability probes."""
    model.eval()
    results = {"score": 0.0, "extreme_passed": False, "random_passed": False}
    checks_passed = 0
    total_checks = 0

    def _check_ids(ids: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            with torch.no_grad(), torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                                    enabled=(dev.type == "cuda")):
                out = model(ids)
            if not (torch.isnan(out).any() or torch.isinf(out).any()):
                return out
        except Exception:
            pass
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
        results["output_range"] = f"[{all_out.min().item():.2f}, {all_out.max().item():.2f}]"

    # Test 2: Repeated tokens (stress test for attention patterns)
    total_checks += 1
    ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=dev)
    if _check_ids(ids) is not None:
        checks_passed += 1
        results["extreme_passed"] = True

    # Test 3: Sequential tokens (1, 2, 3, ...)
    total_checks += 1
    ids = torch.arange(seq_len, device=dev).unsqueeze(0).expand(batch_size, -1) % vocab_size
    if _check_ids(ids) is not None:
        checks_passed += 1

    # Test 4: High token IDs
    total_checks += 1
    ids = torch.full((batch_size, seq_len), vocab_size - 1, dtype=torch.long, device=dev)
    if _check_ids(ids) is not None:
        checks_passed += 1

    results["score"] = checks_passed / max(total_checks, 1)
    model.train()
    return results
