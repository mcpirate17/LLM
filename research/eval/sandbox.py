"""
Sandbox Execution

Safe evaluation of synthesized programs with:
- Timeout enforcement
- OOM catching
- NaN/Inf detection
- Gradient health checking
- Memory tracking
"""

from __future__ import annotations

import gc
import signal
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


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
        t0 = time.perf_counter()

        model = model.to(dev)
        result.param_count = sum(p.numel() for p in model.parameters())

        t1 = time.perf_counter()
        result.compile_time_ms = (t1 - t0) * 1000

        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)

        # Forward pass
        result.stage = "forward"
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)

        with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                enabled=(dev.type == "cuda")):
            logits = model(input_ids)

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t2 = time.perf_counter()
        result.forward_time_ms = (t2 - t1) * 1000

        result.output_shape = str(tuple(logits.shape))

        # Validate output shape: must be (batch, seq, vocab)
        if logits.dim() != 3:
            result.error = (
                f"Expected 3D logits (batch, seq, vocab), got shape {tuple(logits.shape)}"
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
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            input_ids.reshape(-1),
        )
        loss.backward()

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t3 = time.perf_counter()
        result.backward_time_ms = (t3 - t2) * 1000

        # Check gradients
        total_norm = 0.0
        has_nan = False
        has_zero = True
        n_with_grad = 0

        for p in model.parameters():
            if p.grad is not None:
                n_with_grad += 1
                pnorm = p.grad.data.norm(2).item()
                total_norm += pnorm ** 2
                if torch.isnan(p.grad).any():
                    has_nan = True
                if pnorm > 1e-10:
                    has_zero = False

        result.grad_norm = total_norm ** 0.5
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

        # ── Stage 0.5: Numerical stability probe ──
        if run_stability_probe:
            result.stage = "stability"
            stability = _stability_probe(model, dev, batch_size, seq_len, vocab_size)
            result.stability_score = stability["score"]
            result.extreme_input_passed = stability["extreme_passed"]
            result.random_input_passed = stability["random_passed"]
            result.output_range = stability.get("output_range")

        if dev.type == "cuda":
            result.peak_memory_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

        result.passed = True

    except TimeoutError:
        result.error = f"Timeout after {timeout_seconds}s in stage {result.stage}"
        result.error_type = "timeout"
    except torch.cuda.OutOfMemoryError:
        result.error = "CUDA out of memory"
        result.error_type = "oom"
    except Exception as e:
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

        # Cleanup
        if dev.type == "cuda":
            torch.cuda.empty_cache()
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

    with torch.no_grad():
        # Test 1: Multiple random inputs (check consistency)
        total_checks += 1
        try:
            outputs = []
            for _ in range(3):
                ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)
                with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                        enabled=(dev.type == "cuda")):
                    out = model(ids)
                if not (torch.isnan(out).any() or torch.isinf(out).any()):
                    outputs.append(out)

            if len(outputs) == 3:
                checks_passed += 1
                results["random_passed"] = True
                # Check output range
                all_out = torch.cat([o.flatten() for o in outputs])
                results["output_range"] = f"[{all_out.min().item():.2f}, {all_out.max().item():.2f}]"
        except Exception:
            pass

        # Test 2: Repeated tokens (stress test for attention patterns)
        total_checks += 1
        try:
            ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=dev)
            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                    enabled=(dev.type == "cuda")):
                out = model(ids)
            if not (torch.isnan(out).any() or torch.isinf(out).any()):
                checks_passed += 1
                results["extreme_passed"] = True
        except Exception:
            pass

        # Test 3: Sequential tokens (1, 2, 3, ...)
        total_checks += 1
        try:
            ids = torch.arange(seq_len, device=dev).unsqueeze(0).expand(batch_size, -1)
            ids = ids % vocab_size
            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                    enabled=(dev.type == "cuda")):
                out = model(ids)
            if not (torch.isnan(out).any() or torch.isinf(out).any()):
                checks_passed += 1
        except Exception:
            pass

        # Test 4: High token IDs
        total_checks += 1
        try:
            ids = torch.full((batch_size, seq_len), vocab_size - 1,
                           dtype=torch.long, device=dev)
            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                    enabled=(dev.type == "cuda")):
                out = model(ids)
            if not (torch.isnan(out).any() or torch.isinf(out).any()):
                checks_passed += 1
        except Exception:
            pass

    results["score"] = checks_passed / max(total_checks, 1)
    model.train()
    return results
