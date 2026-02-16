"""
Architecture Evaluator

Multi-stage evaluation funnel for candidate architectures:
- Stage 0: Smoke test (seconds) — can it forward/backward without crashing?
- Stage 1: Micro-train (minutes) — does the loss actually decrease?

Each stage produces a structured result that feeds into the experiment database.
"""

from __future__ import annotations

import gc
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .morphological_box import ArchSpec
from .arch_builder import BuildConfig, ExplorerModel, build_model


# ── Result Types ───────────────────────────────────────────────────────

@dataclass
class Stage0Result:
    """Result from Stage 0 smoke test."""
    spec_id: str
    passed: bool = False
    error: Optional[str] = None
    error_type: Optional[str] = None
    # Metrics (only if passed)
    param_count: int = 0
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    output_shape: Optional[str] = None
    grad_norm: float = 0.0
    has_nan_grad: bool = False
    has_zero_grad: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Stage1Result:
    """Result from Stage 1 micro-training."""
    spec_id: str
    passed: bool = False
    error: Optional[str] = None
    # Training metrics
    steps_completed: int = 0
    initial_loss: float = float("inf")
    final_loss: float = float("inf")
    best_loss: float = float("inf")
    loss_ratio: float = float("inf")  # final/initial — <1 means learning
    avg_step_time_ms: float = 0.0
    throughput_tok_s: float = 0.0
    peak_memory_mb: float = 0.0
    loss_curve: List[float] = field(default_factory=list)
    # Gradient health
    avg_grad_norm: float = 0.0
    max_grad_norm: float = 0.0
    grad_norm_curve: List[float] = field(default_factory=list)
    # Convergence quality
    loss_decreasing: bool = False
    loss_stable: bool = False  # no NaN/Inf
    converges: bool = False  # loss_ratio < 0.8

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Stage 0: Smoke Test ────────────────────────────────────────────────

def stage0_smoke_test(
    spec: ArchSpec,
    config: Optional[BuildConfig] = None,
    device: str = "cuda",
    batch_size: int = 2,
    seq_len: int = 128,
) -> Stage0Result:
    """
    Stage 0: Can the model forward and backward without dying?

    Tests:
    1. Model instantiation
    2. Forward pass produces correct output shape
    3. Backward pass produces finite gradients
    4. No NaN/Inf in outputs or gradients

    Takes ~1-5 seconds per candidate.
    """
    result = Stage0Result(spec_id=spec.id)

    if config is None:
        config = BuildConfig(max_seq_len=seq_len)

    # Use config's max_seq_len if seq_len wasn't explicitly set smaller
    seq_len = min(seq_len, config.max_seq_len)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    try:
        # Reset memory tracking
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(dev)

        # 1. Build model
        model = build_model(spec, config).to(dev)
        result.param_count = model.param_count()

        # 2. Forward pass
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=dev)

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()

        with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
            logits = model(input_ids)

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t1 = time.perf_counter()
        result.forward_time_ms = (t1 - t0) * 1000

        # Check output shape
        expected_shape = (batch_size, seq_len, config.vocab_size)
        result.output_shape = str(tuple(logits.shape))
        if logits.shape != expected_shape:
            result.error = f"Bad output shape: got {logits.shape}, expected {expected_shape}"
            result.error_type = "shape_mismatch"
            return result

        # Check for NaN/Inf in output
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            result.error = "NaN/Inf in forward output"
            result.error_type = "nan_forward"
            return result

        # 3. Backward pass
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            input_ids.reshape(-1),
        )

        t2 = time.perf_counter()
        loss.backward()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t3 = time.perf_counter()
        result.backward_time_ms = (t3 - t2) * 1000

        # 4. Check gradients
        total_norm = 0.0
        has_nan = False
        has_zero = True  # assume zero until we find non-zero
        n_params_with_grad = 0

        for p in model.parameters():
            if p.grad is not None:
                n_params_with_grad += 1
                pnorm = p.grad.data.norm(2).item()
                total_norm += pnorm ** 2
                if torch.isnan(p.grad).any():
                    has_nan = True
                if pnorm > 1e-10:
                    has_zero = False

        result.grad_norm = total_norm ** 0.5
        result.has_nan_grad = has_nan
        result.has_zero_grad = has_zero and n_params_with_grad > 0

        if has_nan:
            result.error = "NaN in gradients"
            result.error_type = "nan_grad"
            return result

        if has_zero:
            result.error = "All gradients are zero (model is a no-op?)"
            result.error_type = "zero_grad"
            return result

        # Memory
        if dev.type == "cuda":
            result.peak_memory_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

        result.passed = True

    except Exception as e:
        result.error = f"{type(e).__name__}: {str(e)}"
        result.error_type = type(e).__name__
        # Capture first 3 lines of traceback
        tb = traceback.format_exc().strip().split("\n")
        result.error = "\n".join(tb[-3:])

    finally:
        # Cleanup
        if "model" in dir():
            del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return result


# ── Stage 1: Micro-Train ──────────────────────────────────────────────

def stage1_micro_train(
    spec: ArchSpec,
    config: Optional[BuildConfig] = None,
    device: str = "cuda",
    batch_size: int = 4,
    seq_len: int = 128,
    n_steps: int = 500,
    lr: float = 3e-4,
    log_every: int = 10,
    max_grad_norm: float = 1.0,
) -> Stage1Result:
    """
    Stage 1: Does the model actually learn anything?

    Trains for n_steps on random data, measuring:
    - Loss curve (should decrease)
    - Gradient health (should stay finite and non-zero)
    - Throughput (tokens/sec)
    - Memory usage

    Takes ~2-10 minutes per candidate depending on model size.
    """
    result = Stage1Result(spec_id=spec.id)

    if config is None:
        config = BuildConfig(max_seq_len=seq_len)

    seq_len = min(seq_len, config.max_seq_len)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    try:
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(dev)

        # Build model
        model = build_model(spec, config).to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        # Generate "training data" — random tokens
        # For Stage 1, we use random data. The point is whether the model can
        # memorize/learn patterns, not generalize.
        data = torch.randint(0, config.vocab_size, (n_steps, batch_size, seq_len), device=dev)

        model.train()
        step_times = []
        total_tokens = 0

        for step in range(n_steps):
            input_ids = data[step]
            targets = input_ids  # next-token prediction on same tokens (shifted internally)

            t0 = time.perf_counter()

            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, config.vocab_size),
                    targets[:, 1:].reshape(-1),
                )

            if torch.isnan(loss) or torch.isinf(loss):
                result.error = f"NaN/Inf loss at step {step}"
                result.steps_completed = step
                return result

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()

            optimizer.step()

            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t1 = time.perf_counter()

            step_time = (t1 - t0) * 1000
            step_times.append(step_time)
            total_tokens += batch_size * seq_len
            loss_val = loss.item()

            # Record metrics
            if step % log_every == 0:
                result.loss_curve.append(loss_val)
                result.grad_norm_curve.append(grad_norm)

            if step == 0:
                result.initial_loss = loss_val
            result.final_loss = loss_val
            result.best_loss = min(result.best_loss, loss_val)
            result.steps_completed = step + 1

            # Track grad norm stats
            result.avg_grad_norm += grad_norm
            result.max_grad_norm = max(result.max_grad_norm, grad_norm)

        # Compute summary stats
        result.avg_grad_norm /= n_steps
        result.avg_step_time_ms = sum(step_times) / len(step_times)
        result.throughput_tok_s = total_tokens / (sum(step_times) / 1000)

        if dev.type == "cuda":
            result.peak_memory_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

        # Convergence analysis
        result.loss_ratio = result.final_loss / max(result.initial_loss, 1e-6)
        result.loss_stable = not any(
            math.isnan(l) or math.isinf(l) for l in result.loss_curve
        )
        result.loss_decreasing = len(result.loss_curve) >= 2 and result.loss_curve[-1] < result.loss_curve[0]
        result.converges = result.loss_ratio < 0.8  # at least 20% loss reduction

        result.passed = result.loss_stable and result.converges

    except Exception as e:
        result.error = f"{type(e).__name__}: {str(e)}"
        tb = traceback.format_exc().strip().split("\n")
        result.error = "\n".join(tb[-3:])

    finally:
        if "model" in dir():
            del model
        if "optimizer" in dir():
            del optimizer
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return result


import math  # needed for loss_stable check
