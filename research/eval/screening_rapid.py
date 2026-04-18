"""Rapid pre-screening filter for architecture candidates.

Runs 150 gradient steps to detect fatal issues BEFORE committing
to full Stage 1 training. Catches:
  - NaN/Inf gradients (steps 1-5)
  - Exploding grad norms (step 10)
  - Stalled loss trajectory (steps 25, 50)
  - Routing collapse / expert starvation (step 50)
  - Post-minimum loss spike (step 100) — entropy collapse detection
  - Learning signal check (step 150) — must show improvement

Budget: 150 gradient steps, < 90 seconds on GPU.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .utils import language_model_loss, make_adamw

logger = logging.getLogger(__name__)

# Average GPU-minutes for a full Stage 1 run (used for savings estimates)
_AVG_S1_GPU_MINUTES = 2.5
_ENTROPY_SAMPLE_STEPS = frozenset({10, 25, 50, 75, 100, 150})


@dataclass(slots=True)
class ScreeningResult:
    """Outcome of the rapid screening check."""

    passed: bool
    kill_reason: Optional[str] = None
    kill_step: Optional[int] = None
    kill_metric: Optional[str] = None
    kill_value: Optional[float] = None
    kill_threshold: Optional[float] = None
    degraded: bool = False
    degraded_reasons: list = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    gpu_minutes_saved: float = 0.0


class RapidScreeningCheck:
    """Fast pre-screening filter. Runs before full Stage 1 training.

    Kills bad architectures in < 90 seconds.
    Budget: 150 gradient steps max.
    """

    __slots__ = (
        "grad_norm_hard_limit",
        "grad_norm_warning",
        "loss_at_step_25_limit",
        "loss_at_step_50_limit",
        "loss_spike_ratio",
        "routing_entropy_minimum",
        "nan_grace_steps",
        "max_steps",
        "lr",
        "clip_grad",
    )

    GRAD_NORM_HARD_LIMIT: float = 500.0
    GRAD_NORM_WARNING: float = 50.0
    # Cross-entropy is bounded above by ln(vocab_size); the historical 500/300
    # limits were tuned implicitly against vocab=256 (ln=5.55). Larger vocabs
    # legitimately produce higher initial loss, so the per-step limits below
    # are scaled by max(1, ln(vocab)/ln(256)) at check time. Calibrated 2026-04-17.
    LOSS_AT_STEP_25_LIMIT: float = 500.0
    LOSS_AT_STEP_50_LIMIT: float = 300.0
    LOSS_LIMIT_VOCAB_BASELINE: int = 256  # vocab the absolute limits are tuned for
    LOSS_SPIKE_RATIO: float = 2.0
    ROUTING_ENTROPY_MINIMUM: float = 0.05
    NAN_GRACE_STEPS: int = 5
    LOSS_CHECK_FINAL_STEP: int = 150
    MAX_STEPS: int = 150

    def __init__(
        self,
        *,
        grad_norm_hard_limit: float = GRAD_NORM_HARD_LIMIT,
        grad_norm_warning: float = GRAD_NORM_WARNING,
        loss_at_step_25_limit: float = LOSS_AT_STEP_25_LIMIT,
        loss_at_step_50_limit: float = LOSS_AT_STEP_50_LIMIT,
        loss_spike_ratio: float = LOSS_SPIKE_RATIO,
        routing_entropy_minimum: float = ROUTING_ENTROPY_MINIMUM,
        nan_grace_steps: int = NAN_GRACE_STEPS,
        max_steps: int = MAX_STEPS,
        lr: float = 3e-4,
        clip_grad: float = 1.0,
    ) -> None:
        self.grad_norm_hard_limit = grad_norm_hard_limit
        self.grad_norm_warning = grad_norm_warning
        self.loss_at_step_25_limit = loss_at_step_25_limit
        self.loss_at_step_50_limit = loss_at_step_50_limit
        self.loss_spike_ratio = loss_spike_ratio
        self.routing_entropy_minimum = routing_entropy_minimum
        self.nan_grace_steps = nan_grace_steps
        self.max_steps = max_steps
        self.lr = lr
        self.clip_grad = clip_grad

    def _init_run(
        self,
        model: nn.Module,
        device: str,
    ) -> tuple[torch.device, List[torch.Tensor], torch.optim.Optimizer]:
        dev = torch.device(device)
        model = model.to(dev)
        model.train()
        param_values = [param for param in model.parameters() if param.requires_grad]
        opt = make_adamw(param_values, lr=self.lr)
        return dev, param_values, opt

    def _handle_forward(
        self,
        model: nn.Module,
        batch: torch.Tensor,
        result: ScreeningResult,
        step: int,
    ) -> Optional[torch.Tensor]:
        try:
            return model(batch)
        except Exception as e:
            self._kill(
                result,
                step,
                "forward_error",
                None,
                None,
                f"Forward pass error at step {step}: {e}",
            )
            return None

    def _handle_backward(
        self,
        loss: torch.Tensor,
        result: ScreeningResult,
        step: int,
    ) -> bool:
        try:
            loss.backward()
            return True
        except Exception as e:
            self._kill(
                result,
                step,
                "backward_error",
                None,
                None,
                f"Backward pass error at step {step}: {e}",
            )
            return False

    def _sample_entropy_gate(
        self,
        step: int,
        has_entropy_gate: bool,
        entropy_values: List[float],
        entropy_gate_trajectory: List[float],
        metrics: Dict[str, Any],
        result: ScreeningResult,
    ) -> bool:
        should_sample_entropy = has_entropy_gate and step in _ENTROPY_SAMPLE_STEPS
        if not should_sample_entropy or not entropy_values:
            return True
        eg_val = sum(entropy_values) / len(entropy_values)
        entropy_gate_trajectory.append(eg_val)
        if eg_val < 0.05:
            metrics["routing_collapse_score"] = 1.0
            logger.warning(
                "entropy_gate_collapse_detected at step %d: value=%.4f",
                step,
                eg_val,
            )
            if step >= 50:
                self._kill(
                    result,
                    step,
                    "entropy_gate_collapse",
                    eg_val,
                    0.05,
                    f"Entropy gate collapsed to {eg_val:.4f} at step {step} "
                    f"— branch death imminent",
                )
                return False
        return True

    def _check_step_thresholds(
        self,
        step: int,
        loss_val: float,
        grad_norm: float,
        has_routing: bool,
        routing_modules: List[nn.Module],
        metrics: Dict[str, Any],
        result: ScreeningResult,
        min_loss_so_far: float,
        vocab_size: int,
    ) -> bool:
        # Cross-entropy is upper-bounded by ln(vocab_size); the absolute limits
        # were tuned for vocab=256 (ln=5.55). Scale linearly in entropy floor
        # so a vocab=50000 model isn't unfairly killed for a higher initial loss
        # that's still well under its entropy ceiling.
        vocab_norm = max(
            1.0,
            math.log(max(int(vocab_size), 2))
            / math.log(self.LOSS_LIMIT_VOCAB_BASELINE),
        )
        limit_25 = self.loss_at_step_25_limit * vocab_norm
        limit_50 = self.loss_at_step_50_limit * vocab_norm
        if step == 25 and loss_val > limit_25:
            self._kill(
                result,
                step,
                "loss_stalled_25",
                loss_val,
                limit_25,
                f"Loss {loss_val:.1f} > {limit_25:.1f} at step 25 "
                f"(vocab_norm={vocab_norm:.2f}, base={self.loss_at_step_25_limit:.0f})",
            )
            return False
        if step == 50 and loss_val > limit_50:
            self._kill(
                result,
                step,
                "loss_stalled_50",
                loss_val,
                limit_50,
                f"Loss {loss_val:.1f} > {limit_50:.1f} at step 50 "
                f"(vocab_norm={vocab_norm:.2f}, base={self.loss_at_step_50_limit:.0f})",
            )
            return False
        if step == 50 and has_routing:
            entropy = self._measure_routing_entropy(routing_modules)
            metrics["routing_entropy"] = entropy
            if entropy is not None and entropy < self.routing_entropy_minimum:
                self._kill(
                    result,
                    step,
                    "routing_collapse",
                    entropy,
                    self.routing_entropy_minimum,
                    f"Routing entropy {entropy:.4f} < {self.routing_entropy_minimum} at step 50",
                )
                return False
        if (
            step == 50
            and math.isfinite(grad_norm)
            and grad_norm > self.grad_norm_warning
        ):
            result.degraded = True
            result.degraded_reasons.append(
                f"Grad norm {grad_norm:.1f} > {self.grad_norm_warning} at step 50"
            )
        if (
            step == 100
            and min_loss_so_far > 0
            and loss_val > min_loss_so_far * self.loss_spike_ratio
        ):
            self._kill(
                result,
                step,
                "loss_spike_post_minimum",
                loss_val,
                min_loss_so_far * self.loss_spike_ratio,
                f"Loss spiked from {min_loss_so_far:.3f} to {loss_val:.3f} "
                f"at step {step} — entropy collapse suspected",
            )
            return False
        if step == self.max_steps and len(metrics["losses"]) >= self.max_steps:
            init_l = metrics["losses"][0]
            if init_l > 0:
                entropy_floor = math.log(vocab_size) if vocab_size > 0 else 10.37
                if init_l >= entropy_floor * 1.15:
                    improvement_rate = 0.02 * min(1.0, init_l / 25.0)
                    threshold = init_l * (1.0 - improvement_rate)
                    if loss_val >= threshold:
                        self._kill(
                            result,
                            step,
                            "no_learning_signal",
                            loss_val,
                            threshold,
                            f"No learning after {step} steps: "
                            f"init={init_l:.3f} final={loss_val:.3f} "
                            f"(threshold={threshold:.3f}, rate={improvement_rate:.3f})",
                        )
                        return False
        return True

    def _finalize_run(
        self,
        result: ScreeningResult,
        metrics: Dict[str, Any],
        entropy_gate_trajectory: List[float],
        t0: float,
    ) -> ScreeningResult:
        if entropy_gate_trajectory:
            metrics["entropy_gate_trajectory_json"] = entropy_gate_trajectory
        if metrics["losses"]:
            metrics["initial_loss"] = metrics["losses"][0]
            metrics["final_loss"] = metrics["losses"][-1]
            for checkpoint in (10, 25, 50, 75, 100, 150):
                if len(metrics["losses"]) >= checkpoint:
                    metrics[f"loss_at_{checkpoint}"] = metrics["losses"][checkpoint - 1]
        if metrics["grad_norms"]:
            finite_norms = [g for g in metrics["grad_norms"] if math.isfinite(g)]
            if finite_norms:
                metrics["max_grad_norm"] = max(finite_norms)
                metrics["mean_grad_norm"] = sum(finite_norms) / len(finite_norms)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        result.elapsed_ms = round(elapsed_ms, 1)
        if not result.passed:
            result.gpu_minutes_saved = round(_AVG_S1_GPU_MINUTES, 2)
        if result.passed:
            logger.info(
                "Rapid screening PASSED (%.0fms, %d steps)",
                elapsed_ms,
                metrics.get("steps_completed", 0),
            )
        else:
            logger.info(
                "Rapid screening KILLED at step %d: %s (saved ~%.1f GPU-min, %.0fms)",
                result.kill_step or 0,
                result.kill_reason or "unknown",
                result.gpu_minutes_saved,
                elapsed_ms,
            )
        return result

    def _run_step(
        self,
        model: nn.Module,
        batch: torch.Tensor,
        opt: torch.optim.Optimizer,
        param_values: List[torch.Tensor],
        step: int,
        vocab_size: int,
        has_entropy_gate: bool,
        has_routing: bool,
        routing_modules: List[nn.Module],
        entropy_values: List[float],
        entropy_gate_trajectory: List[float],
        metrics: Dict[str, Any],
        result: ScreeningResult,
        min_loss_so_far: float,
    ) -> tuple[bool, float]:
        should_sample_entropy = has_entropy_gate and step in _ENTROPY_SAMPLE_STEPS
        entropy_values.clear()
        logits = self._handle_forward(model, batch, result, step)
        if logits is None:
            return False, min_loss_so_far
        loss = language_model_loss(logits, batch, vocab_size)
        loss_val = loss.item()
        metrics["losses"].append(loss_val)

        if not math.isfinite(loss_val):
            self._kill(
                result,
                step,
                "loss_nan_inf",
                loss_val,
                None,
                f"Loss is {'NaN' if math.isnan(loss_val) else 'Inf'} at step {step}",
            )
            return False, min_loss_so_far
        if not self._handle_backward(loss, result, step):
            return False, min_loss_so_far

        if self.clip_grad > 0:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(param_values, self.clip_grad)
            )
        else:
            grad_norm = self._compute_grad_norm(param_values)
        metrics["grad_norms"].append(grad_norm)

        if step <= self.nan_grace_steps and not math.isfinite(grad_norm):
            self._kill(
                result,
                step,
                "grad_nan_inf",
                grad_norm,
                None,
                f"Gradient NaN/Inf at step {step} (within grace period)",
            )
            return False, min_loss_so_far
        if (
            step >= 10
            and math.isfinite(grad_norm)
            and grad_norm > self.grad_norm_hard_limit
        ):
            self._kill(
                result,
                step,
                "grad_norm_exploding",
                grad_norm,
                self.grad_norm_hard_limit,
                f"Grad norm {grad_norm:.1f} > {self.grad_norm_hard_limit} at step {step}",
            )
            return False, min_loss_so_far

        opt.step()
        metrics["steps_completed"] = step
        min_loss_so_far = min(min_loss_so_far, loss_val)
        if not self._sample_entropy_gate(
            step,
            should_sample_entropy,
            entropy_values,
            entropy_gate_trajectory,
            metrics,
            result,
        ):
            return False, min_loss_so_far
        ok = self._check_step_thresholds(
            step,
            loss_val,
            grad_norm,
            has_routing,
            routing_modules,
            metrics,
            result,
            min_loss_so_far,
            vocab_size,
        )
        return ok, min_loss_so_far

    def run(
        self,
        model: nn.Module,
        vocab_size: int,
        seq_len: int,
        batch_size: int,
        device: str,
    ) -> ScreeningResult:
        """Run the rapid screening check.

        Generates random data and trains for up to max_steps gradient steps.
        Returns ScreeningResult — check .passed to decide whether to proceed
        to Stage 1.

        Fail-fast: first fatal check kills immediately.
        """
        t0 = time.perf_counter()
        result = ScreeningResult(passed=True)
        metrics: Dict[str, Any] = {
            "grad_norms": [],
            "losses": [],
            "steps_completed": 0,
        }
        result.metrics = metrics

        dev, param_values, opt = self._init_run(model, device)

        entropy_modules, routing_modules = self._collect_probe_modules(model)
        has_routing = bool(routing_modules)
        metrics["has_routing"] = has_routing

        has_entropy_gate = bool(entropy_modules)
        metrics["has_entropy_gate"] = has_entropy_gate
        entropy_gate_trajectory: List[float] = []
        min_loss_so_far = float("inf")
        batch = torch.empty((batch_size, seq_len), dtype=torch.long, device=dev)
        entropy_capture_enabled = False
        entropy_values: List[float] = []
        entropy_hooks = self._register_entropy_hooks(
            entropy_modules,
            entropy_values,
            lambda: entropy_capture_enabled,
        )

        try:
            for step in range(1, self.max_steps + 1):
                batch.random_(0, vocab_size)
                opt.zero_grad(set_to_none=True)
                entropy_capture_enabled = (
                    has_entropy_gate and step in _ENTROPY_SAMPLE_STEPS
                )
                try:
                    ok, min_loss_so_far = self._run_step(
                        model,
                        batch,
                        opt,
                        param_values,
                        step,
                        vocab_size,
                        has_entropy_gate,
                        has_routing,
                        routing_modules,
                        entropy_values,
                        entropy_gate_trajectory,
                        metrics,
                        result,
                        min_loss_so_far,
                    )
                finally:
                    entropy_capture_enabled = False
                if not ok:
                    break
        finally:
            for hook in entropy_hooks:
                hook.remove()
        return self._finalize_run(result, metrics, entropy_gate_trajectory, t0)

    @staticmethod
    def _kill(
        result: ScreeningResult,
        step: int,
        metric_name: str,
        value: Optional[float],
        threshold: Optional[float],
        reason: str,
    ) -> None:
        result.passed = False
        result.kill_step = step
        result.kill_metric = metric_name
        result.kill_value = value
        result.kill_threshold = threshold
        result.kill_reason = reason

    @staticmethod
    def _compute_grad_norm(parameters: List[torch.Tensor]) -> float:
        """Compute total L2 gradient norm across all parameters."""
        total = 0.0
        for param in parameters:
            grad = param.grad
            if grad is None:
                continue
            norm = float(grad.norm().item())
            total += norm * norm
        return math.sqrt(total)

    @staticmethod
    def _collect_probe_modules(
        model: nn.Module,
    ) -> tuple[List[nn.Module], List[nn.Module]]:
        entropy_modules: List[nn.Module] = []
        routing_modules: List[nn.Module] = []
        for module in model.modules():
            op_name = getattr(module, "_op_name", None)
            if op_name and "entropy_score" in str(op_name):
                entropy_modules.append(module)
            if hasattr(module, "routing_telemetry"):
                routing_modules.append(module)
                continue
            if op_name and any(
                kw in str(op_name)
                for kw in ("route", "moe", "mixture", "expert", "gate_routing")
            ):
                routing_modules.append(module)
        return entropy_modules, routing_modules

    @staticmethod
    def _register_entropy_hooks(
        modules: List[nn.Module],
        storage: List[float],
        is_enabled,
    ) -> List[torch.utils.hooks.RemovableHandle]:
        hooks = []

        def hook_fn(module: nn.Module, input: Any, output: Any) -> None:  # noqa: A002
            if is_enabled() and isinstance(output, torch.Tensor):
                storage.append(output.abs().mean().item())

        for module in modules:
            hooks.append(module.register_forward_hook(hook_fn))
        return hooks

    @staticmethod
    def _measure_routing_entropy(routing_modules: List[nn.Module]) -> Optional[float]:
        """Extract routing utilization entropy from model's routing telemetry.

        Returns the average entropy across all routing modules, or None
        if no routing telemetry is available.
        """
        entropy_sum = 0.0
        count = 0
        for module in routing_modules:
            telemetry = getattr(module, "routing_telemetry", None)
            if telemetry is None or not isinstance(telemetry, dict):
                continue
            telem_count = telemetry.get("count", 0)
            if telem_count > 0:
                entropy_sum += telemetry.get("entropy_sum", 0.0) / telem_count
                count += 1
        if count == 0:
            return None
        return entropy_sum / count
