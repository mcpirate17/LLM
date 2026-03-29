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
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Average GPU-minutes for a full Stage 1 run (used for savings estimates)
_AVG_S1_GPU_MINUTES = 2.5


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
    LOSS_AT_STEP_25_LIMIT: float = 500.0
    LOSS_AT_STEP_50_LIMIT: float = 300.0
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

        dev = torch.device(device)
        model = model.to(dev)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr)

        has_routing = self._detect_routing(model)
        metrics["has_routing"] = has_routing

        has_entropy_gate = self._detect_entropy_gate(model)
        metrics["has_entropy_gate"] = has_entropy_gate
        entropy_gate_trajectory: List[float] = []
        _ENTROPY_SAMPLE_STEPS = frozenset({10, 25, 50, 75, 100, 150})
        min_loss_so_far = float("inf")

        for step in range(1, self.max_steps + 1):
            # Generate random training batch
            batch = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)

            opt.zero_grad(set_to_none=True)

            # Forward
            try:
                logits = model(batch)
            except Exception as e:
                self._kill(
                    result,
                    step,
                    "forward_error",
                    None,
                    None,
                    f"Forward pass error at step {step}: {e}",
                )
                break

            sl = logits[:, :-1].contiguous()
            if sl.shape[-1] > vocab_size:
                sl = sl[..., :vocab_size]
            loss = F.cross_entropy(
                sl.reshape(-1, sl.shape[-1]),
                batch[:, 1:].reshape(-1),
            )
            loss_val = loss.item()
            metrics["losses"].append(loss_val)

            # CHECK 1: NaN/Inf loss
            if not math.isfinite(loss_val):
                self._kill(
                    result,
                    step,
                    "loss_nan_inf",
                    loss_val,
                    None,
                    f"Loss is {'NaN' if math.isnan(loss_val) else 'Inf'} at step {step}",
                )
                break

            # Backward
            try:
                loss.backward()
            except Exception as e:
                self._kill(
                    result,
                    step,
                    "backward_error",
                    None,
                    None,
                    f"Backward pass error at step {step}: {e}",
                )
                break

            # Compute grad norm (pre-clip)
            grad_norm = self._compute_grad_norm(model)
            metrics["grad_norms"].append(grad_norm)

            # CHECK 2: NaN/Inf in gradients (steps 1-NAN_GRACE_STEPS)
            if step <= self.nan_grace_steps:
                if not math.isfinite(grad_norm):
                    self._kill(
                        result,
                        step,
                        "grad_nan_inf",
                        grad_norm,
                        None,
                        f"Gradient NaN/Inf at step {step} (within grace period)",
                    )
                    break

            # CHECK 3: Grad norm hard limit (step 10+)
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
                break

            # Clip and step
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
            opt.step()
            metrics["steps_completed"] = step
            min_loss_so_far = min(min_loss_so_far, loss_val)

            # Entropy gate trajectory sampling
            if has_entropy_gate and step in _ENTROPY_SAMPLE_STEPS:
                eg_val = self._sample_entropy_gate(model, batch)
                if eg_val is not None:
                    entropy_gate_trajectory.append(eg_val)
                    # CHECK: entropy gate collapse (near-zero → branch death)
                    if eg_val < 0.05:
                        metrics["routing_collapse_score"] = 1.0
                        logger.warning(
                            "entropy_gate_collapse_detected at step %d: value=%.4f",
                            step,
                            eg_val,
                        )
                        # Kill at step 50+ if collapsed
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
                            break

            # CHECK 4: Loss trajectory at step 25
            if step == 25 and loss_val > self.loss_at_step_25_limit:
                self._kill(
                    result,
                    step,
                    "loss_stalled_25",
                    loss_val,
                    self.loss_at_step_25_limit,
                    f"Loss {loss_val:.1f} > {self.loss_at_step_25_limit} at step 25",
                )
                break

            # CHECK 5: Loss trajectory at step 50
            if step == 50 and loss_val > self.loss_at_step_50_limit:
                self._kill(
                    result,
                    step,
                    "loss_stalled_50",
                    loss_val,
                    self.loss_at_step_50_limit,
                    f"Loss {loss_val:.1f} > {self.loss_at_step_50_limit} at step 50",
                )
                break

            # CHECK 6: Routing collapse at step 50
            if step == 50 and has_routing:
                entropy = self._measure_routing_entropy(model)
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
                    break

            # CHECK 7: Grad norm warning at step 50 (degraded, not killed)
            if (
                step == 50
                and math.isfinite(grad_norm)
                and grad_norm > self.grad_norm_warning
            ):
                result.degraded = True
                result.degraded_reasons.append(
                    f"Grad norm {grad_norm:.1f} > {self.grad_norm_warning} at step 50"
                )

            # CHECK 8: Post-minimum loss spike at step 100 (entropy collapse)
            if step == 100:
                min_loss = min(metrics["losses"])
                if min_loss > 0 and loss_val > min_loss * self.loss_spike_ratio:
                    self._kill(
                        result,
                        step,
                        "loss_spike_post_minimum",
                        loss_val,
                        min_loss * self.loss_spike_ratio,
                        f"Loss spiked from {min_loss:.3f} to {loss_val:.3f} "
                        f"at step {step} — entropy collapse suspected",
                    )
                    break

            # CHECK 9: Learning signal at final step — must show some
            # improvement over initial loss. Uses an adaptive threshold
            # that accounts for the initial loss level.
            #
            # Problem: fixed 2% threshold penalizes complex architectures
            # that start closer to the entropy floor (low init_loss).
            # Data shows: 45% of 11-op graphs killed vs 0% of 4-op graphs.
            #
            # Fix: scale the required improvement by initial loss level.
            # High init_loss (>50): require 2% relative improvement (unchanged)
            # Low init_loss (<50): require progressively less, down to 0.5%
            # at init_loss near ln(vocab)≈5.5 (entropy floor).
            #
            # Formula: threshold = init * (1 - improvement_rate)
            # where improvement_rate = 0.02 * min(1, init_loss / 25)
            # At init=190: rate=2.0%, must drop 3.8 nats
            # At init=25:  rate=2.0%, must drop 0.5 nats
            # At init=12:  rate=1.0%, must drop 0.12 nats
            # At init=6:   rate=0.5%, must drop 0.03 nats
            if step == self.max_steps and len(metrics["losses"]) >= self.max_steps:
                init_l = metrics["losses"][0]
                if init_l > 0:
                    # Entropy floor bypass: if a model starts within 15%
                    # of ln(vocab), it's already well-initialized. The
                    # "no learning" check is meaningless because there's
                    # no headroom. This occurs with deeper models (6+
                    # layers) at small qualifying vocab sizes.
                    entropy_floor = math.log(vocab_size) if vocab_size > 0 else 10.37
                    if init_l < entropy_floor * 1.15:
                        # Near entropy floor — skip learning signal check
                        pass
                    else:
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
                            break

        # Entropy gate trajectory
        if entropy_gate_trajectory:
            metrics["entropy_gate_trajectory_json"] = entropy_gate_trajectory

        # Summary metrics
        if metrics["losses"]:
            metrics["initial_loss"] = metrics["losses"][0]
            metrics["final_loss"] = metrics["losses"][-1]
            if len(metrics["losses"]) >= 10:
                metrics["loss_at_10"] = metrics["losses"][9]
            if len(metrics["losses"]) >= 25:
                metrics["loss_at_25"] = metrics["losses"][24]
            if len(metrics["losses"]) >= 50:
                metrics["loss_at_50"] = metrics["losses"][49]
            if len(metrics["losses"]) >= 75:
                metrics["loss_at_75"] = metrics["losses"][74]
            if len(metrics["losses"]) >= 100:
                metrics["loss_at_100"] = metrics["losses"][99]
            if len(metrics["losses"]) >= 150:
                metrics["loss_at_150"] = metrics["losses"][149]
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
    def _compute_grad_norm(model: nn.Module) -> float:
        """Compute total L2 gradient norm across all parameters."""
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.data.float().norm().item() ** 2
        return total**0.5

    @staticmethod
    def _detect_entropy_gate(model: nn.Module) -> bool:
        """Check if model contains entropy_score ops (entropy-gated routing)."""
        for module in model.modules():
            op_name = getattr(module, "_op_name", None)
            if op_name and "entropy_score" in str(op_name):
                return True
        return False

    @staticmethod
    def _sample_entropy_gate(model: nn.Module, batch: torch.Tensor) -> Optional[float]:
        """Run a forward pass and capture the mean entropy_score output.

        Returns the mean absolute value of all entropy_score op outputs,
        or None if no entropy_score ops are found.
        """
        entropy_values: List[float] = []

        hooks = []

        def _make_hook(storage: List[float]):
            def hook_fn(module: nn.Module, input: Any, output: Any) -> None:  # noqa: A002 — shadow builtin
                if isinstance(output, torch.Tensor):
                    storage.append(output.abs().mean().item())

            return hook_fn

        for module in model.modules():
            op_name = getattr(module, "_op_name", None)
            if op_name and "entropy_score" in str(op_name):
                hooks.append(module.register_forward_hook(_make_hook(entropy_values)))

        if not hooks:
            return None

        with torch.no_grad():
            try:
                model(batch)
            except Exception:
                pass
            finally:
                for h in hooks:
                    h.remove()

        if not entropy_values:
            return None
        return sum(entropy_values) / len(entropy_values)

    @staticmethod
    def _detect_routing(model: nn.Module) -> bool:
        """Check if model contains routing/MoE ops by looking for routing_telemetry."""
        for module in model.modules():
            if hasattr(module, "routing_telemetry"):
                return True
            # CompiledOp marks routing ops in the graph
            op_name = getattr(module, "_op_name", None)
            if op_name and any(
                kw in str(op_name)
                for kw in ("route", "moe", "mixture", "expert", "gate_routing")
            ):
                return True
        return False

    @staticmethod
    def _measure_routing_entropy(model: nn.Module) -> Optional[float]:
        """Extract routing utilization entropy from model's routing telemetry.

        Returns the average entropy across all routing modules, or None
        if no routing telemetry is available.
        """
        entropy_sum = 0.0
        count = 0
        for module in model.modules():
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
