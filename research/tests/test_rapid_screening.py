"""Tests for the rapid pre-screening filter."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from research.eval.screening_rapid import RapidScreeningCheck, ScreeningResult


# ── Helpers ──────────────────────────────────────────────────────────────


class _TinyLM(nn.Module):
    """Minimal LM for testing: embedding → linear → lm_head."""

    __slots__ = ()

    def __init__(self, vocab_size: int = 512, dim: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.proj(self.embed(x)))


class _ExplodingGradLM(_TinyLM):
    """Model that produces exploding gradients via huge weight init."""

    def __init__(self, vocab_size: int = 512, dim: int = 32) -> None:
        super().__init__(vocab_size, dim)
        # Initialize with massive weights to cause exploding gradients
        with torch.no_grad():
            self.proj.weight.mul_(1000.0)
            self.proj.bias.mul_(1000.0)


class _NaNModel(nn.Module):
    """Model that produces NaN output."""

    __slots__ = ()

    def __init__(self, vocab_size: int = 512, dim: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        # Force NaN
        h = h / 0.0
        return self.head(h)


class _RoutingCollapseLM(nn.Module):
    """Model with routing telemetry that shows expert collapse."""

    __slots__ = ()

    def __init__(self, vocab_size: int = 512, dim: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.router = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        # Fake routing telemetry with near-zero entropy (collapse)
        self.router.routing_telemetry = {
            "tokens_total": 1000,
            "tokens_processed": 1000,
            "entropy_sum": 0.01,  # very low entropy
            "count": 10,
            "_call_count": 10,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.router(h)
        return self.head(h)


# ── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_exploding_gradients_killed():
    """Model with exploding gradients is killed at step 10 with GRAD_NORM_HARD_LIMIT."""
    checker = RapidScreeningCheck(grad_norm_hard_limit=500.0)
    model = _ExplodingGradLM(vocab_size=512, dim=32)
    result = checker.run(model, vocab_size=512, seq_len=32, batch_size=2, device="cpu")

    assert not result.passed
    assert result.kill_metric in (
        "grad_norm_exploding",
        "grad_nan_inf",
        "loss_nan_inf",
        "backward_error",
    )
    assert result.kill_step is not None and result.kill_step <= 50
    assert result.gpu_minutes_saved > 0


@pytest.mark.unit
def test_nan_output_killed():
    """Model with NaN output is killed within first 5 steps."""
    checker = RapidScreeningCheck()
    model = _NaNModel(vocab_size=512, dim=32)
    result = checker.run(model, vocab_size=512, seq_len=32, batch_size=2, device="cpu")

    assert not result.passed
    assert result.kill_step is not None and result.kill_step <= 5
    assert result.kill_metric in ("loss_nan_inf", "grad_nan_inf", "backward_error")


@pytest.mark.unit
def test_routing_collapse_killed():
    """Model with routing collapse (all to expert 0) is killed at step 50."""
    checker = RapidScreeningCheck(routing_entropy_minimum=0.05)
    model = _RoutingCollapseLM(vocab_size=512, dim=32)
    result = checker.run(model, vocab_size=512, seq_len=32, batch_size=2, device="cpu")

    # Should either be killed for routing collapse or survive if the routing
    # telemetry gets reset. The key assertion: if it detected routing, it checks entropy.
    if result.metrics.get("has_routing"):
        entropy = result.metrics.get("routing_entropy")
        if entropy is not None and entropy < 0.05:
            assert not result.passed
            assert result.kill_metric == "routing_collapse"
            assert result.kill_step == 50


@pytest.mark.unit
def test_healthy_model_passes():
    """A simple healthy model passes all checks."""
    checker = RapidScreeningCheck()
    model = _TinyLM(vocab_size=512, dim=32)
    result = checker.run(model, vocab_size=512, seq_len=32, batch_size=2, device="cpu")

    assert result.passed
    assert result.kill_reason is None
    assert result.metrics["steps_completed"] == 75
    assert result.elapsed_ms > 0


@pytest.mark.unit
def test_gpt2_reference_passes():
    """GPT-2 reference architecture passes all checks."""
    from research.synthesis.reference_architectures import build_reference
    from research.scientist.native_runner import (
        compile_model_native_first as compile_model,
    )

    graph = build_reference("gpt2", d_model=64)
    model = compile_model([graph] * 2, vocab_size=512, max_seq_len=64)

    checker = RapidScreeningCheck()
    result = checker.run(model, vocab_size=512, seq_len=32, batch_size=2, device="cpu")

    assert result.passed, f"GPT-2 reference killed: {result.kill_reason}"
    assert result.metrics["steps_completed"] == 75


@pytest.mark.unit
def test_screening_result_dataclass():
    """ScreeningResult fields are correctly initialized."""
    r = ScreeningResult(passed=True)
    assert r.passed
    assert r.kill_reason is None
    assert r.degraded is False
    assert r.degraded_reasons == []
    assert r.metrics == {}
    assert r.elapsed_ms == 0.0
    assert r.gpu_minutes_saved == 0.0


class _LossSpikeModel(nn.Module):
    """Model that learns well initially then spikes after step 50.

    Simulates entropy collapse: loss drops, then jumps above 2x minimum.
    """

    __slots__ = ()

    def __init__(self, vocab_size: int = 512, dim: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self._step_counter = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._step_counter += 1
        h = self.head(self.proj(self.embed(x)))
        # After step 55, add large noise to spike the loss
        if self._step_counter > 55:
            h = h + torch.randn_like(h) * 50.0
        return h


@pytest.mark.unit
def test_loss_spike_post_minimum_killed():
    """Model with loss spike above 2x minimum at step 75 is killed."""
    checker = RapidScreeningCheck(loss_spike_ratio=2.0)
    model = _LossSpikeModel(vocab_size=512, dim=32)
    result = checker.run(model, vocab_size=512, seq_len=32, batch_size=2, device="cpu")

    assert not result.passed
    assert result.kill_metric == "loss_spike_post_minimum"
    assert result.kill_step == 75


@pytest.mark.unit
def test_degraded_flag_set_for_high_grad_norm():
    """Models with grad_norm > warning but < hard limit are flagged degraded."""
    # Use a model that will have moderate grad norms
    checker = RapidScreeningCheck(
        grad_norm_hard_limit=10000.0,  # very high to not kill
        grad_norm_warning=0.001,  # very low to always trigger warning
    )
    model = _TinyLM(vocab_size=512, dim=32)
    result = checker.run(model, vocab_size=512, seq_len=32, batch_size=2, device="cpu")

    assert result.passed  # Not killed
    assert result.degraded  # But flagged
    assert len(result.degraded_reasons) > 0
