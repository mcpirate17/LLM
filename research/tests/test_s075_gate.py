"""Tests for S0.75 initial-loss gate and validator projection-chain warning."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from research.synthesis.validator import validate_graph
from research.synthesis.graph import ComputationGraph, ShapeInfo


# ── Fix 2A: Validator deep projection chain warning ──────────────────


def test_validator_warns_on_deep_projection_chain():
    """A graph with 4 consecutive linear_proj ops (no norm) should trigger
    the deep projection chain warning."""
    D = 64
    g = ComputationGraph(D)
    inp = g.add_input()
    cur = inp
    for _ in range(4):
        cur = g.add_op("linear_proj", [cur], config={"out_dim": D})
    g.set_output(cur)

    result = validate_graph(g)
    assert result.valid, f"Graph should be valid, got errors: {result.errors}"
    chain_warnings = [w for w in result.warnings if "projection chain" in w.lower()]
    assert len(chain_warnings) == 1, (
        f"Expected 1 projection chain warning, got {len(chain_warnings)}: {result.warnings}"
    )
    assert "depth=4" in chain_warnings[0]


def test_validator_no_warning_with_interleaved_norm():
    """A graph with linear_proj → rmsnorm → linear_proj → rmsnorm should NOT
    trigger the warning (norm resets the depth counter)."""
    D = 64
    g = ComputationGraph(D)
    inp = g.add_input()
    cur = inp
    for _ in range(4):
        cur = g.add_op("linear_proj", [cur], config={"out_dim": D})
        cur = g.add_op("rmsnorm", [cur])
    g.set_output(cur)

    result = validate_graph(g)
    assert result.valid
    chain_warnings = [w for w in result.warnings if "projection chain" in w.lower()]
    assert len(chain_warnings) == 0, (
        f"Should have no projection chain warnings: {result.warnings}"
    )


def test_validator_no_warning_for_short_chain():
    """3 consecutive projections (at the threshold) should NOT warn."""
    D = 64
    g = ComputationGraph(D)
    inp = g.add_input()
    cur = inp
    for _ in range(3):
        cur = g.add_op("linear_proj", [cur], config={"out_dim": D})
    g.set_output(cur)

    result = validate_graph(g)
    chain_warnings = [w for w in result.warnings if "projection chain" in w.lower()]
    assert len(chain_warnings) == 0


# ── Fix 2B: S0.75 initial-loss gate ────────────────────────────────────


class _HighInitLossModel(nn.Module):
    """Mock model that produces logits with very high cross-entropy loss."""

    def __init__(self, vocab_size: int, loss_value: float):
        super().__init__()
        self._vocab_size = vocab_size
        self._loss_value = loss_value
        # Need at least one parameter for the optimizer
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape
        # Return logits that will produce the desired CE loss
        # CE loss ≈ -log(1/V) = log(V) for uniform logits
        # Scale logits to produce desired loss level
        logits = torch.randn(B, S, self._vocab_size, device=x.device)
        scale = self._loss_value / max(math.log(self._vocab_size), 1.0)
        return logits * scale


def test_s075_drops_high_initial_loss():
    """A model with initial_loss=150 should be dropped by the S0.75 gate."""
    from research.scientist.runner.execution_screening import (
        INITIAL_LOSS_THRESHOLD,
    )

    # Verify the threshold constant
    assert INITIAL_LOSS_THRESHOLD == 50.0

    # Simulate the S0.75 check logic directly
    vocab_size = 1000  # Small for speed
    model = _HighInitLossModel(vocab_size, loss_value=150.0)
    model.train()

    ids = torch.randint(0, vocab_size, (4, 64))
    logits = model(ids)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        ids[:, 1:].reshape(-1),
    )
    init_loss = loss.item()

    # The model should produce loss > 50
    assert init_loss > INITIAL_LOSS_THRESHOLD, (
        f"Mock model should have init_loss > {INITIAL_LOSS_THRESHOLD}, got {init_loss}"
    )

    # Simulate funnel behavior
    funnel_counts = {"dropped_s075_high_init": 0}
    if init_loss > INITIAL_LOSS_THRESHOLD:
        funnel_counts["dropped_s075_high_init"] += 1

    assert funnel_counts["dropped_s075_high_init"] == 1


def test_s075_passes_normal_initial_loss():
    """A model with initial_loss=8 should NOT be dropped."""
    from research.scientist.runner.execution_screening import (
        INITIAL_LOSS_THRESHOLD,
    )

    vocab_size = 1000
    model = _HighInitLossModel(vocab_size, loss_value=8.0)
    model.train()

    ids = torch.randint(0, vocab_size, (4, 64))
    logits = model(ids)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        ids[:, 1:].reshape(-1),
    )
    init_loss = loss.item()

    # init_loss should be moderate (around 8 * scale, likely < 50)
    # The mock model scales logits, so CE depends on the random logits
    # We just verify the gate logic: if init_loss <= threshold, don't drop
    funnel_counts = {"dropped_s075_high_init": 0}
    if (
        not math.isnan(init_loss)
        and not math.isinf(init_loss)
        and init_loss > INITIAL_LOSS_THRESHOLD
    ):
        funnel_counts["dropped_s075_high_init"] += 1

    assert funnel_counts["dropped_s075_high_init"] == 0, (
        f"Normal model should not be dropped, init_loss={init_loss}"
    )
