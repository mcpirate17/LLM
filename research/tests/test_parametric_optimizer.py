"""The optimizer family must train, be gradeable, and slide AdamW<->Lion."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from research.synthesis.parametric_optimizer import (
    ParametricOptimizer,
    UpdateSpec,
    grade_optimizer,
)


def test_default_is_adamw_and_trains() -> None:
    score = grade_optimizer(UpdateSpec(), steps=80, seed=0)
    assert score > 0.3  # default AdamW makes real progress on the nano problem


def test_lion_blend_also_trains() -> None:
    score = grade_optimizer(UpdateSpec(mix=1.0, log_lr=math.log(3e-3)), steps=80)
    assert score > 0.0


def test_grade_is_deterministic() -> None:
    a = grade_optimizer(UpdateSpec(), steps=40, seed=2)
    b = grade_optimizer(UpdateSpec(), steps=40, seed=2)
    assert a == b


def test_divergent_optimizer_scored_low_not_raised() -> None:
    # An absurd learning rate diverges; the harness reports the low score.
    score = grade_optimizer(UpdateSpec(log_lr=math.log(1e6)), steps=40)
    assert score <= 0.0


def test_lion_step_is_sign_magnitude() -> None:
    p = nn.Parameter(torch.randn(8, 8))
    p.grad = torch.randn(8, 8)
    opt = ParametricOptimizer(
        [p], UpdateSpec(mix=1.0, beta1=0.0, beta3=0.0, log_lr=math.log(1e-2))
    )
    before = p.detach().clone()
    opt.step()
    delta = (p.detach() - before).abs()
    # sign(grad) update -> every coordinate moves by exactly lr.
    assert torch.allclose(delta, torch.full_like(delta, 1e-2), atol=1e-6)


def test_spec_validation_fails_loud() -> None:
    with pytest.raises(ValueError, match="mix"):
        UpdateSpec(mix=1.5)


def test_keys_name_the_family() -> None:
    assert UpdateSpec(mix=0.0).key.startswith("adamw")
    assert UpdateSpec(mix=1.0).key.startswith("lion")
    assert UpdateSpec(mix=0.5).key.startswith("blend")
