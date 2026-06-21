"""Regression coverage for shared probe training guardrails."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from component_fab.harness.training_probe import train_lane_head


def test_train_lane_head_reports_non_finite_predictions() -> None:
    weight = nn.Parameter(torch.ones(1))

    def forward(x: torch.Tensor) -> torch.Tensor:
        # Keep the parameter in the graph so the optimizer path is realistic,
        # but force the failure to originate in the forward output.
        return x * weight * float("nan")

    def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ones(4, 1), torch.zeros(4, dtype=torch.long)

    def loss_fn(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        del target
        return predictions.mean()

    with pytest.raises(FloatingPointError, match="non-finite predictions at step 1"):
        train_lane_head(
            forward,
            [weight],
            sample_batch,
            loss_fn,
            n_train_steps=1,
            max_grad_norm=None,
        )


def test_train_lane_head_reports_non_finite_gradients() -> None:
    weight = nn.Parameter(torch.ones(1))

    def forward(x: torch.Tensor) -> torch.Tensor:
        return x * weight

    def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ones(4, 1), torch.zeros(4, dtype=torch.long)

    def loss_fn(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        del predictions, target
        # Forward output and scalar loss are finite, but backward produces an
        # infinite gradient. This catches the failure before optimizer.step().
        return weight.sum() * float("inf")

    with pytest.raises(FloatingPointError, match="non-finite loss at step 1"):
        train_lane_head(
            forward,
            [weight],
            sample_batch,
            loss_fn,
            n_train_steps=1,
            max_grad_norm=None,
        )


def test_train_lane_head_rejects_empty_parameter_iterable() -> None:
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x

    def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ones(4, 1), torch.zeros(4, dtype=torch.long)

    def loss_fn(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        del target
        return predictions.mean()

    with pytest.raises(ValueError, match="at least one trainable parameter"):
        train_lane_head(forward, [], sample_batch, loss_fn, n_train_steps=1)
