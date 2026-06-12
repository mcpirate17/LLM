"""Tests for the adversarial episodic retention battery."""

from __future__ import annotations

import pytest
import torch

from component_fab.harness.adversarial_retention import (
    RetentionTask,
    _pair_positions,
    generate_retention_batch,
    run_retention_task,
)
from component_fab.harness.tiny_lm import SoftmaxCausalAttention


def test_retention_task_rejects_impossible_pair_count() -> None:
    with pytest.raises(ValueError, match="distinct keys and values"):
        RetentionTask("bad", 32, 32, 9, "random")


def test_pair_positions_are_distinct_and_before_query() -> None:
    positions = _pair_positions(64, 8)
    assert positions[0] == 0
    assert len(set(positions)) == 8
    assert all(position + 1 < 61 for position in positions)


def test_batch_is_episodic_and_queries_earliest_pair() -> None:
    task = RetentionTask("episodic", 64, 64, 4, "random")
    ids, query_positions, targets = generate_retention_batch(
        task, 8, 64, torch.Generator().manual_seed(0)
    )
    assert ids.shape == (8, 64)
    assert query_positions.tolist() == [[63]] * 8
    for batch_index in range(8):
        assert ids[batch_index, -2] == ids[batch_index, 0]
        assert targets[batch_index, 0] == ids[batch_index, 1]
    observed = {(int(ids[b, 0]), int(ids[b, 1])) for b in range(8)}
    assert len(observed) > 1


def test_random_filler_varies_and_avoids_key_value_ranges() -> None:
    task = RetentionTask("random", 64, 64, 1, "random")
    ids, _, _ = generate_retention_batch(task, 4, 64, torch.Generator().manual_seed(1))
    filler = ids[:, 2:-3]
    assert filler.unique().numel() > 1
    assert int(filler.min()) >= task.n_keys + task.n_values
    assert int(filler.max()) < task.n_keys + task.n_values + task.n_fillers


def test_length_extrapolation_uses_distinct_train_eval_lengths() -> None:
    task = RetentionTask("extrapolate", 32, 64, 1, "random")
    generator = torch.Generator().manual_seed(2)
    train_ids, _, _ = generate_retention_batch(task, 2, 32, generator)
    eval_ids, _, _ = generate_retention_batch(task, 2, 64, generator)
    assert train_ids.shape[1] == 32
    assert eval_ids.shape[1] == 64


def test_retention_training_smoke() -> None:
    task = RetentionTask("smoke", 16, 16, 1, "constant")
    result = run_retention_task(
        SoftmaxCausalAttention,
        task,
        mixer_label="softmax",
        dim=8,
        n_blocks=1,
        n_train_steps=2,
        batch_size=2,
        n_eval_batches=1,
        seed=0,
    )
    assert 0.0 <= result.eval_accuracy <= 1.0
    assert result.chance_accuracy == pytest.approx(0.125)
    assert result.n_params > 0
