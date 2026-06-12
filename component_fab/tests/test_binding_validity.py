"""Tests for validity-controlled episodic binding tasks."""

from __future__ import annotations

import pytest
import torch

from component_fab.harness.binding_validity import (
    DEFAULT_BINDING_VALIDITY_TASKS,
    HARD_BINDING_VALIDITY_TASKS,
    BindingValidityTask,
    audit_flat_writes,
    binding_validity_load_ladder,
    generate_binding_validity_batch,
    run_binding_validity_task,
)
from component_fab.harness.harder_binding_tasks import (
    _BATCH_GENERATORS,
    default_hard_binding_tasks,
)
from component_fab.harness.tiny_lm import SoftmaxCausalAttention


def _task(name: str) -> BindingValidityTask:
    return next(task for task in DEFAULT_BINDING_VALIDITY_TASKS if task.name == name)


def test_legacy_multi_query_contains_conflicting_duplicate_keys() -> None:
    legacy = next(
        task
        for task in default_hard_binding_tasks(seed=0)
        if task.name == "multi_query_kv_recall"
    )
    ids, _, _ = _BATCH_GENERATORS[legacy.name](
        legacy, 512, False, torch.Generator().manual_seed(0)
    )
    audit = audit_flat_writes(
        ids, n_write_pairs=legacy.n_pairs_in_seq, key_upper_bound=legacy.n_keys
    )
    assert audit.duplicate_key_rate > 0.4
    assert audit.conflicting_value_rate > 0.3


def test_unique_multi_query_has_one_value_per_key() -> None:
    task = _task("episodic_unique_multi_query")
    ids, positions, targets = generate_binding_validity_batch(
        task, 64, torch.Generator().manual_seed(1)
    )
    audit = audit_flat_writes(
        ids, n_write_pairs=task.n_pairs, key_upper_bound=task.n_keys
    )
    assert audit.duplicate_key_rate == 0.0
    assert audit.conflicting_value_rate == 0.0
    for batch_index in range(ids.shape[0]):
        for query_index in range(task.n_queries):
            answer_position = int(positions[batch_index, query_index].item())
            key = int(ids[batch_index, answer_position - 1].item())
            matching = [
                int(ids[batch_index, 2 * pair_index + 1].item())
                for pair_index in range(task.n_pairs)
                if int(ids[batch_index, 2 * pair_index].item()) == key
            ]
            assert matching == [int(targets[batch_index, query_index].item())]


def test_queries_are_not_fixed_to_first_written_pairs() -> None:
    task = _task("episodic_unique_multi_query")
    ids, positions, _ = generate_binding_validity_batch(
        task, 128, torch.Generator().manual_seed(11)
    )
    found_non_prefix_query = False
    found_reordered_query = False
    for batch_index in range(ids.shape[0]):
        written_keys = [
            int(ids[batch_index, 2 * pair_index].item())
            for pair_index in range(task.n_pairs)
        ]
        queried_keys = [
            int(ids[batch_index, int(position.item()) - 1].item())
            for position in positions[batch_index]
        ]
        if any(key not in written_keys[: task.n_queries] for key in queried_keys):
            found_non_prefix_query = True
        if queried_keys != written_keys[: task.n_queries]:
            found_reordered_query = True
    assert found_non_prefix_query
    assert found_reordered_query


def test_same_key_overwrite_targets_latest_value() -> None:
    task = _task("episodic_same_key_overwrite")
    ids, positions, targets = generate_binding_validity_batch(
        task, 32, torch.Generator().manual_seed(2)
    )
    for batch_index in range(ids.shape[0]):
        assert ids[batch_index, 0] == ids[batch_index, 2]
        assert ids[batch_index, 1] != ids[batch_index, 3]
        assert targets[batch_index, 0] == ids[batch_index, 3]
        answer_position = int(positions[batch_index, 0].item())
        assert ids[batch_index, answer_position - 1] == ids[batch_index, 0]


def test_compositional_values_change_across_episodes() -> None:
    task = _task("episodic_compositional")
    ids, _, _ = generate_binding_validity_batch(
        task, 128, torch.Generator().manual_seed(3)
    )
    values_by_pair: dict[tuple[int, int], set[int]] = {}
    for row in ids:
        for pair_index in range(task.n_pairs):
            offset = 3 * pair_index
            pair = (int(row[offset].item()), int(row[offset + 1].item()))
            values_by_pair.setdefault(pair, set()).add(int(row[offset + 2].item()))
    assert any(len(values) > 1 for values in values_by_pair.values())


def test_compositional_queries_are_independent_of_write_order() -> None:
    task = _task("episodic_compositional")
    ids, positions, _ = generate_binding_validity_batch(
        task, 128, torch.Generator().manual_seed(13)
    )
    found_non_prefix_query = False
    for batch_index in range(ids.shape[0]):
        written_pairs = [
            (
                int(ids[batch_index, 3 * pair_index].item()),
                int(ids[batch_index, 3 * pair_index + 1].item()),
            )
            for pair_index in range(task.n_pairs)
        ]
        queried_pairs = [
            (
                int(ids[batch_index, int(position.item()) - 2].item()),
                int(ids[batch_index, int(position.item()) - 1].item()),
            )
            for position in positions[batch_index]
        ]
        if any(pair not in written_pairs[: task.n_queries] for pair in queried_pairs):
            found_non_prefix_query = True
            break
    assert found_non_prefix_query


def test_hard_scattered_layout_has_non_write_tokens_between_pairs() -> None:
    task = next(
        task
        for task in HARD_BINDING_VALIDITY_TASKS
        if task.name == "hard_variable_layout_12_pairs_6_queries_128"
    )
    ids, positions, _ = generate_binding_validity_batch(
        task, 16, torch.Generator().manual_seed(17)
    )
    query_start = int(positions[:, 0].min().item()) - 2
    noise = task.special["NOISE"]
    assert (ids[:, :query_start] == noise).any()
    assert all(
        len(set(int(position.item()) for position in row)) == task.n_queries
        for row in positions
    )


def test_load_ladder_increases_pairs_and_length() -> None:
    ladder = binding_validity_load_ladder()
    assert [task.n_pairs for task in ladder] == [4, 8, 16, 32]
    assert [task.seq_len for task in ladder] == [64, 128, 256, 512]
    for task in ladder:
        ids, positions, targets = generate_binding_validity_batch(
            task, 2, torch.Generator().manual_seed(task.n_pairs)
        )
        assert ids.shape == (2, task.seq_len)
        assert positions.shape == targets.shape == (2, task.n_queries)
        assert int(positions.max().item()) < task.seq_len


def test_task_rejects_too_many_unique_keys() -> None:
    with pytest.raises(ValueError, match="distinct keys"):
        BindingValidityTask(
            "bad", "unique_multi_query", n_keys=4, n_pairs=5, n_queries=1
        )


def test_task_rejects_sequence_too_short_for_writes_and_queries() -> None:
    with pytest.raises(ValueError, match="too short"):
        BindingValidityTask(
            "bad_capacity",
            "episodic_compositional",
            seq_len=24,
            n_pairs=6,
            n_queries=2,
        )


def test_binding_validity_training_smoke() -> None:
    result = run_binding_validity_task(
        SoftmaxCausalAttention,
        BindingValidityTask(
            "smoke",
            "unique_multi_query",
            seq_len=24,
            n_keys=4,
            n_values=4,
            n_pairs=2,
            n_queries=1,
        ),
        mixer_label="softmax",
        dim=8,
        n_blocks=1,
        n_train_steps=2,
        batch_size=2,
        n_eval_batches=1,
    )
    assert result.converged
    assert 0.0 <= result.eval_accuracy <= 1.0
    assert result.chance_accuracy == 0.25
