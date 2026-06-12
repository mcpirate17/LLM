"""Structural equivalence tests for the vectorized binding task generators.

Each preset must reproduce the legacy per-example generators' token-layout
semantics exactly (special tokens, position distributions, answer mapping).
RNG draw order is allowed to differ; layout invariants are not.
"""

from __future__ import annotations

import torch

from component_fab.harness.binding_taskgen import (
    generate_retention_batch,
    pair_positions,
    reserved_offsets,
    sample_unique_rows,
)
from component_fab.harness.binding_validity import (
    DEFAULT_BINDING_VALIDITY_TASKS,
    HARD_BINDING_VALIDITY_TASKS,
    generate_binding_validity_batch,
)
from component_fab.harness.adversarial_retention import RetentionTask
from component_fab.harness.harder_binding_tasks import (
    _BATCH_GENERATORS,
    HardBindingTask,
    default_hard_binding_tasks,
)
from component_fab.harness.probe_tasks import _causal_induction


def _task(name: str) -> HardBindingTask:
    return next(t for t in default_hard_binding_tasks(seed=0) if t.name == name)


def _rng(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_sample_unique_rows_is_unique_per_row() -> None:
    rows = sample_unique_rows(16, 12, 64, _rng(0))
    for row in rows:
        assert len(set(row.tolist())) == 12


def test_hard_kv_layout_queries_and_first_match_answers() -> None:
    task = _task("multi_query_kv_recall")
    res = reserved_offsets(task.vocab_size)
    ids, qpos, tgt = _BATCH_GENERATORS[task.name](task, 64, False, _rng(1))
    assert ids.shape == (64, task.seq_len)
    assert qpos.shape == tgt.shape == (64, task.n_queries)
    n = task.n_pairs_in_seq
    for b in range(64):
        keys = ids[b, 0 : 2 * n : 2].tolist()
        values = ids[b, 1 : 2 * n : 2].tolist()
        assert all(0 <= k < task.n_keys for k in keys)
        assert all(task.n_keys <= v < task.n_keys + task.n_values for v in values)
        for qi in range(task.n_queries):
            p = int(qpos[b, qi])
            assert int(ids[b, p]) == res["ANS"]
            assert int(ids[b, p - 2]) == res["QUERY"]
            queried = int(ids[b, p - 1])
            assert queried in keys  # query points at a real written key
            first = keys.index(queried)
            assert int(tgt[b, qi]) == values[first]  # FIRST bound value wins
        # Ordered query cycle: query qi echoes pair qi % n.
        assert [int(ids[b, int(qpos[b, qi]) - 1]) for qi in range(task.n_queries)] == [
            keys[qi % n] for qi in range(task.n_queries)
        ]


def test_hard_kv_pairs_come_from_the_right_split() -> None:
    task = _task("heldout_pair_recall")
    train_pool = set(task.train_pairs)
    eval_pool = set(task.eval_pairs)
    n = task.n_pairs_in_seq
    for eval_split, pool in ((False, train_pool), (True, eval_pool)):
        ids, _, _ = _BATCH_GENERATORS[task.name](task, 32, eval_split, _rng(2))
        for b in range(32):
            written = {(int(ids[b, 2 * j]), int(ids[b, 2 * j + 1])) for j in range(n)}
            assert written <= pool


def test_hard_distractors_share_key_but_never_true_value() -> None:
    task = _task("distractor_kv_recall")
    n = task.n_pairs_in_seq
    d = task.distractors_per_key
    ids, _, _ = _BATCH_GENERATORS[task.name](task, 64, False, _rng(3))
    for b in range(64):
        keys = ids[b, 0 : 2 * n : 2].tolist()
        values = ids[b, 1 : 2 * n : 2].tolist()
        base = 2 * n
        for j in range(n):
            for r in range(d):
                offset = base + 2 * (j * d + r)
                assert int(ids[b, offset]) == keys[j]
                assert int(ids[b, offset + 1]) != values[j]


def test_hard_long_gap_window_and_answer() -> None:
    task = _task("long_gap_recall")
    res = reserved_offsets(task.vocab_size)
    ids, qpos, tgt = _BATCH_GENERATORS[task.name](task, 32, False, _rng(4))
    for b in range(32):
        p = int(qpos[b, 0])
        assert p >= 2 + task.long_gap_min
        assert p <= min(2 + task.long_gap_max, task.seq_len - 4) + 2
        assert int(ids[b, p]) == res["ANS"]
        assert int(ids[b, p - 1]) == int(ids[b, 0])  # query echoes the key
        assert int(tgt[b, 0]) == int(ids[b, 1])  # answer is the bound value
        gap_region = ids[b, 2 : p - 2]
        assert (gap_region == res["NOISE"]).all()


def test_hard_variable_layout_reorders_queries() -> None:
    task = _task("variable_layout_recall")
    n = task.n_pairs_in_seq
    ids, qpos, _ = _BATCH_GENERATORS[task.name](task, 64, False, _rng(5))
    reordered = 0
    for b in range(64):
        keys = ids[b, 0 : 2 * n : 2].tolist()
        queried = [int(ids[b, int(qpos[b, qi]) - 1]) for qi in range(task.n_queries)]
        assert sorted(queried) == sorted(keys)  # a full permutation cycle
        if queried != keys:
            reordered += 1
    assert reordered > 0


def test_hard_compositional_triples_and_ordered_queries() -> None:
    task = _task("compositional_binding")
    res = reserved_offsets(task.vocab_size)
    n = task.n_pairs_in_seq
    ids, qpos, tgt = _BATCH_GENERATORS[task.name](task, 32, False, _rng(6))
    for b in range(32):
        triples = [
            (int(ids[b, 3 * j]), int(ids[b, 3 * j + 1]), int(ids[b, 3 * j + 2]))
            for j in range(n)
        ]
        for e, a, v in triples:
            assert 0 <= e < task.n_entities
            assert task.n_entities <= a < task.n_entities + task.n_attributes
        for qi in range(task.n_queries):
            p = int(qpos[b, qi])
            assert int(ids[b, p]) == res["ANS"]
            e, a, v = triples[qi % n]
            assert int(ids[b, p - 2]) == e
            assert int(ids[b, p - 1]) == a
            assert int(tgt[b, qi]) == v


def test_validity_scattered_writes_sit_on_the_item_grid() -> None:
    task = next(
        t
        for t in HARD_BINDING_VALIDITY_TASKS
        if t.name == "hard_variable_layout_12_pairs_6_queries_128"
    )
    ids, _, _ = generate_binding_validity_batch(task, 16, _rng(7))
    write_region_end = task.seq_len - task.n_queries * 4
    for b in range(16):
        key_positions = [
            p for p in range(write_region_end) if int(ids[b, p]) < task.n_keys
        ]
        assert len(key_positions) == task.n_pairs
        assert all(p % 2 == 0 for p in key_positions)
        for p in key_positions:  # each key is followed by its value token
            value = int(ids[b, p + 1])
            assert task.value_start <= value < task.value_start + task.n_values


def test_validity_flat_values_are_episodic() -> None:
    task = DEFAULT_BINDING_VALIDITY_TASKS[0]
    ids, _, _ = generate_binding_validity_batch(task, 128, _rng(8))
    values_by_key: dict[int, set[int]] = {}
    for b in range(128):
        for j in range(task.n_pairs):
            values_by_key.setdefault(int(ids[b, 2 * j]), set()).add(
                int(ids[b, 2 * j + 1])
            )
    assert any(len(vals) > 1 for vals in values_by_key.values())


def test_retention_layout_matches_pair_positions_and_earliest_query() -> None:
    task = RetentionTask("episodic", 64, 64, 4, "random")
    ids, qpos, tgt = generate_retention_batch(task, 16, 64, _rng(9))
    starts = pair_positions(64, 4)
    assert qpos.tolist() == [[63]] * 16
    for b in range(16):
        keys = [int(ids[b, s]) for s in starts]
        values = [int(ids[b, s + 1]) for s in starts]
        assert len(set(keys)) == 4  # unique keys per episode
        assert len(set(values)) == 4  # unique values per episode
        assert int(ids[b, -2]) == keys[0]  # query echoes the earliest key
        assert int(tgt[b, 0]) == values[0]  # answer is the earliest value
        assert int(ids[b, -3]) == task.vocab_size - 3
        assert int(ids[b, -1]) == task.vocab_size - 2


def test_causal_induction_matches_reference_loop() -> None:
    torch.manual_seed(10)
    x = torch.randn(4, 24, 8)

    def reference(x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        keys = (x[..., 0] > 0).float()
        out = x.clone()
        last = torch.full((batch_size,), -1, dtype=torch.long)
        for i in range(seq_len):
            active = last >= 0
            if active.any():
                idx = last.clamp_min(0).view(-1, 1, 1).expand(-1, 1, dim)
                out[active, i] = torch.gather(x, 1, idx).squeeze(1)[active]
            update = keys[:, i].bool()
            last = torch.where(update, torch.tensor(i), last)
        return out

    assert torch.allclose(_causal_induction(x), reference(x))
