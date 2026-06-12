"""Batched, torch-vectorized KV-binding batch generation.

One generator core replaces the six per-example Python loops that lived in
``harder_binding_tasks`` (basic / long-gap / compositional),
``binding_validity`` (flat / compositional) and ``adversarial_retention``.
Every legacy task name keeps its exact token-layout semantics (special-token
convention per family, position distribution, answer mapping); only the RNG
draw order differs (vectorized sampling).

Parameter axes covered by the presets:

- pair layout: contiguous (``hard_*``, validity non-scatter), scattered
  (validity ``scatter_writes``), fixed positions (retention).
- query selection: ordered cycle (``multi_query_kv_recall``), shuffled cycle
  (``variable_layout_recall``), random subset (validity), first pair
  (retention), latest write (``same_key_overwrite``).
- key reuse: with replacement (hard family, duplicate keys possible —
  answer maps to the FIRST bound value), unique per example (validity,
  retention), explicit overwrite (answer maps to the LATEST value).
- value pool: global train/eval split pools (hard family) vs episodic
  per-example resampling (validity, retention).
- write arity: kv pairs vs entity-attribute-value triples.
- gap filler: PAD background, NOISE regions, constant / random filler vocab.

All helpers are vectorized (argsort-of-random for per-row permutations,
``scatter_`` for block placement) — no per-example Python loops.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .adversarial_retention import RetentionTask
    from .binding_validity import BindingValidityTask
    from .harder_binding_tasks import HardBindingTask

TokenBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# Special token ids reserved at the END of vocab for the hard-binding family.
RESERVED_TOKENS = 4  # PAD, QUERY, ANS, NOISE


def reserved_offsets(vocab_size: int) -> dict[str, int]:
    base = vocab_size - RESERVED_TOKENS
    return {"PAD": base, "QUERY": base + 1, "ANS": base + 2, "NOISE": base + 3}


# ---------- Vectorized building blocks ----------


def rows_randperm(batch_size: int, n: int, generator: torch.Generator) -> torch.Tensor:
    """One independent permutation of ``range(n)`` per row -> ``[B, n]``."""
    return torch.argsort(torch.rand(batch_size, n, generator=generator), dim=1)


def sample_unique_rows(
    n_options: int, take: int, batch_size: int, generator: torch.Generator
) -> torch.Tensor:
    """Per-row sample of ``take`` distinct indices from ``range(n_options)``."""
    if take > n_options:
        raise ValueError(f"cannot take {take} unique items from {n_options}")
    return rows_randperm(batch_size, n_options, generator)[:, :take]


def _scatter_block(
    ids: torch.Tensor, starts: torch.Tensor, tokens: torch.Tensor
) -> None:
    """Write ``tokens[B, K, W]`` at row positions ``starts[B, K] + arange(W)``."""
    b, k, w = tokens.shape
    positions = starts.unsqueeze(-1) + torch.arange(w)
    if int(positions.max()) >= ids.shape[1]:
        raise ValueError(
            f"write block exceeds seq_len {ids.shape[1]}"
            f" (max position {int(positions.max())})"
        )
    ids.scatter_(1, positions.reshape(b, k * w), tokens.reshape(b, k * w))


def _scattered_starts_rows(
    *,
    region_end: int,
    item_width: int,
    n_items: int,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Per-row sorted starts on the ``item_width``-stride slot grid."""
    n_slots = max(0, (region_end - item_width) // item_width + 1)
    if n_slots < n_items:
        raise ValueError(
            f"write region has {n_slots} slots, but {n_items} are required"
        )
    chosen = sample_unique_rows(n_slots, n_items, batch_size, generator)
    return chosen.sort(dim=1).values * item_width


def _lay_queries(
    ids: torch.Tensor,
    *,
    q_starts: torch.Tensor,
    key_tokens: torch.Tensor,
    query_token: int,
    answer_token: int,
    stride: int,
) -> torch.Tensor:
    """Write ``[QUERY, key..., ANS]`` blocks every ``stride`` positions.

    ``q_starts[B]`` is the per-row position of the first query;
    ``key_tokens[B, Q, T]`` are the echoed key tokens. Returns the ANS
    (answer/readout) positions ``[B, Q]``.
    """
    b, n_queries, t = key_tokens.shape
    starts = q_starts.unsqueeze(1) + stride * torch.arange(n_queries)
    block = torch.empty((b, n_queries, t + 2), dtype=torch.long)
    block[:, :, 0] = query_token
    block[:, :, 1 : 1 + t] = key_tokens
    block[:, :, 1 + t] = answer_token
    _scatter_block(ids, starts, block)
    return starts + 1 + t


# ---------- Hard-binding family (harder_binding_tasks token convention) ----------


def _sample_pool_rows(
    pool: tuple[tuple[int, int], ...],
    batch_size: int,
    n: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``n`` pool entries per row WITH replacement -> two ``[B, n]``."""
    pool_t = torch.tensor(pool, dtype=torch.long)
    idx = torch.randint(0, len(pool), (batch_size, n), generator=generator)
    return pool_t[idx, 0], pool_t[idx, 1]


def generate_hard_kv_batch(
    task: HardBindingTask,
    batch_size: int,
    eval_split: bool,
    rng: torch.Generator,
) -> TokenBatch:
    """Multi-query KV recall (plus distractor / gap / variable-layout variants).

    Layout: ``[k1, v1, ..., kN, vN, (distractors), (noise gap), QUERY, q1,
    ANS, _, ...]``. Keys are sampled with replacement, so duplicate keys can
    bind conflicting values — the answer is the FIRST bound value (legacy
    semantics, audited by ``binding_validity.audit_flat_writes``).
    """
    pool = task.eval_pairs if eval_split else task.train_pairs
    res = reserved_offsets(task.vocab_size)
    n_pairs = task.n_pairs_in_seq
    keys, values = _sample_pool_rows(pool, batch_size, n_pairs, rng)
    ids = torch.full((batch_size, task.seq_len), res["PAD"], dtype=torch.long)
    pair_starts = (2 * torch.arange(n_pairs)).expand(batch_size, n_pairs)
    _scatter_block(ids, pair_starts, torch.stack([keys, values], dim=-1))
    cursor = 2 * n_pairs

    if task.distractors_per_key > 0:
        n_distractors = n_pairs * task.distractors_per_key
        d_keys = keys.repeat_interleave(task.distractors_per_key, dim=1)
        true_value_idx = (
            values.repeat_interleave(task.distractors_per_key, dim=1) - task.n_keys
        )
        draw = torch.randint(
            0, task.n_values, (batch_size, n_distractors), generator=rng
        )
        draw = torch.where(draw == true_value_idx, (draw + 1) % task.n_values, draw)
        d_starts = (cursor + 2 * torch.arange(n_distractors)).expand(
            batch_size, n_distractors
        )
        _scatter_block(ids, d_starts, torch.stack([d_keys, task.n_keys + draw], dim=-1))
        cursor += 2 * n_distractors

    if task.long_gap_min > 0:
        gap_target = torch.randint(
            task.long_gap_min, task.long_gap_max + 1, (batch_size,), generator=rng
        )
        q_starts = gap_target.clamp(max=task.seq_len - 4).clamp(min=cursor)
        span = torch.arange(task.seq_len).unsqueeze(0)
        noise_mask = (span >= cursor) & (span < q_starts.unsqueeze(1))
        ids[noise_mask] = res["NOISE"]
    else:
        q_starts = torch.full((batch_size,), cursor, dtype=torch.long)

    if task.variable_layout:
        order = rows_randperm(batch_size, n_pairs, rng)
        repeats = -(-task.n_queries // n_pairs)
        query_idx = order.repeat(1, repeats)[:, : task.n_queries]
    else:
        query_idx = (torch.arange(task.n_queries) % n_pairs).expand(
            batch_size, task.n_queries
        )
    query_keys = keys.gather(1, query_idx)
    # Answer = value of the FIRST pair carrying the queried key.
    match = keys.unsqueeze(1) == query_keys.unsqueeze(2)
    pair_index = torch.arange(n_pairs).expand(batch_size, task.n_queries, n_pairs)
    first_match = torch.where(match, pair_index, n_pairs).min(dim=2).values
    targets = values.gather(1, first_match)

    positions = _lay_queries(
        ids,
        q_starts=q_starts,
        key_tokens=query_keys.unsqueeze(-1),
        query_token=res["QUERY"],
        answer_token=res["ANS"],
        stride=4,
    )
    return ids, positions, targets


def generate_hard_long_gap_batch(
    task: HardBindingTask,
    batch_size: int,
    eval_split: bool,
    rng: torch.Generator,
) -> TokenBatch:
    """One (k,v), noise filler of length in [long_gap_min, long_gap_max],
    then ``QUERY k ANS ?``. Pure long-context binding test.
    """
    pool = task.eval_pairs if eval_split else task.train_pairs
    res = reserved_offsets(task.vocab_size)
    keys, values = _sample_pool_rows(pool, batch_size, 1, rng)
    ids = torch.full((batch_size, task.seq_len), res["NOISE"], dtype=torch.long)
    ids[:, 0] = keys[:, 0]
    ids[:, 1] = values[:, 0]
    gap = torch.randint(
        task.long_gap_min, task.long_gap_max + 1, (batch_size,), generator=rng
    )
    q_starts = (2 + gap).clamp(max=task.seq_len - 4)
    positions = _lay_queries(
        ids,
        q_starts=q_starts,
        key_tokens=keys.unsqueeze(-1),
        query_token=res["QUERY"],
        answer_token=res["ANS"],
        stride=4,
    )
    return ids, positions, values


def generate_hard_compositional_batch(
    task: HardBindingTask,
    batch_size: int,
    eval_split: bool,
    rng: torch.Generator,
) -> TokenBatch:
    """(entity, attribute) -> value with disjoint train/eval (e,a) combos."""
    pool = task.eval_pairs if eval_split else task.train_pairs
    res = reserved_offsets(task.vocab_size)
    n_triples = task.n_pairs_in_seq
    encoded_ea, values = _sample_pool_rows(pool, batch_size, n_triples, rng)
    entities = encoded_ea // task.n_attributes
    attributes = task.n_entities + encoded_ea % task.n_attributes
    ids = torch.full((batch_size, task.seq_len), res["PAD"], dtype=torch.long)
    starts = (3 * torch.arange(n_triples)).expand(batch_size, n_triples)
    _scatter_block(ids, starts, torch.stack([entities, attributes, values], dim=-1))

    query_idx = (torch.arange(task.n_queries) % n_triples).expand(
        batch_size, task.n_queries
    )
    key_tokens = torch.stack(
        [entities.gather(1, query_idx), attributes.gather(1, query_idx)], dim=-1
    )
    q_starts = torch.full((batch_size,), 3 * n_triples, dtype=torch.long)
    positions = _lay_queries(
        ids,
        q_starts=q_starts,
        key_tokens=key_tokens,
        query_token=res["QUERY"],
        answer_token=res["ANS"],
        stride=4,
    )
    return ids, positions, values.gather(1, query_idx)


# ---------- Validity family (binding_validity token convention) ----------


def _validity_writes(
    task: BindingValidityTask,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sampled write tokens + query selection for the flat validity kinds.

    Returns ``(write_tokens[B, W, 2], query_keys[B, Q], targets[B, Q])``.
    """
    if task.kind == "same_key_overwrite":
        keys = sample_unique_rows(task.n_keys, task.n_pairs, batch_size, generator)
        value_idx = torch.randint(
            0, task.n_values, (batch_size, task.n_pairs), generator=generator
        )
        key = keys[:, 0]
        first = value_idx[:, 0]
        offset = 1 + value_idx[:, 1] % (task.n_values - 1)
        latest = (first + offset) % task.n_values
        write_keys = torch.stack([key, key], dim=1)
        write_values = task.value_start + torch.stack([first, latest], dim=1)
        return (
            torch.stack([write_keys, write_values], dim=-1),
            key.unsqueeze(1),
            (task.value_start + latest).unsqueeze(1),
        )
    keys = sample_unique_rows(task.n_keys, task.n_pairs, batch_size, generator)
    value_idx = torch.randint(
        0, task.n_values, (batch_size, task.n_pairs), generator=generator
    )
    write_values = task.value_start + value_idx
    selected = sample_unique_rows(task.n_pairs, task.n_queries, batch_size, generator)
    return (
        torch.stack([keys, write_values], dim=-1),
        keys.gather(1, selected),
        write_values.gather(1, selected),
    )


def _place_validity_writes(
    ids: torch.Tensor,
    task: BindingValidityTask,
    write_tokens: torch.Tensor,
    generator: torch.Generator,
    *,
    query_width: int,
) -> int:
    """Scatter or contiguously place writes; returns the query-region start."""
    batch_size, n_writes, arity = write_tokens.shape
    special = task.special
    if task.scatter_writes:
        write_region_end = task.seq_len - task.n_queries * query_width
        ids[:, :write_region_end] = special["NOISE"]
        starts = _scattered_starts_rows(
            region_end=write_region_end,
            item_width=arity,
            n_items=n_writes,
            batch_size=batch_size,
            generator=generator,
        )
        cursor = write_region_end
    else:
        starts = (arity * torch.arange(n_writes)).expand(batch_size, n_writes)
        cursor = arity * n_writes
    _scatter_block(ids, starts, write_tokens)
    if task.kind == "distinct_key_interference" and cursor < task.seq_len // 2:
        ids[:, cursor : task.seq_len // 2] = special["NOISE"]
        cursor = task.seq_len // 2
    return cursor


def generate_validity_flat_batch(
    task: BindingValidityTask,
    batch_size: int,
    generator: torch.Generator,
) -> TokenBatch:
    """Flat ``[key, value]`` validity kinds (unique / interference / overwrite)."""
    special = task.special
    ids = torch.full((batch_size, task.seq_len), special["PAD"], dtype=torch.long)
    write_tokens, query_keys, query_targets = _validity_writes(
        task, batch_size, generator
    )
    cursor = _place_validity_writes(ids, task, write_tokens, generator, query_width=4)
    answer_positions = _lay_queries(
        ids,
        q_starts=torch.full((batch_size,), cursor, dtype=torch.long),
        key_tokens=query_keys.unsqueeze(-1),
        query_token=special["QUERY"],
        answer_token=special["ANS"],
        stride=4,
    )
    # same_key_overwrite emits a single query; pad to [B, n_queries] like legacy.
    positions = torch.zeros((batch_size, task.n_queries), dtype=torch.long)
    targets = torch.zeros((batch_size, task.n_queries), dtype=torch.long)
    positions[:, : answer_positions.shape[1]] = answer_positions
    targets[:, : query_targets.shape[1]] = query_targets
    return ids, positions, targets


def generate_validity_compositional_batch(
    task: BindingValidityTask,
    batch_size: int,
    generator: torch.Generator,
) -> TokenBatch:
    """Episodic (entity, attribute) -> value with per-example value resampling."""
    special = task.special
    n_combinations = task.n_entities * task.n_attributes
    combos = sample_unique_rows(n_combinations, task.n_pairs, batch_size, generator)
    value_idx = torch.randint(
        0, task.n_values, (batch_size, task.n_pairs), generator=generator
    )
    entities = combos // task.n_attributes
    attributes = task.n_entities + combos % task.n_attributes
    values = task.value_start + value_idx
    ids = torch.full((batch_size, task.seq_len), special["PAD"], dtype=torch.long)
    write_tokens = torch.stack([entities, attributes, values], dim=-1)
    cursor = _place_validity_writes(ids, task, write_tokens, generator, query_width=5)
    selected = sample_unique_rows(task.n_pairs, task.n_queries, batch_size, generator)
    key_tokens = torch.stack(
        [entities.gather(1, selected), attributes.gather(1, selected)], dim=-1
    )
    positions = _lay_queries(
        ids,
        q_starts=torch.full((batch_size,), cursor, dtype=torch.long),
        key_tokens=key_tokens,
        query_token=special["QUERY"],
        answer_token=special["ANS"],
        stride=5,
    )
    return ids, positions, values.gather(1, selected)


# ---------- Retention family (adversarial_retention token convention) ----------


def pair_positions(seq_len: int, n_pairs: int) -> tuple[int, ...]:
    """Return pair starts, keeping the target pair at position zero."""
    if n_pairs == 1:
        return (0,)
    last_start = seq_len - 6
    span = last_start - 4
    starts = [0]
    for index in range(n_pairs - 1):
        starts.append(4 + round(index * span / max(1, n_pairs - 2)))
    if len(set(starts)) != len(starts):
        raise ValueError(
            f"sequence length {seq_len} cannot place {n_pairs} distinct pair writes"
        )
    return tuple(starts)


def generate_retention_batch(
    task: RetentionTask,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator,
) -> TokenBatch:
    """Episodic pairs over filler, queried on the earliest pair at the end."""
    if seq_len not in (task.train_seq_len, task.eval_seq_len):
        raise ValueError(
            f"seq_len must be train/eval length, got {seq_len} for {task.name}"
        )
    filler_start = task.n_keys + task.n_values
    query_token = task.vocab_size - 3
    answer_token = task.vocab_size - 2
    if task.filler_mode == "constant":
        ids = torch.full((batch_size, seq_len), filler_start, dtype=torch.long)
    else:
        ids = torch.randint(
            filler_start,
            filler_start + task.n_fillers,
            (batch_size, seq_len),
            generator=generator,
        )
    starts = torch.tensor(pair_positions(seq_len, task.n_pairs), dtype=torch.long)
    keys = sample_unique_rows(task.n_keys, task.n_pairs, batch_size, generator)
    values = sample_unique_rows(task.n_values, task.n_pairs, batch_size, generator)
    ids[:, starts] = keys
    ids[:, starts + 1] = task.n_keys + values
    ids[:, -3] = query_token
    ids[:, -2] = keys[:, 0]
    ids[:, -1] = answer_token
    query_positions = torch.full((batch_size, 1), seq_len - 1, dtype=torch.long)
    return ids, query_positions, (task.n_keys + values[:, 0]).unsqueeze(1)
