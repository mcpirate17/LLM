"""Shared grouped-choice log-prob scoring."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

import torch
import torch.nn as nn


def _pack_choice_sequences(
    flat_sequences: Sequence[Sequence[int]],
    flat_starts: Sequence[int],
    group_sizes: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_seq = len(flat_sequences)
    offsets = torch.empty(n_seq + 1, dtype=torch.long)
    offsets[0] = 0
    total_len = 0
    flat_values: list[int] = []
    extend_values = flat_values.extend
    for idx, seq in enumerate(flat_sequences):
        seq_len = len(seq)
        total_len += seq_len
        offsets[idx + 1] = total_len
        if seq_len:
            if isinstance(seq, np.ndarray):
                extend_values(seq.tolist())
            else:
                extend_values(seq)

    packed = torch.tensor(flat_values, dtype=torch.long)
    starts = torch.as_tensor(flat_starts, dtype=torch.long)
    groups = torch.as_tensor(group_sizes, dtype=torch.long)
    return packed, offsets, starts, groups


def concat_choice_tokens(
    prefix_tokens: Sequence[int] | np.ndarray,
    choice_tokens: Sequence[int] | np.ndarray,
    *,
    max_seq_len: int,
) -> tuple[np.ndarray, int]:
    """Return a clipped ``prefix + choice`` token sequence and scoring start."""
    prefix = np.asarray(prefix_tokens, dtype=np.int64)
    choice = np.asarray(choice_tokens, dtype=np.int64)
    if choice.size == 0:
        return prefix[:0], 0

    total_len = prefix.size + choice.size
    if total_len <= max_seq_len:
        return np.concatenate((prefix, choice)), max(0, prefix.size - 1)

    excess = total_len - max_seq_len
    if excess < prefix.size:
        prefix = prefix[excess:]
        ctx_len = prefix.size
        full_tokens = np.concatenate((prefix, choice))
    else:
        choice = choice[excess - prefix.size :]
        ctx_len = 0
        full_tokens = choice

    if ctx_len >= full_tokens.size:
        return full_tokens[:0], 0
    return full_tokens, max(0, ctx_len - 1)


def grouped_choice_scores(
    model: nn.Module,
    grouped_sequences: Sequence[Sequence[Sequence[int]]],
    grouped_start_positions: Sequence[Sequence[int]],
    *,
    vocab_size: int,
    device: str,
) -> List[List[float]]:
    group_sizes = [len(sequences) for sequences in grouped_sequences]
    flat_sequences = [seq for sequences in grouped_sequences for seq in sequences]
    flat_starts = [start for starts in grouped_start_positions for start in starts]

    if not flat_sequences:
        return [[] for _ in grouped_sequences]

    from ._eval_native import load_eval_native

    native_scorer = load_eval_native().grouped_choice_scores_packed_native
    packed, offsets, starts, groups = _pack_choice_sequences(
        flat_sequences,
        flat_starts,
        group_sizes,
    )
    mean_lps = native_scorer(
        model,
        packed,
        offsets,
        starts,
        groups,
        int(vocab_size),
        str(device),
    )
    return [chunk.tolist() for chunk in mean_lps.split(group_sizes)]
