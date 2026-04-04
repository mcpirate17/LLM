"""Shared grouped-choice log-prob scoring."""

from __future__ import annotations

from typing import List, Sequence

import torch.nn as nn

from .utils import batched_span_mean_log_probs


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

    mean_lps = batched_span_mean_log_probs(
        model,
        flat_sequences,
        flat_starts,
        vocab_size=vocab_size,
        device=device,
    )
    return [chunk.tolist() for chunk in mean_lps.split(group_sizes)]
