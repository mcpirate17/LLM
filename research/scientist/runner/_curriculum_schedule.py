from __future__ import annotations

from typing import Optional, Tuple

import torch


def precompute_curriculum_seq_lens(
    curriculum, n_steps: int
) -> Optional[Tuple[int, ...]]:
    """Return a native-precomputed curriculum schedule when available."""
    seq_len_tensor = getattr(curriculum, "seq_len_tensor", None)
    if not callable(seq_len_tensor):
        return None
    schedule = seq_len_tensor(int(n_steps))
    if not torch.is_tensor(schedule) or schedule.numel() < int(n_steps):
        raise ValueError(
            "curriculum seq_len_tensor returned an invalid schedule "
            f"(n_steps={n_steps}, value={type(schedule).__name__})"
        )
    if schedule.device.type != "cpu":
        schedule = schedule.cpu()
    if schedule.dtype != torch.long:
        schedule = schedule.to(dtype=torch.long)
    return tuple(int(value) for value in schedule[: int(n_steps)].tolist())
