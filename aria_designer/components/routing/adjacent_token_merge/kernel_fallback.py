"""Python fallback kernel for token_merge."""

import torch


class ComponentHandler:
    """Fallback handler for token_merge: merge similar tokens then restore.

    build() precomputes n_keep so forward() avoids per-call config parsing.
    The restore index tensor is cached per (S, device) to avoid repeated
    torch.arange + clamp allocations on the hot path.
    """

    __slots__ = ("_restore_cache",)

    def __init__(self):
        self._restore_cache: dict = {}

    def validate_config(self, config):
        return []

    def build(self, config):
        return {"n_keep": config.get("n_keep")}

    def _restore_indices(
        self, S: int, n_keep: int, device: torch.device
    ) -> torch.Tensor:
        key = (S, n_keep, device)
        idx = self._restore_cache.get(key)
        if idx is None:
            idx = torch.arange(S, device=device).clamp(max=n_keep - 1)
            self._restore_cache[key] = idx
        return idx

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        B, S, D = x.shape
        n_keep = (config.get("n_keep") if config else None) or max(1, S // 2)
        n_keep = min(n_keep, S)
        if n_keep >= S:
            return {"y": x}
        # Keep first n_keep tokens, fold remainder into last kept via mean
        kept = x[:, :n_keep, :].clone()
        kept[:, -1, :] = x[:, n_keep - 1 :, :].mean(dim=1)
        # Restore to original length — cached index avoids per-call allocation
        indices = self._restore_indices(S, n_keep, x.device)
        return {"y": kept[:, indices, :]}
