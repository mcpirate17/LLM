"""Python fallback kernel for token_merge."""

import torch


class ComponentHandler:
    """Fallback handler for token_merge: merge similar tokens then restore."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        B, S, D = x.shape
        n_keep = config.get("n_keep") or max(1, S // 2)
        n_keep = min(n_keep, S)
        # Keep first n_keep tokens, average the rest into last kept token
        if n_keep >= S:
            return {"y": x}
        kept = x[:, :n_keep, :].clone()
        kept[:, -1, :] = x[:, n_keep - 1 :, :].mean(dim=1)
        # Restore to original length by nearest-neighbor repeat
        indices = torch.arange(S, device=x.device).clamp(max=n_keep - 1)
        y = kept[:, indices, :]
        return {"y": y}
