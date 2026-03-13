"""Python fallback kernel for token_merge."""

import torch

class ComponentHandler:
    """Fallback handler for token_merge."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        if x.dim() != 3:
            raise ValueError("token_merge expects x with shape [B, S, D]")
        batch_size, seq_len, _ = x.shape
        n_keep = max(1, min(int(config.get("n_keep", seq_len)), seq_len))
        restore_row = torch.arange(seq_len, device=x.device, dtype=torch.long).clamp(max=n_keep - 1)
        restore_map = restore_row.unsqueeze(0).expand(batch_size, -1)
        return {"y": x[:, :n_keep, :], "restore_map": restore_map}
