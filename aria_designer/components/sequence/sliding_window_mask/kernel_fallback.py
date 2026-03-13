"""Python fallback kernel for sliding_window_mask."""

import torch


def _apply_sliding_window(x: torch.Tensor, window_size: int) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape
    safe_window = min(max(1, int(window_size)), seq_len)
    row_idx = torch.arange(seq_len, device=x.device).unsqueeze(1)
    col_idx = torch.arange(seq_len, device=x.device).unsqueeze(0)
    distance = row_idx - col_idx
    mask = (distance >= 0) & (distance < safe_window)
    decay = torch.exp(-distance.float().clamp(min=0) / max(safe_window / 4, 1.0))
    weights = mask.float() * decay
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.bmm(weights.unsqueeze(0).expand(batch_size, -1, -1), x)


class ComponentHandler:
    """Fallback handler for sliding_window_mask."""

    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        return {"y": _apply_sliding_window(x, int(config.get("window_size", 64)))}
