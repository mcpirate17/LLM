"""Python fallback kernel for product_key_memory."""

import torch


class ComponentHandler:
    def validate_config(self, config):
        return []

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]
        B, S, D = x.shape
        num_keys = max(2, min(64, int(config.get("num_keys", 32))))
        top_k = max(1, min(8, int(config.get("top_k", 4))))
        left_dim = max(1, D // 2)
        right_dim = max(1, D - left_dim)
        left_key = torch.linspace(
            -1, 1, num_keys * left_dim, device=x.device, dtype=x.dtype
        ).view(num_keys, left_dim)
        right_key = torch.linspace(
            1, -1, num_keys * right_dim, device=x.device, dtype=x.dtype
        ).view(num_keys, right_dim)
        left = x[..., :left_dim]
        right = x[..., left_dim:]
        left_scores = torch.matmul(left, left_key.t())
        right_scores = torch.matmul(right, right_key.t())
        scores = (left_scores.unsqueeze(-1) + right_scores.unsqueeze(-2)).reshape(
            B, S, -1
        )
        vals = torch.tanh(
            torch.linspace(
                -1, 1, num_keys * num_keys * D, device=x.device, dtype=x.dtype
            )
        ).view(num_keys * num_keys, D)
        top_scores, top_idx = torch.topk(scores, min(top_k, scores.shape[-1]), dim=-1)
        weights = torch.softmax(top_scores, dim=-1)
        gathered = vals[top_idx.reshape(-1)].reshape(B, S, -1, D)
        return {"y": (weights.unsqueeze(-1) * gathered).sum(dim=-2)}
