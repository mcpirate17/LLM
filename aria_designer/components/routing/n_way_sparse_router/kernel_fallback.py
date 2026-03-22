"""Python fallback kernel for n_way_sparse_router."""

import torch
import torch.nn.functional as F


class ComponentHandler:
    """N-way sparse router: N bottleneck experts, top-k activation."""

    def validate_config(self, config):
        errors = []
        n = config.get("n_ways", 4)
        k = config.get("top_k", 2)
        if not isinstance(n, int) or n < 2:
            errors.append("n_ways must be int >= 2")
        if not isinstance(k, int) or k < 1:
            errors.append("top_k must be int >= 1")
        if isinstance(n, int) and isinstance(k, int) and k > n:
            errors.append("top_k must be <= n_ways")
        return errors

    def build(self, config):
        return None

    def forward(self, inputs, config):
        x = inputs["x"]  # (B, S, D)
        B, S, D = x.shape
        n_ways = config.get("n_ways", 4)
        top_k = config.get("top_k", 2)
        n_ways = max(2, min(n_ways, 16))
        top_k = max(1, min(top_k, n_ways))

        hidden = D // n_ways

        # Gate
        W_gate = _lazy_param((D, n_ways), x.device, x.dtype, seed=0)
        gate_logits = x @ W_gate  # (B, S, N)

        # Top-k selection
        topk_vals, topk_idx = gate_logits.topk(top_k, dim=-1)  # (B, S, k)
        gate_weights = F.softmax(topk_vals, dim=-1)  # (B, S, k)

        # Expert forward (all experts, then mask)
        output = torch.zeros_like(x)
        for i in range(n_ways):
            W_down = _lazy_param((D, hidden), x.device, x.dtype, seed=100 + i)
            W_up = _lazy_param((hidden, D), x.device, x.dtype, seed=200 + i)
            expert_out = F.gelu(x @ W_down) @ W_up  # (B, S, D)

            # Mask: is expert i in the top-k for each token?
            (topk_idx == i).any(dim=-1, keepdim=True).float()  # (B, S, 1)
            # Weight: sum of gate weights where this expert was selected
            weight = torch.zeros(B, S, 1, device=x.device, dtype=x.dtype)
            for k_idx in range(top_k):
                match = (topk_idx[:, :, k_idx] == i).unsqueeze(-1).float()
                weight += match * gate_weights[:, :, k_idx].unsqueeze(-1)

            output += expert_out * weight

        return {"y": output}


def _lazy_param(shape, device, dtype, seed=0):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    w = torch.randn(*shape, generator=gen, dtype=dtype).to(device)
    w *= shape[0] ** -0.5
    return w
