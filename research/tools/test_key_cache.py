import torch
from torch import nn

from research.tools._recall_probe_common import run_comparisons


class DeepKeyCacheMemoryLane(nn.Module):
    """Deep Key-Latch with an internal selection head for Compositional Binding."""

    def __init__(self, dim: int, memory_dim: int = 32, latch_len: int = 12) -> None:
        super().__init__()
        self.q = nn.Linear(dim, memory_dim, bias=False)
        self.k = nn.Linear(dim, memory_dim, bias=False)
        self.v = nn.Linear(dim, memory_dim, bias=False)
        self.write_gate = nn.Linear(dim, 1)
        self.selection_q = nn.Linear(dim, memory_dim)
        self.out = nn.Linear(memory_dim, dim, bias=False)
        self.latch_len = latch_len
        self.memory_dim = memory_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        memory = torch.zeros(
            batch_size, self.memory_dim, self.memory_dim, device=x.device, dtype=x.dtype
        )
        key_cache = [
            torch.zeros(batch_size, self.memory_dim, device=x.device, dtype=x.dtype)
            for _ in range(self.latch_len)
        ]
        outputs = []
        for t in range(seq_len):
            token = x[:, t]
            kt = torch.tanh(self.k(token))
            vt = self.v(token)
            gw = torch.sigmoid(self.write_gate(token))

            sq = torch.tanh(self.selection_q(token))
            k_tensor = torch.stack(key_cache, dim=1)
            scores = torch.einsum("bd,bld->bl", sq, k_tensor)
            attn_weights = torch.softmax(scores, dim=-1)
            latched_context = torch.einsum("bl,bld->bd", attn_weights, k_tensor)

            write = gw.unsqueeze(-1) * torch.einsum("bi,bj->bij", latched_context, vt)
            memory = 0.95 * memory + write

            read = torch.einsum("bi,bij->bj", torch.tanh(self.q(token)), memory)
            outputs.append(self.out(read))
            key_cache = key_cache[1:] + [kt]
        return torch.stack(outputs, dim=1)


def run_eval() -> None:
    STEPS = 1000
    DIM = 64
    comparisons = [
        (
            "deep_key_cache",
            lambda d: DeepKeyCacheMemoryLane(d),
            "compositional_binding",
        ),
    ]
    run_comparisons(
        comparisons,
        steps=STEPS,
        dim=DIM,
        device="cuda",
        out_path="research/reports/key_cache_results.json",
    )


if __name__ == "__main__":
    run_eval()
