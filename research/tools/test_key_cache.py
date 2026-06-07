import json
import torch
from torch import nn
from component_fab.harness.harder_binding_tasks import (
    default_hard_binding_tasks,
    run_one_task_checkpoints,
)

class DeepKeyCacheMemoryLane(nn.Module):
    """Deep Key-Latch with an internal selection head for Compositional Binding."""
    def __init__(self, dim: int, memory_dim: int = 32, latch_len: int = 12) -> None:
        super().__init__()
        self.q = nn.Linear(dim, memory_dim, bias=False)
        self.k = nn.Linear(dim, memory_dim, bias=False)
        self.v = nn.Linear(dim, memory_dim, bias=False)
        self.write_gate = nn.Linear(dim, 1)
        
        # Internal selection head over the key cache
        self.selection_q = nn.Linear(dim, memory_dim)
        
        self.out = nn.Linear(memory_dim, dim, bias=False)
        self.latch_len = latch_len
        self.memory_dim = memory_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        memory = torch.zeros(batch_size, self.memory_dim, self.memory_dim, device=x.device, dtype=x.dtype)
        # Deep Key Cache
        key_cache = [torch.zeros(batch_size, self.memory_dim, device=x.device, dtype=x.dtype) for _ in range(self.latch_len)]
        
        outputs = []
        for t in range(seq_len):
            token = x[:, t]
            kt = torch.tanh(self.k(token))
            vt = self.v(token)
            gw = torch.sigmoid(self.write_gate(token))
            
            # 1. Selection Phase: Query the Key Cache
            sq = torch.tanh(self.selection_q(token))
            k_tensor = torch.stack(key_cache, dim=1) # [B, LatchLen, Dim]
            
            # Attention over the cache to pick the right context (Entity + Attribute)
            # [B, 1, Dim] @ [B, Dim, LatchLen] -> [B, 1, LatchLen]
            scores = torch.einsum("bd,bld->bl", sq, k_tensor)
            # Softmax to pick the best context combination
            attn_weights = torch.softmax(scores, dim=-1) 
            
            # The context is the weighted sum of cached keys
            latched_context = torch.einsum("bl,bld->bd", attn_weights, k_tensor)
            
            # 2. Write Phase: Use the selected context as the Key
            write = gw.unsqueeze(-1) * torch.einsum("bi,bj->bij", latched_context, vt)
            memory = 0.95 * memory + write
            
            # 3. Read Phase
            read = torch.einsum("bi,bij->bj", torch.tanh(self.q(token)), memory)
            outputs.append(self.out(read))
            
            # Update Cache
            key_cache = key_cache[1:] + [kt]
            
        return torch.stack(outputs, dim=1)

def run_eval():
    STEPS = 1000
    DIM = 64
    comparisons = [
        ("deep_key_cache", lambda d: DeepKeyCacheMemoryLane(d), "compositional_binding"),
    ]
    results = {}
    for name, factory, tn in comparisons:
        if name not in results: results[name] = {}
        task = next(t for t in default_hard_binding_tasks(seed=0) if t.name == tn)
        print(f"Running {name} on {tn}...", flush=True)
        rows = run_one_task_checkpoints(factory, task, eval_at_steps=(STEPS,), dim=DIM, seed=0, device="cuda", mixer_label=name)
        results[name][tn] = rows[STEPS].eval_accuracy
        print(f"DONE: {name} {tn}: {results[name][tn]:.4f}")
    with open("research/reports/key_cache_results.json", "w") as f: json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_eval()
