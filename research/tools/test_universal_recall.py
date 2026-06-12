import json
import torch
from torch import nn
from component_fab.harness.harder_binding_tasks import (
    default_hard_binding_tasks,
    run_one_task_checkpoints,
)


class UniversalRecallLane(nn.Module):
    """Synthesizes Pooling, Latching, and Slotted Tables.

    Aims to be a general-purpose non-QKV recall expert.
    """

    def __init__(
        self,
        dim: int,
        n_slots: int = 16,
        memory_dim: int = 16,
        latch_len: int = 3,
        pool_period: int = 4,
    ) -> None:
        super().__init__()
        self.q = nn.Linear(dim, memory_dim, bias=False)
        self.k = nn.Linear(dim, memory_dim, bias=False)
        self.v = nn.Linear(dim, memory_dim, bias=False)

        self.latch_mix = nn.Linear(memory_dim * latch_len, memory_dim)
        self.write_route = nn.Linear(dim, n_slots)

        self.out = nn.Linear(memory_dim, dim, bias=False)

        self.n_slots = n_slots
        self.memory_dim = memory_dim
        self.latch_len = latch_len
        self.pool_period = pool_period

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        slot_keys = torch.zeros(
            batch_size, self.n_slots, self.memory_dim, device=x.device, dtype=x.dtype
        )
        slot_vals = torch.zeros(
            batch_size, self.n_slots, self.memory_dim, device=x.device, dtype=x.dtype
        )
        key_latch = [
            torch.zeros(batch_size, self.memory_dim, device=x.device, dtype=x.dtype)
            for _ in range(self.latch_len)
        ]

        # Temporal Pooler State
        pool_accum = torch.zeros(batch_size, dim, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            token = x[:, t]

            # 1. Temporal Pooling (Long-Gap)
            pool_accum = pool_accum + token
            if (t + 1) % self.pool_period == 0:
                pooled_token = pool_accum / float(self.pool_period)
                pool_accum = torch.zeros_like(pool_accum)
            else:
                pooled_token = (
                    token  # Fallback for non-update steps (simplification for testing)
                )

            # 2. Key-Context Latching (Compositional)
            kt = torch.tanh(self.k(pooled_token))
            vt = self.v(pooled_token)
            qt = torch.tanh(self.q(token))  # Read with original token

            latched_context = self.latch_mix(torch.cat(key_latch, dim=-1))

            # 3. Slotted Table (Distractor)
            w_route = torch.softmax(self.write_route(pooled_token), dim=-1)
            w_idx = w_route.argmax(dim=-1)
            mask = (
                torch.nn.functional.one_hot(w_idx, num_classes=self.n_slots)
                .unsqueeze(-1)
                .to(x.dtype)
            )

            # Write latched context as Key, pooled token as Value
            slot_keys = slot_keys * (1.0 - mask) + mask * latched_context.unsqueeze(1)
            slot_vals = slot_vals * (1.0 - mask) + mask * vt.unsqueeze(1)

            # Read
            read_weights = torch.softmax(
                torch.einsum("bd,bsd->bs", qt, slot_keys), dim=-1
            )
            read = torch.einsum("bs,bsd->bd", read_weights, slot_vals)

            outputs.append(self.out(read))

            key_latch = key_latch[1:] + [kt]

        return torch.stack(outputs, dim=1)


def run_eval():
    STEPS = 1000
    DIM = 64
    comparisons = [
        ("universal_recall", lambda d: UniversalRecallLane(d), "distractor_kv_recall"),
        ("universal_recall", lambda d: UniversalRecallLane(d), "long_gap_recall"),
        ("universal_recall", lambda d: UniversalRecallLane(d), "compositional_binding"),
    ]
    results = {}
    for name, factory, tn in comparisons:
        if name not in results:
            results[name] = {}
        task = next(t for t in default_hard_binding_tasks(seed=0) if t.name == tn)
        print(f"Running {name} on {tn}...", flush=True)
        rows = run_one_task_checkpoints(
            factory,
            task,
            eval_at_steps=(STEPS,),
            dim=DIM,
            seed=0,
            device="cuda",
            mixer_label=name,
        )
        results[name][tn] = rows[STEPS].eval_accuracy
        print(f"DONE: {name} {tn}: {results[name][tn]:.4f}")
    with open("research/reports/universal_recall_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_eval()
