import json
import torch
from torch import nn
from component_fab.harness.harder_binding_tasks import (
    default_hard_binding_tasks,
    run_one_task_checkpoints,
)
from component_fab.generator.memory_primitives import LegendreSSMLane
from component_fab.harness.state_tracking_suite import score_state_tracking


class UniversalRecallLane(nn.Module):
    """Synthesizes Pooling, Latching, and Slotted Tables."""

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
        pool_accum = torch.zeros(batch_size, dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            token = x[:, t]
            pool_accum = pool_accum + token
            if (t + 1) % self.pool_period == 0:
                pooled_token = pool_accum / float(self.pool_period)
                pool_accum = torch.zeros_like(pool_accum)
            else:
                pooled_token = token
            kt = torch.tanh(self.k(pooled_token))
            vt = self.v(pooled_token)
            qt = torch.tanh(self.q(token))
            latched_context = self.latch_mix(torch.cat(key_latch, dim=-1))
            w_route = torch.softmax(self.write_route(pooled_token), dim=-1)
            w_idx = w_route.argmax(dim=-1)
            mask = (
                torch.nn.functional.one_hot(w_idx, num_classes=self.n_slots)
                .unsqueeze(-1)
                .to(x.dtype)
            )
            slot_keys = slot_keys * (1.0 - mask) + mask * latched_context.unsqueeze(1)
            slot_vals = slot_vals * (1.0 - mask) + mask * vt.unsqueeze(1)
            read_weights = torch.softmax(
                torch.einsum("bd,bsd->bs", qt, slot_keys), dim=-1
            )
            read = torch.einsum("bs,bsd->bd", read_weights, slot_vals)
            outputs.append(self.out(read))
            key_latch = key_latch[1:] + [kt]
        return torch.stack(outputs, dim=1)


class OrthogonalLaneBlock(nn.Module):
    """Runs a Recall expert and a State-Tracking expert in parallel orthogonal lanes."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        # We divide the dimensionality between the two experts
        dim_recall = dim // 2
        dim_state = dim - dim_recall

        self.in_proj_recall = nn.Linear(dim, dim_recall)
        self.in_proj_state = nn.Linear(dim, dim_state)

        self.recall_lane = UniversalRecallLane(dim_recall)
        self.state_lane = LegendreSSMLane(dim_state)

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input
        x_recall = self.in_proj_recall(x)
        x_state = self.in_proj_state(x)

        # Parallel Execution
        out_recall = self.recall_lane(x_recall)
        out_state = self.state_lane(x_state)

        # Concatenate and Project back
        combined = torch.cat([out_recall, out_state], dim=-1)
        return self.out_proj(combined)


def run_eval():
    STEPS = 1000
    DIM = 64

    comparisons = [
        ("orthogonal_block", lambda d: OrthogonalLaneBlock(d), "distractor_kv_recall"),
        ("orthogonal_block", lambda d: OrthogonalLaneBlock(d), "long_gap_recall"),
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

    print("Running orthogonal_block on state_tracking...", flush=True)
    state_scores = score_state_tracking(
        lambda d: OrthogonalLaneBlock(d),
        dim=32,
        seq_len=32,
        n_steps=400,
        seeds=(0,),
        device="cpu",
    )
    results["orthogonal_block"]["state_tracking"] = state_scores["per_axis"][
        "state_tracking"
    ]
    print(
        f"DONE: orthogonal_block state_tracking: {results['orthogonal_block']['state_tracking']:.4f}"
    )

    with open("research/reports/orthogonal_block_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_eval()
