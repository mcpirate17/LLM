import json
import torch
from torch import nn
from component_fab.generator.memory_primitives import (
    DataDependentDecayMemoryLane,
    DeltaDecayMemoryLane,
    HierarchicalResidualCompressorLane,
)
from component_fab.harness.harder_binding_tasks import (
    default_hard_binding_tasks,
    run_one_task_checkpoints,
)


class FixedHierarchicalResidualCompressorLane(nn.Module):
    """Truly hierarchical version where level l sees level l-1 summary."""

    def __init__(self, dim: int, n_levels: int = 4) -> None:
        super().__init__()
        self.updates = nn.ModuleList(
            [nn.Linear(dim * 2 if i == 0 else dim * 3, dim) for i in range(n_levels)]
        )
        self.gates = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_levels)])
        self.read = nn.Linear(dim * n_levels, dim, bias=False)
        self.n_levels = n_levels
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        summaries = [
            torch.zeros(batch_size, dim, device=x.device, dtype=x.dtype)
            for _ in range(self.n_levels)
        ]
        outputs = []
        for t in range(seq_len):
            token = x[:, t]
            for level, update in enumerate(self.updates):
                period = 2**level
                if t % period != 0:
                    continue
                if level == 0:
                    inp = torch.cat([summaries[level], token], dim=-1)
                else:
                    inp = torch.cat(
                        [summaries[level], summaries[level - 1], token], dim=-1
                    )

                candidate = torch.tanh(update(inp))
                gate = torch.sigmoid(self.gates[level](token))
                summaries[level] = (1.0 - gate) * summaries[level] + gate * candidate
            outputs.append(self.read(torch.cat(summaries, dim=-1)))
        return torch.stack(outputs, dim=1)


def run_eval():
    STEPS = 1000  # Faster validation
    DIM = 64

    comparisons = [
        ("ddecay", lambda d: DataDependentDecayMemoryLane(d), "long_gap_recall"),
        ("delta_ddecay", lambda d: DeltaDecayMemoryLane(d), "distractor_kv_recall"),
        ("ddecay", lambda d: DataDependentDecayMemoryLane(d), "distractor_kv_recall"),
        (
            "hier_n4",
            lambda d: HierarchicalResidualCompressorLane(d, n_levels=4),
            "long_gap_recall",
        ),
        (
            "hier_fixed_n4",
            lambda d: FixedHierarchicalResidualCompressorLane(d, n_levels=4),
            "long_gap_recall",
        ),
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

    with open("research/reports/fix_attempt_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_eval()
