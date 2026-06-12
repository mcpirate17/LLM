import json
import torch

from research.tools.gemini_slot_snapshot import GeminiSlotMemoryLane
from component_fab.harness.harder_binding_tasks import (
    default_hard_binding_tasks,
    run_one_task_checkpoints,
)


class ContentRoutedMasterLane(GeminiSlotMemoryLane):
    """Slotted Latched Memory with Content-Aware Routing.

    The router looks at the Latched Context to decide which slot to use.
    This helps the model 'index' associations by their keys.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
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

        outputs = []
        for t in range(seq_len):
            token = x[:, t]
            kt = torch.tanh(self.k(token))
            vt = self.v(token)
            qt = torch.tanh(self.q(token))

            latched_context = self.latch_mix(torch.cat(key_latch, dim=-1))

            # Content-Aware Routing: route based on the KEY context
            w_route = torch.softmax(self.write_route(latched_context), dim=-1)
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


def run_eval():
    STEPS = 1000
    DIM = 64
    comparisons = [
        (
            "content_master",
            lambda d: ContentRoutedMasterLane(d),
            "distractor_kv_recall",
        ),
        (
            "content_master",
            lambda d: ContentRoutedMasterLane(d),
            "compositional_binding",
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
