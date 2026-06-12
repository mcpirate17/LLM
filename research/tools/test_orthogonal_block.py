import json

import torch
from torch import nn

from component_fab.generator.memory_primitives import LegendreSSMLane
from component_fab.harness.state_tracking_suite import score_state_tracking
from research.tools._recall_probe_common import UniversalRecallLane, run_comparisons


class OrthogonalLaneBlock(nn.Module):
    """Runs a Recall expert and a State-Tracking expert in parallel orthogonal lanes."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        dim_recall = dim // 2
        dim_state = dim - dim_recall
        self.in_proj_recall = nn.Linear(dim, dim_recall)
        self.in_proj_state = nn.Linear(dim, dim_state)
        self.recall_lane = UniversalRecallLane(dim_recall)
        self.state_lane = LegendreSSMLane(dim_state)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_recall = self.in_proj_recall(x)
        x_state = self.in_proj_state(x)
        out_recall = self.recall_lane(x_recall)
        out_state = self.state_lane(x_state)
        combined = torch.cat([out_recall, out_state], dim=-1)
        return self.out_proj(combined)


def run_eval() -> None:
    STEPS = 1000
    DIM = 64

    comparisons = [
        ("orthogonal_block", lambda d: OrthogonalLaneBlock(d), "distractor_kv_recall"),
        ("orthogonal_block", lambda d: OrthogonalLaneBlock(d), "long_gap_recall"),
    ]
    results = run_comparisons(
        comparisons,
        steps=STEPS,
        dim=DIM,
        device="cuda",
        out_path="research/reports/orthogonal_block_results.json",
    )

    # Extra: state-tracking probe (unique to this script)
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

    # Re-write with state_tracking included
    with open("research/reports/orthogonal_block_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_eval()
