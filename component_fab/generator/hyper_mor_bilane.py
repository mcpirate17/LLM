"""Native Hyperbolic Surprise MoR — the integrated lane.

Combines, in one gated bilane, the three mechanisms from the redesign note
(``research/notes/novel_mechanism_architecture_redesign_2026-06-14.md`` NEXT BIG
RUN #2):

- **lane_a** — the learnt-MoR adaptive-surprise trunk
  (``MoRSurpriseRefineMLPLaneA``): native semiring surprise memory with the
  1288+-param surprise+loss MLP halting router, running on the validated
  ``native_mor_refine_cuda.cu`` kernel. The genuinely novel "thinking" trunk.
- **lane_b** — ``HyperbolicAttention``: addressing scored by Lorentz-model
  geodesic distance with a *learned* curvature, replacing the Euclidean
  Titans-MAC aux lane. Non-Euclidean geometry packs hierarchy a flat dot product
  cannot. (Standard attention math — cuBLAS matmuls + softmax over the
  hyperbolic-distance scores; not a custom kernel.)

This is the REAL test of hyperbolic addressing: bare on flat induction it only
*tied* reciprocal — its hierarchy value can only show **with** the surprise
memory and on hierarchical structure, which is exactly this combination.

**Anti-starvation floor (mission-critical).** A free sigmoid gate between a novel
trunk and a softmax-based addressing lane is the same shape as the reciprocal
"softmax twin" gate that starved native to 0.58%% at 100K. Hyperbolic still ends
in a softmax, so the gate could collapse onto it and abandon the surprise-MoR
trunk. We therefore **floor the surprise-MoR branch** at ``SURPRISE_FLOOR`` (the
trunk always carries >= floor of the mix) and log the per-forward gate fraction
so any drift toward the hyperbolic escape hatch is visible, not silent. Set
``SURPRISE_FLOOR = 0.0`` to recover the plain gated bilane for the ablation.
"""

from __future__ import annotations

import torch

from .mor_bilane import MoRSurpriseRefineMLPAdaptiveSemiringBiLaneSurpriseMemoryLane
from .primitive_templates import HyperbolicAttention


class HyperbolicMoRSurpriseRefineMLPBiLane(
    MoRSurpriseRefineMLPAdaptiveSemiringBiLaneSurpriseMemoryLane
):
    """Surprise-MoR trunk (lane_a) gated with hyperbolic-distance addressing (lane_b).

    Everything except the aux lane and the floored blend is inherited: lane_a, the
    MoR router, the surprise memory, the CUDA recursion kernel, the ponder cost,
    and the ``ROUTER_HIDDEN`` / ``SURPRISE_COUPLING_MIN`` knobs the factory bakes in.
    """

    #: Minimum mix weight kept on the novel surprise-MoR trunk so the gate cannot
    #: starve it by collapsing onto the hyperbolic-softmax pathway.
    SURPRISE_FLOOR: float = 0.25

    def _build_aux_lanes(self, dim, memory_dim, gate_bias, semiring_temp_init) -> None:
        self.lane_b = HyperbolicAttention(dim, use_rope=True)
        #: mean mix weight on the surprise-MoR trunk last forward (collapse monitor).
        self.last_trunk_frac: float | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.lane_a(x)  # surprise-MoR trunk  [B, L, dim]
        b = self.lane_b(x)  # hyperbolic addressing [B, L, dim]
        gate = torch.sigmoid(self.gate(x))  # [B, L, 1] in (0, 1)
        floor = self.SURPRISE_FLOOR
        # trunk weight in [floor, 1]; hyperbolic gets at most (1 - floor).
        trunk = floor + (1.0 - floor) * gate
        self.last_trunk_frac = float(trunk.mean().detach())
        return trunk * a + (1.0 - trunk) * b
