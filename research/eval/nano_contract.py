"""The single nano contract: one definition of "nano" + a demonstrated floor.

WHY THIS EXISTS
---------------
"nano" was used for everything from 16K-param operators to the 68M ``*_r7_nano``
lanes — a 4000x range with no shared spec and, worse, no proof that the minimum
size could learn anything. The nano screening leaderboard admitted sub-100K
models at a 100% pass rate while they scored induction AUC ~0.004 (pure chance):
noise masquerading as signal.

THE FLOOR (positive-control evidence, 2026-06-27)
-------------------------------------------------
A KNOWN-CAPABLE architecture — a 2-layer pre-norm single-head transformer, the
canonical induction-head circuit — was run through the real train-then-test
induction probe at increasing size (restricted vocab 256, 800 steps):

    dim  32  /   33.7K params -> induction AUC 0.01   (chance: cannot learn)
    dim  64  /  100K   params -> induction AUC 0.02   (chance: cannot learn)
    dim 128  /  331K   params -> induction AUC 0.30   (onset)
    dim 256  /  1.19M  params -> induction AUC 0.48   (reliable)

So below ~1.2M params even a perfect induction architecture floors at chance.
The nano floor is therefore ``dim=256`` / ``min_params=1_000_000``: the smallest
size at which a capable architecture RELIABLY learns. Models below the floor are
rejected at the write gate (``program_writes``), not screened.

NOTE on the reference: the repo's ``reference_training.BaselineTransformer``
(``nn.MultiheadAttention``, post-norm) is a WEAK induction learner — it reached
only ~0.10 AUC at 2500 steps at dim 256 — so it is NOT used as the positive
control here. ``reference_model`` below is the clean pre-norm circuit that
demonstrably learns; this is a measurement reference (the baseline to beat), not
a proposed architecture, so it does not touch the novel-mechanism mandate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # keep the write-path import (NANO/meets_param_floor) torch-free
    from torch import nn


@dataclass(frozen=True)
class NanoContract:
    """The single, enforced definition of a nano model.

    Every nano screening model is evaluated at this spec, and a candidate whose
    instantiated model falls below ``min_params`` is rejected — it is below the
    demonstrated learnability floor and cannot carry screening signal.
    """

    dim: int = 256
    vocab: int = 256
    n_layers: int = 2
    seq_len: int = 64
    # Evidence-based floor: a capable 2-layer transformer reaches induction AUC
    # ~0.48 here and ~chance below it (see module docstring).
    min_params: int = 1_000_000
    # The reference positive control must clear this induction AUC at the floor,
    # or the floor is noise. Set below the measured 0.48 for a stable regression
    # gate, well above the sub-floor chance level (~0.02).
    learnability_threshold: float = 0.35


NANO = NanoContract()


def meets_param_floor(param_count: int | None) -> bool:
    """True iff ``param_count`` is at or above the nano learnability floor.

    A missing/zero count is treated as NOT meeting the floor only by callers that
    require a count; this predicate itself returns False for ``None``/``<=0`` so
    the floor is never silently passed by an absent measurement.
    """
    return param_count is not None and int(param_count) >= NANO.min_params


def reference_model(
    *, vocab: int | None = None, dim: int | None = None, n_layers: int | None = None
) -> "nn.Module":
    """The canonical induction-capable positive control at the nano floor.

    A 2-layer pre-norm single-head transformer — the classic induction-head
    circuit (previous-token head feeding an induction head). Used by the
    learnability gate to PROVE the floor can learn; if this stops clearing
    ``learnability_threshold``, the floor (or the probe) has regressed.
    """
    import torch
    from torch import nn

    d = dim or NANO.dim
    v = vocab or NANO.vocab
    layers = n_layers or NANO.n_layers

    class _Block(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.q = nn.Linear(dim, dim)
            self.k = nn.Linear(dim, dim)
            self.v = nn.Linear(dim, dim)
            self.o = nn.Linear(dim, dim)
            self.n1 = nn.LayerNorm(dim)
            self.n2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
            )

        def forward(self, h: "torch.Tensor") -> "torch.Tensor":
            x = self.n1(h)
            _, length, dim = x.shape
            attn = (self.q(x) @ self.k(x).transpose(1, 2)) / dim**0.5
            attn = attn + torch.triu(
                torch.full((length, length), float("-inf"), device=h.device), 1
            )
            h = h + self.o(torch.softmax(attn, dim=-1) @ self.v(x))
            return h + self.mlp(self.n2(h))

    class _RefLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok = nn.Embedding(v, d)
            self.blocks = nn.ModuleList(_Block(d) for _ in range(layers))
            self.head = nn.Linear(d, v)

        def forward(self, ids: "torch.Tensor") -> "torch.Tensor":
            h = self.tok(ids)
            for blk in self.blocks:
                h = blk(h)
            return self.head(h)

    return _RefLM()


def measure_reference_induction_auc(
    *, n_train_steps: int = 800, seed: int = 0, device: str = "cpu"
) -> float:
    """Train-then-test induction AUC of the reference model at the nano floor.

    This IS the demonstrated evidence that the minimum size can learn. Expensive
    (it trains the reference) — call it from the learnability gate / on demand,
    not in a hot path.
    """
    from research.eval.induction_intermediate_probe import run_induction_intermediate

    model = reference_model()
    result = run_induction_intermediate(
        model,
        seeds=(seed,),
        gaps=(4, 16, 64),
        n_train_steps=n_train_steps,
        n_eval=64,
        batch_size=8,
        device=device,
    )
    return float(result.auc)
