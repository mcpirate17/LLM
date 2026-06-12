"""Transplant portability gate (WS-7).

Operationalizes the question every nano "win" this project has had to answer:
is a mechanism's advantage *intrinsic to the mechanism* or an *artifact of its
host*? (The slot-memory composer win, the surprise-memory family, etc.) We drop
the candidate mechanism into several structurally distinct host blocks and, in
each, measure its paired lift over a fixed baseline mixer occupying the same slot
(reusing the WS-2 paired probe). ``transplant_portability`` is the fraction of
hosts where the mechanism gives a CI-positive lift — a mechanism that only helps
in one host is a host artifact; one that helps across hosts is intrinsic.

The non-target slots of multi-lane hosts are filled with the same baseline in
both arms, so each host is a clean A/B differing only in the transplanted slot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch
from torch import nn

from ..generator.block_templates import (
    GatedParallelBlock,
    LatentCompressBlock,
    RecursiveDepthBlock,
    SparseMoEBlock,
)
from ..harness.standard_block import LaneTestBlock
from .paired import PairedDeltaCI, run_paired_probe

if TYPE_CHECKING:
    from ..proposer.spec_generator import ProposalSpec

LaneFactory = Callable[[int], nn.Module]
HostBuilder = Callable[[LaneFactory, LaneFactory, int], nn.Module]


class _CausalAttention(nn.Module):
    """Minimal single-head causal self-attention — the fixed baseline mixer.

    Always builds (no dispatch), maps [B,S,D]->[B,S,D]. A neutral "generic mixer"
    reference so transplant lift measures the mechanism against a known quantity.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        seq = x.shape[1]
        mask = torch.triu(
            torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return self.proj(torch.matmul(attn, v))


def _default_baseline_factory(dim: int) -> nn.Module:
    return _CausalAttention(dim)


# Host builders: (target_factory, filler_factory, dim) -> module. Non-target
# slots use filler_factory so the mechanism-vs-baseline arms differ only in the
# transplanted slot. Each host is a structurally distinct composition.
TRANSPLANT_HOSTS: dict[str, HostBuilder] = {
    "standard": lambda target, _filler, dim: LaneTestBlock(target(dim), dim),
    "latent_compress": lambda target, _filler, dim: LatentCompressBlock(target, dim),
    "recursive_depth": lambda target, _filler, dim: RecursiveDepthBlock(target, dim),
    "gated_parallel": lambda target, filler, dim: GatedParallelBlock(
        target, filler, dim
    ),
    "sparse_moe": lambda target, filler, dim: SparseMoEBlock(
        target, (filler, filler), dim
    ),
}


@dataclass(slots=True)
class HostLift:
    host: str
    ci: PairedDeltaCI
    positive: bool  # mechanism lift over baseline is CI-positive in this host


@dataclass(slots=True)
class TransplantScorecard:
    portability: float  # fraction of hosts with CI-positive mechanism lift
    n_hosts: int
    per_host: list[HostLift] = field(default_factory=list)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "transplant_portability": round(self.portability, 4),
            "transplant_n_hosts": self.n_hosts,
            "transplant_positive_hosts": [h.host for h in self.per_host if h.positive],
            "transplant_per_host_delta": {
                h.host: round(h.ci.mean, 6) for h in self.per_host
            },
        }


def transplant_portability(
    mechanism_factory: LaneFactory,
    *,
    baseline_factory: LaneFactory = _default_baseline_factory,
    hosts: dict[str, HostBuilder] | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    dim: int = 32,
    seq_len: int = 32,
    n_steps: int = 100,
) -> TransplantScorecard:
    """Per-host paired lift of ``mechanism_factory`` over ``baseline_factory``."""
    host_map = hosts if hosts is not None else TRANSPLANT_HOSTS
    per_host: list[HostLift] = []
    for name, build in host_map.items():
        # The baseline-vs-baseline arm is identical for every candidate when
        # the default fixed baseline is in play — cache it per host so each
        # additional candidate pays only its own arm (halves transplant cost).
        cacheable = baseline_factory is _default_baseline_factory
        ci = run_paired_probe(
            lambda b=build: b(mechanism_factory, baseline_factory, dim),
            lambda b=build: b(baseline_factory, baseline_factory, dim),
            seeds=seeds,
            dim=dim,
            seq_len=seq_len,
            n_steps=n_steps,
            anchor_cache_key=("transplant_baseline", name) if cacheable else None,
        )
        per_host.append(HostLift(host=name, ci=ci, positive=ci.excludes_zero))
    positive = sum(1 for h in per_host if h.positive)
    portability = positive / len(per_host) if per_host else 0.0
    return TransplantScorecard(
        portability=portability, n_hosts=len(per_host), per_host=per_host
    )


def transplant_metadata_for_spec(
    spec: "ProposalSpec",
    *,
    seeds: Sequence[int] = (0, 1, 2),
    dim: int = 32,
    seq_len: int = 32,
    n_steps: int = 100,
) -> dict[str, Any]:
    """Transplant scorecard metadata for a spec's mechanism, or an explicit skip.

    The mechanism is built parametrically (``generate_module_from_spec(spec, dim=d)``)
    so it can be re-instantiated at each host's working dimension. If the mechanism
    cannot build (un-dispatchable → raises since the fail-fast fix), records an
    explicit reason rather than fabricating a portability number.
    """
    from ..generator.code_generator import (
        UndispatchableSpecError,
        generate_module_from_spec,
    )

    def mechanism_factory(d: int) -> nn.Module:
        return generate_module_from_spec(spec, dim=d)

    try:
        mechanism_factory(dim)  # probe buildability once at the base dim
    except UndispatchableSpecError as exc:
        return {"transplant_skipped_reason": f"mechanism_unbuildable:{exc}"[:120]}
    card = transplant_portability(
        mechanism_factory, seeds=seeds, dim=dim, seq_len=seq_len, n_steps=n_steps
    )
    return card.to_metadata()
