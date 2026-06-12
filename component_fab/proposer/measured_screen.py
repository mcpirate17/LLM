"""Measured graph-property screen for fab specs.

Runs the position-Jacobian descriptor instrument
(``research.tools.measured_descriptors``) directly on the REAL generated fab
module — not a proxy graph. This replaces the NAS oracle proxy, which a
2026-06-03 audit found gate-failed 34/34 Tier-2-evidence candidates (incl. 7/7
winners) and was anti-correlated with Tier-2 because ``spec_to_proxy_graph``'s
3-op stub is OOD for the oracle. The measured descriptors are read from the
candidate's actual computation at random init and generalize to novel archs
(single-feat ROC 0.76-0.82, OOF 0.977; see ``project_measured_descriptors``).

This is the "ton of graph properties → filter what won't bind → rank" screen the
NAS pipeline is meant to provide, applied to fab candidates on their real module.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.proposer.spec_generator import ProposalSpec

logger = logging.getLogger(__name__)

_PROBE_VOCAB = 256  # matches research.eval.induction_probe._RESTRICTED_VOCAB
# Validated operating point (n=1102, 2026-05-25): long_range_reach >= 0.01 keeps
# 99.3% of induction-capable graphs while pruning ~55% of incapable ones.
LONG_RANGE_THRESHOLD = 0.01
# A near-causal operator keeps almost no query mass on future positions.
MAX_CAUSALITY_VIOLATION = 0.5


class _FabProbeAdapter(nn.Module):
    """Present a fab lane as the SynthesizedModel probe contract.

    The descriptor instrument calls ``embed(ids)`` then
    ``_fingerprint_forward_from_embed(emb)``; a fab lane is just an
    ``[B, L, dim] -> [B, L, dim]`` sequence mixer, so we add a random embedding
    table and forward the lane.
    """

    def __init__(self, module: nn.Module, dim: int, vocab: int = _PROBE_VOCAB) -> None:
        super().__init__()
        self.module = module
        self.embed_table = nn.Embedding(vocab, dim)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed_table(ids)

    def _fingerprint_forward_from_embed(self, emb: torch.Tensor) -> torch.Tensor:
        out = self.module(emb)
        if isinstance(out, tuple):
            out = out[0]
        return out


REASON_UNSTABLE = "unstable_nonfinite"
REASON_NON_BINDER = "non_binder"


@dataclass(frozen=True, slots=True)
class MeasuredScreen:
    """Measured graph-property screen for one fab spec."""

    proposal_id: str
    available: bool
    binds_likely: bool
    long_range_reach: float
    content_match_gating: float
    content_dependence: float
    causality_violation: float
    rank_score: float
    capability_score: float = 0.0
    unstable: bool = False
    descriptors: dict[str, float] | None = None
    reason: str = ""


def _capability_score(d: dict[str, float]) -> float:
    """Label-free capability rank from the shared measured-descriptor instrument.

    Delegates to ``research.tools.measured_descriptors.capability_score_from_descriptors``
    (the closed-book, validate-once instrument owned by the NAS side) — single source
    of truth, no bespoke re-derivation. Higher = routes info backward, content-gated,
    content-dependent, causal, stable. This is a CROSS-FAMILY filter signal; a
    2026-06-03 audit found it does NOT rank 'beats baseline' WITHIN a homogeneous
    family (use the comparative_probe measurement for that).
    """

    from research.tools.measured_descriptors import capability_score_from_descriptors

    return capability_score_from_descriptors(d)


def _unavailable(proposal_id: str, reason: str) -> MeasuredScreen:
    # Fail OPEN: never silently drop a candidate the probe could not measure.
    return MeasuredScreen(
        proposal_id=proposal_id,
        available=False,
        binds_likely=True,
        long_range_reach=0.0,
        content_match_gating=0.0,
        content_dependence=0.0,
        causality_violation=0.0,
        rank_score=0.0,
        reason=reason,
    )


def measured_screen_for_spec(
    spec: ProposalSpec,
    *,
    dim: int = 32,
    extractor: Any | None = None,
) -> MeasuredScreen:
    """Probe the real fab module and return its measured graph-property screen."""

    try:
        from research.tools.measured_descriptors import MeasuredDescriptorExtractor

        extractor = extractor or MeasuredDescriptorExtractor(n_seeds=2)

        def factory(seed: int) -> nn.Module:
            torch.manual_seed(seed)
            module = generate_module_from_spec(spec, dim=dim)
            return _FabProbeAdapter(module, dim).to(extractor.device).eval()

        d = extractor.descriptors_from_factory(factory)
        if d is None:
            return _unavailable(spec.proposal_id, "descriptor probe returned None")
        # Non-finite descriptors mean the module's forward/backward blew up (NaN/inf)
        # at random init — a genuinely broken candidate. Hard-reject as unstable
        # (fail CLOSED — distinct from a healthy non-binder).
        if not all(math.isfinite(v) for v in d.values()):
            return MeasuredScreen(
                proposal_id=spec.proposal_id,
                available=True,
                binds_likely=False,
                long_range_reach=0.0,
                content_match_gating=0.0,
                content_dependence=0.0,
                causality_violation=0.0,
                rank_score=0.0,
                unstable=True,
                descriptors=d,
                reason=REASON_UNSTABLE,
            )
        cap = _capability_score(d)
        return MeasuredScreen(
            proposal_id=spec.proposal_id,
            available=True,
            binds_likely=d["long_range_reach"] >= LONG_RANGE_THRESHOLD,
            long_range_reach=d["long_range_reach"],
            content_match_gating=d["content_match_gating"],
            content_dependence=d["content_dependence"],
            causality_violation=d["causality_violation"],
            rank_score=max(0.0, cap),
            capability_score=cap,
            descriptors=d,
            reason=""
            if d["long_range_reach"] >= LONG_RANGE_THRESHOLD
            else REASON_NON_BINDER,
        )
    except Exception as exc:  # noqa: BLE001 - screen is best-effort, fail open
        logger.warning("measured screen unavailable for %s: %s", spec.proposal_id, exc)
        return _unavailable(spec.proposal_id, str(exc))
