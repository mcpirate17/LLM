"""Capability-screener scoring for fab specs (induction / nano predictors).

Wraps the validated NAS capability screeners (op-presence GBM, OOD leave-family-out
ROC ~0.80-0.91, generalizes to novel winners — see ``project_capability_screener_rebuild_ood``)
so they can rank fab candidates. The screener consumes op-presence + op_count +
pair_count, NOT graph topology, so the proxy-graph OOD problem (which broke the
``pls_partition_oracle`` axes) does not apply — but the op-set must be FAITHFUL:
a 3-op stub scores softmax at ~0.015, while a real op multiset ranks attention >>
SSM >> MLP correctly (validated 2026-06-03).

We get a faithful op multiset by introspecting the real generated ``nn.Module`` and
mapping its submodule classes to the screener's op vocabulary, scaled by the block
count the harness will stack.

Honest scope: the screeners predict CAPABILITY (will it learn induction / nano
binding) — necessary, not sufficient, for beating a baseline. Within a homogeneous
candidate family (e.g. many tropical variants) discrimination is weak; the power is
cross-family. Use as a ranking signal, not a promotion gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.proposer.spec_generator import ProposalSpec

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]
_INDUCTION_DIR = _REPO / "research" / "runtime" / "capability_screener"
_NANO_DIR = _REPO / "research" / "runtime" / "capability_screener_nano"

# Map generated fab lane/block classes to the screener's op vocabulary. Unmapped
# classes are skipped (the screener tolerates partial op-presence). Keep keys in
# sync with classes emitted by component_fab.generator.code_generator.
_FAB_CLASS_TO_OP: dict[str, str] = {
    "Linear": "linear_proj",
    "TropicalAttention": "tropical_attention",
    "MixtureOfRecursionsLane": "adaptive_recursion",
    "GatedParallelBlock": "gated_linear",
    "ThreeLaneAdaptive": "route_lanes",
    "PoincareAttention": "hyp_linear",
    "HyperbolicBridgeBlock": "hyperbolic_norm",
    "QuaternionAttention": "rotor_transform",
    "FisherAttention": "softmax_attention",
    "GraphAttentionBlock": "graph_attention",
    "MultiscaleWaveletLane": "wavelet_packet_mix",
    "RandomFeatureKernelLane": "linear_attention",
    "LinearStateSpaceLane": "state_space",
    "CliffordAttention": "geometric_product",
    "SpikingActivationGate": "lif_neuron",
    "PadicProjection": "padic_gate",
    "SemiringSurpriseMemoryLane": "associative_memory",
    "ChebyshevSpectralMix": "chebyshev_spectral_mix",
    "TensorTuckerLane": "low_rank_proj",
}


@dataclass(frozen=True, slots=True)
class CapabilityScreen:
    proposal_id: str
    available: bool
    induction_pred: float
    nano_pred: float
    op_count: int
    n_distinct_ops: int
    reason: str = ""


def fab_op_multiset(
    spec: ProposalSpec, *, dim: int = 32, n_blocks: int = 2
) -> list[str]:
    """Faithful op multiset of the full stacked model for ``spec``.

    Introspects the real generated lane module, maps submodule classes to the op
    vocab, repeats per stacked block, and adds the embedding/head/norm scaffold the
    harness wraps around it — matching the shape the screener trained on.
    """

    module = generate_module_from_spec(spec, dim=dim)
    lane_ops: list[str] = []
    for sub in module.modules():
        op = _FAB_CLASS_TO_OP.get(type(sub).__name__)
        if op is not None:
            lane_ops.append(op)
    full = lane_ops * max(1, n_blocks)
    full.append("embedding_lookup")
    full.extend(["layernorm"] * (n_blocks + 1))
    full.append("linear_proj")  # tied lm head
    return full


def capability_screen_for_spec(
    spec: ProposalSpec,
    *,
    dim: int = 32,
    n_blocks: int = 2,
    models: dict[str, Any] | None = None,
) -> CapabilityScreen:
    """Score a fab spec with the induction + nano capability screeners.

    ``models`` optionally carries pre-loaded ``{"induction": (model, vocab),
    "nano": (model, vocab)}`` to avoid reloading per spec.
    """

    try:
        loaded = models or load_capability_screeners()
        ops = fab_op_multiset(spec, dim=dim, n_blocks=n_blocks)
        op_set = set(ops)
        op_count = len(ops)
        pair_count = max(0, op_count - 1)
        preds: dict[str, float] = {}
        for key, (model, vocab) in loaded.items():
            from research.tools.capability_screener import featurize_op_sets

            x = featurize_op_sets([op_set], [op_count], [pair_count], list(vocab))
            preds[key] = float(model.predict(x)[0])
        return CapabilityScreen(
            proposal_id=spec.proposal_id,
            available=True,
            induction_pred=preds.get("induction", 0.0),
            nano_pred=preds.get("nano", 0.0),
            op_count=op_count,
            n_distinct_ops=len(op_set),
        )
    except Exception as exc:  # noqa: BLE001 - screen is best-effort, fail open
        logger.debug("capability screen unavailable for %s: %s", spec.proposal_id, exc)
        return CapabilityScreen(
            proposal_id=spec.proposal_id,
            available=False,
            induction_pred=0.0,
            nano_pred=0.0,
            op_count=0,
            n_distinct_ops=0,
            reason=str(exc),
        )


def load_capability_screeners() -> dict[str, Any]:
    """Load the induction + nano screeners as ``{key: (model, op_vocab)}``."""

    from research.tools.capability_screener import load_screener

    out: dict[str, Any] = {}
    for key, state_dir in (("induction", _INDUCTION_DIR), ("nano", _NANO_DIR)):
        if not state_dir.exists():
            continue
        model, meta = load_screener(state_dir)
        out[key] = (model, list(meta["op_vocab"]))
    if not out:
        raise FileNotFoundError("no capability screeners found on disk")
    return out
