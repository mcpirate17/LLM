"""Cross-pollinate component_fab inventions into the NAS grammar as motifs.

The diversity-generator charter's cross-pollination unlock
(``research/notes/diversity_generator_charter_2026-06-03.md``, M2): register the
semantic, literature-grounded ``component_fab`` inventions as grammar motifs so
the *volume* engine composes them into novel topologies the fab inventor would
never enumerate — and so each population's collapse is broken by the other's
primitives (tropical/semiring components composed into non-tropical topologies).

Mirrors ``_motifs_mined.py``: env-gated (``ARIA_ENABLE_FAB_MOTIFS``), folded into
a dedicated ``fab_invention`` motif_class disjoint from the validated catalog so
it never silently replaces a human-designed motif. Unlike the mined pairs these
carry REAL project evidence (seed lifts below the validated top, above the
sampling floor), and they are reachable through the wildcard exploration path
(``fab_invention`` is added to ``_template_helpers._ALL_CLASSES``) plus explicit
``GrammarConfig.exploration_targets`` (M4).

Each invention lowers to the canonical, context-legal ``mixer -> linear_proj``
attention-motif shape; ops absent from the primitive registry or whose chain
violates a context rule are skipped (fail-safe, never emit an illegal motif).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List

from ._motif_types import MOTIF_CLASS_FAB, Motif, MotifStep
from ._selection_utils import context_pair_allowed
from .op_roles import OpRole, get_role
from .primitives import PRIMITIVE_REGISTRY

logger = logging.getLogger(__name__)

_ENV_FLAG = "ARIA_ENABLE_FAB_MOTIFS"
_TAIL_OP = "linear_proj"


@dataclass(frozen=True, slots=True)
class _FabInvention:
    """A component_fab invention's characteristic op + its evidence-seeded lift."""

    name: str
    op: str
    lift: float
    evidence: str


# Curated from the project's confirmed signal (see memory: STDP-attention
# e656938e induction 0.894 = the ONLY confirmed novel learner; learnable-semiring
# bAbI 0.88; reciprocal/tropical family). Seed lifts are evidence-tiered and kept
# modest (<= validated-top ~3.5) — these are SEEDS to be proven downstream, not
# pre-blessed winners. The Tier-2 suite is associative recall/binding, so the set
# targets the recall-biased niches softmax+Mamba lack: hard-selection pointer
# copy, delta-rule memory, and explicit slot/KV memory.
_FAB_INVENTIONS: tuple[_FabInvention, ...] = (
    # ── confirmed novel learners ─────────────────────────────────────
    _FabInvention(
        "fab_stdp_attention",
        "stdp_attention",
        1.6,
        "confirmed novel learner e656938e (induction 0.894, 5/5 seeds)",
    ),
    _FabInvention(
        "fab_learnable_semiring",
        "learnable_semiring_attention",
        1.6,
        "differentiable hard-selection binder (bAbI 0.88)",
    ),
    # ── hard-selection / pointer-copy family (the project's unique edge) ──
    _FabInvention(
        "fab_tropical_hard_select",
        "tropical_attention",
        1.3,
        "max-plus exact pointer copy — what binding wants",
    ),
    _FabInvention(
        "fab_ultrametric",
        "ultrametric_attention",
        1.3,
        "ultrametric reciprocal-family attention",
    ),
    # ── delta-rule memory (SOTA associative recall; hybridize to beat Mamba2) ──
    _FabInvention(
        "fab_gated_delta",
        "gated_delta",
        1.3,
        "gated delta-rule error-correcting KV memory",
    ),
    _FabInvention(
        "fab_dplr_gated_delta",
        "dplr_gated_delta",
        1.3,
        "DPLR gated delta-rule memory",
    ),
    _FabInvention(
        "fab_retention",
        "retention_mix",
        1.1,
        "gated linear recurrence (global recall partner for local attn)",
    ),
    _FabInvention(
        "fab_local_window",
        "local_window_attn",
        1.1,
        "local exact attention — hybrid partner for delta-rule recurrence",
    ),
    # ── explicit slot / addressable KV memory (under-used in vocab) ──
    _FabInvention(
        "fab_product_key_memory",
        "product_key_memory",
        1.1,
        "addressable product-key memory slots",
    ),
    _FabInvention(
        "fab_role_slot_attention",
        "role_slot_attention",
        1.1,
        "role-addressed slot attention (binding-shaped)",
    ),
    _FabInvention(
        "fab_associative_memory",
        "associative_memory",
        1.1,
        "explicit associative memory read",
    ),
)


def _make_fab_motif(inv: _FabInvention) -> Motif | None:
    """Lower one invention to a context-legal ``mixer -> linear_proj`` motif.

    Returns None (skip) when the op is unknown, not arity-1, role-unsafe, or the
    chain violates a context rule — we never register an illegal motif.
    """

    prim = PRIMITIVE_REGISTRY.get(inv.op)
    tail = PRIMITIVE_REGISTRY.get(_TAIL_OP)
    if prim is None or tail is None or prim.n_inputs != 1:
        return None
    role = get_role(inv.op)
    if role is OpRole.UNSAFE:
        return None
    if not context_pair_allowed(inv.op, _TAIL_OP):
        return None
    return Motif(
        name=inv.name,
        motif_class=MOTIF_CLASS_FAB,
        steps=(
            MotifStep(op_name=inv.op, role=role),
            MotifStep(op_name=_TAIL_OP, role=get_role(_TAIL_OP)),
        ),
        description=f"component_fab invention: {inv.evidence}",
        support=0,
        avg_loss_ratio=0.0,
        lift=inv.lift,
    )


def register_fab_motifs(
    validated_motifs: Dict[str, Motif],
    motifs_by_class: Dict[str, List[Motif]],
    *,
    enable: bool | None = None,
) -> List[str]:
    """Merge component_fab invention motifs into the catalog under ``fab_invention``.

    Args mirror ``register_mined_motifs``. ``enable=None`` falls back to the
    ``ARIA_ENABLE_FAB_MOTIFS`` env flag (default off). Returns registered names.
    """

    if enable is None:
        enable = os.environ.get(_ENV_FLAG, "0") not in ("", "0", "false", "False")
    if not enable:
        return []

    registered: List[str] = []
    bucket = motifs_by_class.setdefault(MOTIF_CLASS_FAB, [])
    for inv in _FAB_INVENTIONS:
        if inv.name in validated_motifs:
            continue
        motif = _make_fab_motif(inv)
        if motif is None:
            logger.debug(
                "skipping fab invention %s (op %s unbuildable)", inv.name, inv.op
            )
            continue
        validated_motifs[inv.name] = motif
        bucket.append(motif)
        registered.append(inv.name)

    if registered:
        logger.info(
            "Registered %d component_fab invention motifs under class '%s'",
            len(registered),
            MOTIF_CLASS_FAB,
        )
    return registered


def fab_invention_ops() -> tuple[str, ...]:
    """The characteristic op of every registered fab invention (M4 vocabulary).

    Archive-guided generation (``archive_guided``) targets these ops to refill
    empty behavior niches, so it reads the op set from here rather than
    re-listing it — the two stay in sync by construction.
    """

    return tuple(inv.op for inv in _FAB_INVENTIONS)


__all__ = ("register_fab_motifs", "fab_invention_ops", "MOTIF_CLASS_FAB")
