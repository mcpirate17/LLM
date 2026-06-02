"""Auto-register untapped stable pair compositions as mined motifs.

The ``pair_proposer`` surfaces pairs the profiler measured as stable but
that the grammar has never assembled. When ``ARIA_ENABLE_MINED_MOTIFS`` is
set, this module materializes the top-K candidates into ``Motif`` objects
and folds them into the motif catalog under the ``mined_pair`` motif_class.

Mined motifs are weighted below the validated catalog (lift=0.5) so they
never crowd out human-designed motifs until they prove themselves through
the existing screening pipeline. The motif_class ``mined_pair`` is
deliberately disjoint from the existing classes so the grammar never picks
mined motifs as drop-in replacements — they only appear when a template
explicitly requests the ``mined_pair`` slot class.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from ._motif_types import Motif, MotifStep
from .op_roles import OpRole, get_role
from .primitives import PRIMITIVE_REGISTRY

logger = logging.getLogger(__name__)

_ENV_FLAG = "ARIA_ENABLE_MINED_MOTIFS"
_DEFAULT_PAIR_PATH = Path(
    "research/data/synthesis_candidates/untapped_pair_proposals.json"
)
_MINED_MOTIF_CLASS = "mined_pair"
_DEFAULT_LIFT = 0.5
_MAX_MINED_MOTIFS = 48


def _make_pair_motif(name: str, op_a: str, op_b: str, lift: float) -> Motif | None:
    """Construct a ``Motif`` from two arity-1 ops. Returns None if either op
    is unknown to the primitive registry or has incompatible arity."""
    prim_a = PRIMITIVE_REGISTRY.get(op_a)
    prim_b = PRIMITIVE_REGISTRY.get(op_b)
    if prim_a is None or prim_b is None:
        return None
    if prim_a.n_inputs != 1 or prim_b.n_inputs != 1:
        return None
    role_a = get_role(op_a)
    role_b = get_role(op_b)
    if role_a is OpRole.UNSAFE or role_b is OpRole.UNSAFE:
        return None
    return Motif(
        name=name,
        motif_class=_MINED_MOTIF_CLASS,
        steps=(
            MotifStep(op_name=op_a, role=role_a),
            MotifStep(op_name=op_b, role=role_b),
        ),
        description=f"untapped stable pair {op_a}->{op_b}",
        support=0,
        avg_loss_ratio=0.0,
        lift=lift,
    )


def _load_pair_proposals(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [c for c in (payload.get("candidates") or []) if isinstance(c, dict)]


def register_mined_motifs(
    validated_motifs: Dict[str, Motif],
    motifs_by_class: Dict[str, List[Motif]],
    *,
    json_path: str | Path | None = None,
    enable: bool | None = None,
    max_register: int = _MAX_MINED_MOTIFS,
    default_lift: float = _DEFAULT_LIFT,
) -> List[str]:
    """Merge mined motifs into the existing catalog.

    Args:
        validated_motifs: target name→Motif dict (typically
            ``synthesis._motif_selection.VALIDATED_MOTIFS``).
        motifs_by_class: target class→[Motif] dict
            (``synthesis._motif_selection.MOTIFS_BY_CLASS``).
        json_path: untapped-pair proposals JSON; defaults to the notes path.
        enable: explicit override; when None, falls back to the env flag.
        max_register: hard cap on number of mined motifs registered.
        default_lift: lift assigned to mined motifs (kept below 1.0 to
            avoid crowding out validated motifs).

    Returns:
        Registered motif names.
    """
    if enable is None:
        enable = os.environ.get(_ENV_FLAG, "0") not in ("", "0", "false", "False")
    if not enable:
        return []

    path = Path(json_path) if json_path else _DEFAULT_PAIR_PATH
    proposals = _load_pair_proposals(path)
    if not proposals:
        return []

    registered: List[str] = []
    bucket = motifs_by_class.setdefault(_MINED_MOTIF_CLASS, [])
    for proposal in proposals:
        if len(registered) >= max_register:
            break
        op_a = str(proposal.get("op_a") or "").strip()
        op_b = str(proposal.get("op_b") or "").strip()
        composition = str(proposal.get("composition") or "").strip()
        if not op_a or not op_b or composition != "sequential":
            continue
        motif_name = f"mined_{op_a}_then_{op_b}"
        if motif_name in validated_motifs:
            continue
        motif = _make_pair_motif(motif_name, op_a, op_b, default_lift)
        if motif is None:
            continue
        validated_motifs[motif_name] = motif
        bucket.append(motif)
        registered.append(motif_name)

    if registered:
        logger.info(
            "Registered %d mined motifs from %s under class '%s'",
            len(registered),
            path,
            _MINED_MOTIF_CLASS,
        )
    return registered


__all__ = ("register_mined_motifs", "_MINED_MOTIF_CLASS")
