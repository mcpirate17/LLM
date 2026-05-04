"""Multi-label op composition classifier for `program_graph_features.motifs_json`.

Codex's `real_lm_quickcheck_audit` classifies each row into a single family by
substring on `template_name`.  That undercounts hybrid templates: e.g.
``latent_attn_ssm_hybrid`` is "ssm_recurrent" by codex's classifier but
contains both attention and SSM motifs.  This module classifies on the actual
motifs the graph uses, returning a multi-label set.

Pure-Python, no GPU, no DB writes.  Designed to be imported by audit tools or
queried inline (e.g. dashboard filters, CLI ranking helpers).
"""

from __future__ import annotations

import json
from typing import Iterable, Sequence

# Substring rules.  Order matters only for ``primary_family``.
_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("attention", ("attn_", "attention")),
    ("ssm", ("ssm_", "scan", "mamba")),
    ("recurrent", ("recurrent", "cumprod", "decay_")),
    ("routing", ("route_", "moe_", "router", "conditional_skip")),
    ("conv", ("conv_",)),
    ("gate", ("gate_",)),
    ("sparse", ("sparse",)),
    ("compress", ("compress", "bottleneck", "merge", "codebook")),
    ("retrieval", ("retrieval", "topk_retrieval")),
    ("norm", ("norm_",)),
    ("padic", ("padic", "hierarchy")),
    ("tropical", ("tropical",)),
)

OP_LABELS: tuple[str, ...] = tuple(name for name, _ in _RULES)


def classify_motifs(motifs: Sequence[str] | str | None) -> dict[str, bool]:
    """Return a bool per label in :data:`OP_LABELS` indicating whether any
    motif in ``motifs`` matches that label's substring rule.

    Accepts either a parsed list of motif strings or a raw JSON-encoded
    string straight from ``program_graph_features.motifs_json``.
    """
    if motifs is None:
        return {label: False for label in OP_LABELS}
    if isinstance(motifs, str):
        try:
            motifs = json.loads(motifs)
        except (TypeError, ValueError):
            return {label: False for label in OP_LABELS}
    out = {label: False for label in OP_LABELS}
    for motif in motifs or ():
        m = str(motif).lower()
        for label, patterns in _RULES:
            if out[label]:
                continue
            if any(p in m for p in patterns):
                out[label] = True
    return out


def primary_family(flags: dict[str, bool]) -> str:
    """Pick a single "primary" family for back-compat with single-label
    consumers.  Order matches :data:`_RULES` priority (attention > ssm >
    recurrent > routing > conv > ...)."""
    for label in OP_LABELS:
        if flags.get(label):
            return label
    return "none"


def has_any(flags: dict[str, bool], wanted: Iterable[str]) -> bool:
    return any(flags.get(label) for label in wanted)


__all__ = [
    "OP_LABELS",
    "classify_motifs",
    "primary_family",
    "has_any",
]
