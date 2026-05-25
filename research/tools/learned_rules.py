#!/usr/bin/env python
"""Learned failure-risk + good-template scoring — CPU, no torch.

Consumes the data-grounded rules mined by `mine_failure_rules.py` and the structural rules validated
across this project, exposing two cheap functions for the cascade / any consumer:

  - graph_failure_risk(nodes) -> per-mode probability the graph dies of compile / lookahead /
    instability, estimated from the empirical failure rates of the ops it contains.
  - score_template_quality(nodes) -> a "good template" checklist (mixer on path, mixer_depth>=2,
    normalization, residual, no double-gating, low failure risk) → score in [0,1] + per-check reasons.

Hard structural rules (forbidden prev/next pairs, etc.) remain in
`synthesis._context_validation.find_graph_context_violations`; this module is the SOFT, empirical,
support-weighted layer on top.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from research.synthesis._context_op_sets import _GATING_OPS
from research.synthesis.op_roles import OpRole, get_role
from research.tools.static_capability_gate import mixer_chain_depth, on_path_op_names

_RULES_PATH = Path("research/data/learned_failure_rules/rules.json")
_MODES = ("compile", "lookahead", "instability", "resource")


@lru_cache(maxsize=1)
def _rules() -> Dict[str, Any]:
    if not _RULES_PATH.exists():
        return {"rules": {m: {} for m in _MODES}, "base_rate": {m: 0.0 for m in _MODES}}
    return json.loads(_RULES_PATH.read_text())


def risky_ops(mode: str) -> Dict[str, float]:
    """{op_name: empirical failure rate} for ``mode`` (compile|lookahead|instability)."""
    return {op: d["rate"] for op, d in _rules()["rules"].get(mode, {}).items()}


@lru_cache(maxsize=1)
def _unstable_triplets() -> frozenset:
    """Frozenset of (a,b,c) op chains measured numerically unstable as a true 3-op interaction."""
    return frozenset(tuple(t) for t in _rules().get("unstable_triplets", []))


def unstable_triplets_in_graph(nodes: Any) -> List[tuple]:
    """Direct a→b→c op chains in the graph that match a mined unstable triplet."""
    nl = _node_list(nodes)
    by_id = {n["id"]: str(n["op_name"]) for n in nl}
    bad = _unstable_triplets()
    hits = []
    for c in nl:
        if c.get("is_input"):
            continue
        for bid in c.get("input_ids", []) or []:
            if bid not in by_id:
                continue
            b_node = next((n for n in nl if n["id"] == bid), None)
            for aid in (b_node.get("input_ids", []) or []) if b_node else []:
                if aid in by_id:
                    tri = (by_id[aid], by_id[bid], str(c["op_name"]))
                    if tri in bad:
                        hits.append(tri)
    return hits


def _node_list(nodes: Any) -> List[Any]:
    return list(nodes.values()) if isinstance(nodes, dict) else list(nodes)


def graph_failure_risk(nodes: Any) -> Dict[str, float]:
    """Per-mode P(graph dies of mode) ≈ 1 − Π(1 − rate_op) over ops present (independent-op model)."""
    ops = [str(n["op_name"]) for n in _node_list(nodes) if not n.get("is_input")]
    out: Dict[str, float] = {}
    for mode in _MODES:
        rates = risky_ops(mode)
        surv = 1.0
        for op in ops:
            if op in rates:
                surv *= 1.0 - rates[op]
        out[mode] = round(1.0 - surv, 4)
    return out


def score_template_quality(nodes: Any) -> Dict[str, Any]:
    """Good-template checklist → score in [0,1] + reasons. Higher = more likely a trainable, capable,
    failure-free template."""
    nl = _node_list(nodes)
    ops = [str(n["op_name"]) for n in nl if not n.get("is_input")]
    on_path = on_path_op_names(nodes)
    n_mix = sum(1 for op in on_path if get_role(op) is OpRole.MIX)
    depth = mixer_chain_depth(nodes)
    has_norm = any(get_role(op) is OpRole.NORMALIZE for op in ops)
    has_resid = any(
        len(n.get("input_ids", []) or []) > 1
        or get_role(str(n["op_name"])) is OpRole.RESIDUAL
        for n in nl
        if not n.get("is_input")
    )
    double_gate = _has_double_gating(nl)
    risk = graph_failure_risk(nodes)
    bad_triplets = unstable_triplets_in_graph(nodes)
    checks = {
        "has_mixer_on_path": n_mix >= 1,  # MUST — cross-position skill needs a mixer
        "mixer_depth_ge_2": depth >= 2,  # GOOD — induction circuit (88% of capable)
        "has_normalization": has_norm,  # MUST — trainability
        "has_residual": has_resid,  # MUST — gradient flow / residual context
        "no_double_gating": not double_gate,  # MUST — double-gating = ~100% fail (encoded rule)
        "no_unstable_triplet": not bad_triplets,  # GOOD — mined 3-op instability
        "low_compile_risk": risk["compile"] < 0.25,
        "low_lookahead_risk": risk["lookahead"] < 0.20,
        "low_instability_risk": risk["instability"] < 0.20,
        "low_resource_risk": risk["resource"]
        < 0.20,  # OOM/cuda_fatal (wastes a full GPU run)
    }
    # MUST checks gate hard (score 0 if any fails); GOOD + risk checks are weighted.
    must = (
        "has_mixer_on_path",
        "has_normalization",
        "has_residual",
        "no_double_gating",
    )
    weighted = [k for k in checks if k not in must]
    if not all(checks[k] for k in must):
        score = 0.0
    else:
        score = round(sum(1.0 for k in weighted if checks[k]) / len(weighted), 3)
    return {
        "score": score,
        "passes_must": all(checks[k] for k in must),
        "mixer_depth": depth,
        "n_mixers_on_path": n_mix,
        "failure_risk": risk,
        "checks": checks,
        "reasons": [k for k, ok in checks.items() if not ok],
    }


def _has_double_gating(node_list: List[Any]) -> bool:
    """True iff a gating op is directly fed by another gating op (conflicting gradients ⇒ ~100% fail)."""
    by_id = {n["id"]: str(n["op_name"]) for n in node_list}
    for n in node_list:
        if n.get("is_input") or str(n["op_name"]) not in _GATING_OPS:
            continue
        for src in n.get("input_ids", []) or []:
            if by_id.get(src) in _GATING_OPS:
                return True
    return False
