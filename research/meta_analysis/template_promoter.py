"""Promote mined subgraph chains to a persistent template candidate registry.

The mining tool ``research/tools/mine_template_subpatterns_v2.py`` emits
``research/reports/mined_novel_chain_proposals.json`` listing chains that
appear in passing programs and are not produced by any existing template.
That report is auto-pruned with the rest of ``reports/`` and has no live
consumer.

This promoter applies hard thresholds (support, lift, pass-rate), dedupes
against the existing ``TEMPLATES`` registry, and writes a stable registry
file under ``research/notes/`` that the grammar (or a human reviewer) can
load. Promotion is advisory — it does NOT auto-register with TEMPLATES.
The next phase wires the registry into grammar weights or compiles the
``code_skeleton`` into actual template callables behind a feature flag.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .ar_binding_overlay import overlay_for_chain
from .metadata_db import DEFAULT_META_ANALYSIS_DB


_DEFAULT_MIN_N_TOTAL = 5
_DEFAULT_MIN_LIFT = 1.20
_DEFAULT_MIN_PASS_RATE = 0.30
_DEFAULT_TOP_K = 25


def _normalize_chain(chain: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(op) for op in chain)


def _candidate_passes(
    record: Dict[str, Any],
    *,
    min_n_total: int,
    min_lift: float,
    min_pass_rate: float,
) -> bool:
    if int(record.get("n_total", 0)) < min_n_total:
        return False
    if float(record.get("lift_vs_cohort", 0.0)) < min_lift:
        return False
    if float(record.get("pass_rate", 0.0)) < min_pass_rate:
        return False
    return True


def _annotate_promotion_score(record: Dict[str, Any]) -> float:
    """Higher = more confident the chain deserves a dedicated template.

    Combines support (sqrt to dampen large-N dominance), lift over cohort,
    and absolute pass-rate. No probabilistic model — this is a triage
    ordering, not a calibrated prediction.
    """
    n_total = float(record.get("n_total", 0))
    lift = float(record.get("lift_vs_cohort", 0.0))
    pass_rate = float(record.get("pass_rate", 0.0))
    return (n_total**0.5) * lift * pass_rate


def promote_mined_chains(
    report_path: str | Path,
    *,
    existing_template_names: Sequence[str] | None = None,
    min_n_total: int = _DEFAULT_MIN_N_TOTAL,
    min_lift: float = _DEFAULT_MIN_LIFT,
    min_pass_rate: float = _DEFAULT_MIN_PASS_RATE,
    top_k: int = _DEFAULT_TOP_K,
    include_rare: bool = False,
    include_ar_binding_overlay: bool = False,
    meta_db_path: str | Path = DEFAULT_META_ANALYSIS_DB,
) -> List[Dict[str, Any]]:
    """Filter and rank mined chains for promotion to template candidates.

    Args:
        report_path: path to ``mined_novel_chain_proposals.json`` from V2 miner.
        existing_template_names: names already registered in ``TEMPLATES``;
            mined chains whose ``proposed_template_name`` collides with one of
            these are skipped.
        min_n_total: minimum support (graphs containing the chain).
        min_lift: minimum lift over cohort baseline pass rate.
        min_pass_rate: minimum absolute pass rate of graphs containing the chain.
        top_k: cap on returned candidates.
        include_rare: include "rare" candidates (1-2 templates already produce
            the chain). Default False — only fully-novel chains promote.
        include_ar_binding_overlay: annotate emitted candidates with the shared
            AR/binding overlay. This is advisory and does not affect ordering.
        meta_db_path: meta-analysis DB used by the overlay when annotation is
            enabled.

    Returns:
        Ranked list of promotion candidates with promotion_score and the
        original mining metadata. Each candidate dict is JSON-safe.
    """
    path = Path(report_path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))

    pool = list(payload.get("novel_candidates") or [])
    if include_rare:
        pool.extend(payload.get("rare_candidates") or [])

    existing_names = set(existing_template_names or ())
    promoted: List[Dict[str, Any]] = []
    for record in pool:
        if not _candidate_passes(
            record,
            min_n_total=min_n_total,
            min_lift=min_lift,
            min_pass_rate=min_pass_rate,
        ):
            continue
        proposed_name = str(record.get("proposed_template_name") or "")
        if proposed_name and proposed_name in existing_names:
            continue
        chain = _normalize_chain(record.get("chain") or ())
        promoted.append(
            {
                "proposed_template_name": proposed_name,
                "chain": list(chain),
                "chain_length": int(record.get("length", len(chain))),
                "anchor_op": str(record.get("anchor_op") or ""),
                "n_total": int(record.get("n_total", 0)),
                "n_pass": int(record.get("n_pass", 0)),
                "pass_rate": float(record.get("pass_rate", 0.0)),
                "lift_vs_cohort": float(record.get("lift_vs_cohort", 0.0)),
                "covered_by_templates": list(record.get("covered_by_templates") or []),
                "code_skeleton": record.get("code_skeleton"),
                "promotion_score": _annotate_promotion_score(record),
                "promotion_thresholds": {
                    "min_n_total": min_n_total,
                    "min_lift": min_lift,
                    "min_pass_rate": min_pass_rate,
                },
            }
        )

    promoted.sort(key=lambda d: -d["promotion_score"])
    emitted = promoted[:top_k]
    if include_ar_binding_overlay:
        for candidate in emitted:
            candidate["ar_binding_overlay"] = overlay_for_chain(
                candidate["chain"], meta_db_path=meta_db_path
            )
    return emitted


def write_promotion_registry(
    candidates: List[Dict[str, Any]],
    output_path: str | Path,
    *,
    metadata: Dict[str, Any] | None = None,
) -> Path:
    """Persist a promotion registry alongside the lineage metadata."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "count": len(candidates),
        "candidates": candidates,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
