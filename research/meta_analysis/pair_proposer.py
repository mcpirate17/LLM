"""Surface untapped stable pair compositions for grammar exploration.

A "new math" wedge: the profiler has measured 5,979 op-pair compositions in
``op_pair_profile_catalog``. ~2,200 are healthy (no NaN, healthy gradient,
non-zero output). But only ~3,100 unique pair signatures appear in
``program_graph_pairs`` — programs the grammar has actually built.

The set difference — pairs the profiler measured as stable but the grammar
has never composed — is the system's surface of "untried but viable" math.
Surfacing them as motif candidates lets the grammar explore compositions no
one (human or earlier-generation grammar) has assembled.

This module is read-only. Wiring the candidates into the motif catalog or
template registry is the next phase; do not auto-promote untapped pairs into
generation without holdout evidence.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ar_binding_overlay import overlay_for_pair
from .priors import _connect_readonly

_HEALTHY_PAIR_FILTER = (
    "output_has_nan = 0 "
    "AND grad_has_nan = 0 "
    "AND grad_vanishing = 0 "
    "AND grad_exploding = 0 "
    "AND output_std > ?"
)


def _observed_pair_signatures(runs_conn: sqlite3.Connection) -> set[str]:
    """Distinct pair signatures (``a->b``) recorded in successful programs."""
    rows = runs_conn.execute(
        "SELECT DISTINCT signature FROM program_graph_pairs"
    ).fetchall()
    return {row[0] for row in rows if row[0]}


def _stability_score(row: sqlite3.Row) -> float:
    """Lower-is-better scalar combining gradient health and Lipschitz tightness.

    Tight Lipschitz estimates and balanced gradient norms produce small scores.
    Used only as a tie-breaker when emitting top-K candidates; the grammar
    decides final acceptance through downstream evaluation.
    """
    grad_norm = float(row["grad_norm"] or 0.0)
    lipschitz = float(row["lipschitz_estimate"] or 0.0)
    grad_term = abs(grad_norm - 1.0) if grad_norm > 0 else 1.0
    lip_term = lipschitz if lipschitz > 0 else 0.5
    return grad_term + lip_term


def propose_untapped_pairs(
    meta_db_path: str | Path,
    runs_db_path: str | Path,
    *,
    composition: str = "sequential",
    limit: int = 50,
    min_output_std: float = 1e-4,
    include_ar_binding_overlay: bool = True,
) -> List[Dict[str, Any]]:
    """Return ranked stable pair compositions never assembled in real programs.

    Args:
        meta_db_path: path to ``meta_analysis.db`` (must contain
            ``op_pair_profile_catalog``).
        runs_db_path: path to ``runs.db`` (must contain ``program_graph_pairs``).
        composition: which topology to consider (``sequential`` or ``residual``).
            ``sequential`` is what the ``program_graph_pairs.signature`` set
            represents, so the diff is meaningful only for that case.
        limit: max candidates to return.
        min_output_std: filter out near-zero-output pairs (often hidden NaNs).
        include_ar_binding_overlay: annotate emitted candidates with the shared
            AR/binding overlay. This is advisory and does not affect ordering.

    Returns:
        List of candidate dicts ordered by stability_score ascending.
    """
    meta_conn = _connect_readonly(meta_db_path)
    runs_conn = _connect_readonly(runs_db_path)
    try:
        observed = (
            _observed_pair_signatures(runs_conn)
            if composition == "sequential"
            else set()
        )
        rows = meta_conn.execute(
            f"""
            SELECT op_a, op_b, composition, output_std, grad_norm,
                   lipschitz_estimate, jacobian_spectral_norm,
                   stability_delta, distribution_shift
            FROM op_pair_profile_catalog
            WHERE composition = ?
              AND {_HEALTHY_PAIR_FILTER}
            """,
            (composition, float(min_output_std)),
        ).fetchall()
    finally:
        meta_conn.close()
        runs_conn.close()

    candidates: List[Dict[str, Any]] = []
    for row in rows:
        signature = f"{row['op_a']}->{row['op_b']}"
        is_observed = composition == "sequential" and signature in observed
        if is_observed:
            continue
        candidates.append(
            {
                "op_a": row["op_a"],
                "op_b": row["op_b"],
                "composition": row["composition"],
                "signature": signature,
                "output_std": float(row["output_std"] or 0.0),
                "grad_norm": float(row["grad_norm"] or 0.0),
                "lipschitz_estimate": _optional_float(row["lipschitz_estimate"]),
                "jacobian_spectral_norm": _optional_float(
                    row["jacobian_spectral_norm"]
                ),
                "stability_delta": _optional_float(row["stability_delta"]),
                "distribution_shift": _optional_float(row["distribution_shift"]),
                "stability_score": _stability_score(row),
                "novelty": "fully_untapped",
            }
        )

    candidates.sort(key=lambda d: d["stability_score"])
    emitted = candidates[:limit]
    if include_ar_binding_overlay:
        for candidate in emitted:
            candidate["ar_binding_overlay"] = overlay_for_pair(
                str(candidate["op_a"]),
                str(candidate["op_b"]),
                meta_db_path=meta_db_path,
            )
    return emitted


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # filter NaN
