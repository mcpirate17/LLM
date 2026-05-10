"""Derive a do-not-compose blocklist from emergent-instability triplets.

The pair profiler at ``op_pair_profile_catalog`` measures component stability.
The triplet profiler at ``op_triplet_profile_catalog`` measures three-op
compositions. ~80 triplets show ``triplet_stable=0`` despite both constituent
pairs predicting stable — emergent instability that pairwise analysis cannot
catch. These are concrete "do not compose this triple" rules backed by
measurement, not heuristic.

The blocklist is read-only output. Wiring it into the grammar's failure
signatures or pre-validation is a follow-up; this module just surfaces the
candidates with provenance.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .priors import _connect_readonly


def _classify_failure(row: sqlite3.Row) -> str:
    """Map measured failure mode to a categorical reason string."""
    if int(row["output_has_nan"] or 0):
        return "output_has_nan"
    if int(row["grad_has_nan"] or 0):
        return "grad_has_nan"
    if int(row["grad_exploding"] or 0):
        return "grad_exploding"
    if int(row["grad_vanishing"] or 0):
        return "grad_vanishing"
    if float(row["output_std"] or 0.0) < 1e-12:
        return "output_collapsed"
    return "unstable"


def derive_triplet_blocklist(
    meta_db_path: str | Path,
    *,
    require_pair_predicted_stable: bool = True,
) -> List[Dict[str, Any]]:
    """Return triplets the profiler measured as unstable.

    Args:
        meta_db_path: path to ``meta_analysis.db``.
        require_pair_predicted_stable: if True (default), restrict to triplets
            where pair-level prediction said both pairs were stable — i.e.
            emergent instability that pair-level analysis cannot detect. Set
            False to include every measured-unstable triplet regardless of
            pair predictions.

    Returns:
        List of blocked-triplet records ordered by op_a, op_b, op_c.
    """
    conn = _connect_readonly(meta_db_path)
    try:
        sql = """
            SELECT op_a, op_b, op_c,
                   output_std, output_has_nan,
                   grad_norm, grad_has_nan, grad_vanishing, grad_exploding,
                   pair_ab_predicted_stable, pair_bc_predicted_stable
            FROM op_triplet_profile_catalog
            WHERE triplet_stable = 0
        """
        if require_pair_predicted_stable:
            sql += " AND pair_ab_predicted_stable = 1 AND pair_bc_predicted_stable = 1"
        sql += " ORDER BY op_a, op_b, op_c"
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()

    blocklist: List[Dict[str, Any]] = []
    for row in rows:
        blocklist.append(
            {
                "op_a": row["op_a"],
                "op_b": row["op_b"],
                "op_c": row["op_c"],
                "signature": f"{row['op_a']}->{row['op_b']}->{row['op_c']}",
                "reason": _classify_failure(row),
                "output_std": _optional_float(row["output_std"]),
                "grad_norm": _optional_float(row["grad_norm"]),
                "pair_ab_predicted_stable": int(row["pair_ab_predicted_stable"] or 0),
                "pair_bc_predicted_stable": int(row["pair_bc_predicted_stable"] or 0),
                "emergent": (
                    int(row["pair_ab_predicted_stable"] or 0) == 1
                    and int(row["pair_bc_predicted_stable"] or 0) == 1
                ),
            }
        )
    return blocklist


def blocked_triplet_set(blocklist: List[Dict[str, Any]]) -> set[tuple[str, str, str]]:
    """Convenience: extract the (op_a, op_b, op_c) set for grammar lookup.

    The grammar can use this to reject candidate graphs that contain any
    blocked triple as a 3-op chain. Lookup is O(1) per triple.
    """
    return {(row["op_a"], row["op_b"], row["op_c"]) for row in blocklist}


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None
