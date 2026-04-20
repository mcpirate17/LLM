from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple


logger = logging.getLogger(__name__)


def _fetch_rows(
    profiling_db: Path,
    sql: str,
    parameters: Sequence[Any] = (),
    *,
    row_factory: Any = None,
) -> list[Any]:
    if not profiling_db.exists():
        return []
    try:
        conn = sqlite3.connect(str(profiling_db), timeout=5)
        if row_factory is not None:
            conn.row_factory = row_factory
        try:
            return conn.execute(sql, tuple(parameters)).fetchall()
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("Failed profiling DB query for %s: %s", profiling_db, exc)
        return []


def load_op_feature_rows(
    profiling_db: Path,
    feature_names: Sequence[str],
) -> list[sqlite3.Row]:
    columns = ", ".join(str(name) for name in feature_names)
    return _fetch_rows(
        profiling_db,
        f"SELECT op_name, {columns} FROM op_profiles WHERE error IS NULL ORDER BY op_name",
        row_factory=sqlite3.Row,
    )


def load_op_profiles(profiling_db: Path) -> Dict[str, Dict[str, float]]:
    rows = _fetch_rows(
        profiling_db,
        """SELECT op_name, output_std, grad_norm, lipschitz_estimate,
                  grad_vanishing, grad_exploding, output_has_nan, has_params
           FROM op_profiles WHERE error IS NULL""",
    )
    profiles: Dict[str, Dict[str, float]] = {}
    for (
        op,
        out_std,
        grad_norm,
        lipschitz,
        grad_vanishing,
        grad_exploding,
        has_nan,
        has_params,
    ) in rows:
        profiles[str(op)] = {
            "output_std": float(out_std) if out_std else 1.0,
            "grad_norm": float(grad_norm) if grad_norm else 1.0,
            "lipschitz": float(lipschitz) if lipschitz else 1.0,
            "grad_vanishing": float(grad_vanishing or 0.0),
            "grad_exploding": float(grad_exploding or 0.0),
            "has_nan": float(has_nan or 0.0),
            "has_params": float(has_params or 0.0),
        }
    return profiles


def load_op_categories(profiling_db: Path) -> Dict[str, str]:
    rows = _fetch_rows(
        profiling_db,
        "SELECT op_name, category FROM op_profiles WHERE category IS NOT NULL",
    )
    return {str(op_name): str(category) for op_name, category in rows}


def load_pair_stability_map(
    profiling_db: Path,
    *,
    composition: str = "sequential",
) -> Dict[Tuple[str, str], float]:
    rows = _fetch_rows(
        profiling_db,
        """SELECT op_a, op_b,
                  (output_has_nan = 0 AND grad_has_nan = 0 AND grad_vanishing = 0) AS stable
           FROM pair_profiles
           WHERE error IS NULL AND composition = ?""",
        (composition,),
    )
    return {(str(op_a), str(op_b)): float(stable) for op_a, op_b, stable in rows}


def load_pair_stability_labels(
    profiling_db: Path,
    op_to_idx: Dict[str, int],
) -> list[tuple[int, int, bool]]:
    rows = _fetch_rows(
        profiling_db,
        """SELECT op_a, op_b,
                  (output_has_nan = 0 AND grad_has_nan = 0 AND grad_vanishing = 0) AS stable
           FROM pair_profiles
           WHERE error IS NULL""",
    )
    labels: list[tuple[int, int, bool]] = []
    for op_a, op_b, stable in rows:
        left = op_to_idx.get(str(op_a))
        right = op_to_idx.get(str(op_b))
        if left is None or right is None:
            continue
        labels.append((left, right, bool(stable)))
    return labels


def load_pair_stability_training_rows(
    profiling_db: Path,
) -> list[tuple[str, str, bool, float | None]]:
    rows = _fetch_rows(
        profiling_db,
        """SELECT op_a, op_b,
                  (output_has_nan = 0 AND grad_has_nan = 0 AND grad_vanishing = 0) AS stable,
                  lipschitz_estimate
           FROM pair_profiles
           WHERE error IS NULL""",
    )
    return [
        (
            str(op_a),
            str(op_b),
            bool(stable),
            float(lipschitz) if lipschitz is not None else None,
        )
        for op_a, op_b, stable, lipschitz in rows
    ]


def load_pair_profile_rows(
    profiling_db: Path,
) -> list[tuple[str, str, float | None, float | None, Any, Any, Any, Any]]:
    rows = _fetch_rows(
        profiling_db,
        """SELECT op_a, op_b, stability_delta, lipschitz_estimate,
                  grad_vanishing, grad_exploding, output_has_nan, grad_has_nan
           FROM pair_profiles
           WHERE error IS NULL""",
    )
    return [
        (
            str(op_a),
            str(op_b),
            float(stability_delta) if stability_delta is not None else None,
            float(lipschitz) if lipschitz is not None else None,
            grad_vanishing,
            grad_exploding,
            output_has_nan,
            grad_has_nan,
        )
        for (
            op_a,
            op_b,
            stability_delta,
            lipschitz,
            grad_vanishing,
            grad_exploding,
            output_has_nan,
            grad_has_nan,
        ) in rows
    ]
