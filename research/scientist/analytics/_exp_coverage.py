"""Math family coverage and math-space operator impact mixin."""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from ..intelligence.graph_ops import extract_unique_graph_ops

logger = logging.getLogger(__name__)

# Op-to-family mappings (shared between coverage methods)
_HYPERBOLIC_OPS = frozenset(
    {
        "poincare_add",
        "exp_map",
        "log_map",
        "hyp_linear",
        "hyp_distance",
        "hyp_tangent_nonlinear",
    }
)
_TROPICAL_OPS = frozenset(
    {
        "tropical_matmul",
        "tropical_add",
        "tropical_attention",
        "tropical_center",
    }
)
_PADIC_OPS = frozenset({"padic_expand", "ultrametric_attention", "padic_gate"})
_CLIFFORD_OPS = frozenset(
    {
        "geometric_product",
        "rotor_transform",
        "grade_select",
        "grade_mix",
    }
)
_FUNCTIONAL_OPS = frozenset(
    {
        "basis_expansion",
        "integral_kernel",
        "fixed_point_iter",
    }
)

_OP_FAMILY_MAP = {
    **{op: "hyperbolic" for op in _HYPERBOLIC_OPS},
    **{op: "tropical" for op in _TROPICAL_OPS},
    **{op: "p-adic" for op in _PADIC_OPS},
    **{op: "clifford" for op in _CLIFFORD_OPS},
}

_FAMILY_ORDER = [
    "euclidean",
    "hyperbolic",
    "tropical",
    "p-adic",
    "clifford",
    "functional",
]


def _extract_op_names(graph_json: Optional[str]) -> set[str]:
    """Extract op names from graph JSON string."""
    if not graph_json:
        return set()
    return set(extract_unique_graph_ops(graph_json))


def _family_from_row(graph_json: Optional[str], arch_spec_json: Optional[str]) -> str:
    """Determine math family from graph/arch-spec JSON."""
    op_names = _extract_op_names(graph_json)

    token_mixing = None
    channel_mixing = None
    if arch_spec_json:
        try:
            arch = json.loads(arch_spec_json)
            choices = arch.get("choices", {}) if isinstance(arch, dict) else {}
            if isinstance(choices, dict):
                token_mixing = choices.get("token_mixing")
                channel_mixing = choices.get("channel_mixing")
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

    if (
        (op_names & _FUNCTIONAL_OPS)
        or token_mixing == "integral_kernel_mixing"
        or channel_mixing in {"basis_expansion_layer", "implicit_fixed_point"}
    ):
        return "functional"
    if op_names & _HYPERBOLIC_OPS:
        return "hyperbolic"
    if op_names & _TROPICAL_OPS:
        return "tropical"
    if op_names & _PADIC_OPS:
        return "p-adic"
    if op_names & _CLIFFORD_OPS:
        return "clifford"
    return "euclidean"


def _ensure_bucket(store: Dict[str, Dict[str, float]], key: str) -> Dict[str, float]:
    """Get or create a metric bucket in a store dict."""
    if key not in store:
        store[key] = {
            "n_tested": 0,
            "n_stage1_passed": 0,
            "n_validation_passed": 0,
            "n_baseline_wins": 0,
            "novelty_sum": 0.0,
            "novelty_count": 0,
        }
    return store[key]


def _finalize_buckets(
    rows_by_key: Dict[str, Dict[str, float]], label_key: str
) -> List[Dict]:
    """Finalize metric buckets into sorted summary rows."""
    finalized: List[Dict] = []
    for key, bucket in rows_by_key.items():
        n_tested = int(bucket["n_tested"])
        if n_tested <= 0:
            continue
        novelty_count = int(bucket["novelty_count"])
        stage1_rate = float(bucket["n_stage1_passed"]) / n_tested
        validation_rate = float(bucket["n_validation_passed"]) / n_tested
        baseline_win_rate = float(bucket["n_baseline_wins"]) / n_tested
        sample_weight = min(1.0, n_tested / 25.0)
        trust_score = (
            0.5 * stage1_rate + 0.3 * validation_rate + 0.2 * baseline_win_rate
        ) * sample_weight
        if trust_score >= 0.6 and n_tested >= 20:
            trust_label = "high"
        elif trust_score >= 0.35 and n_tested >= 8:
            trust_label = "medium"
        else:
            trust_label = "low"
        finalized.append(
            {
                label_key: key,
                "n_tested": n_tested,
                "n_stage1_passed": int(bucket["n_stage1_passed"]),
                "n_validation_passed": int(bucket["n_validation_passed"]),
                "n_baseline_wins": int(bucket["n_baseline_wins"]),
                "stage1_pass_rate": round(stage1_rate, 4),
                "validation_pass_rate": round(validation_rate, 4),
                "baseline_win_rate": round(baseline_win_rate, 4),
                "trust_score": round(trust_score, 4),
                "trust_label": trust_label,
                "avg_novelty_score": (
                    round(float(bucket["novelty_sum"]) / novelty_count, 4)
                    if novelty_count > 0
                    else None
                ),
            }
        )
    return sorted(finalized, key=lambda row: (-row["n_tested"], row[label_key]))


class _CoverageMixin:
    """Math family coverage and math-space operator impact analysis."""

    __slots__ = ()

    def math_family_coverage(self) -> Dict:
        """Summarize evaluated/surviving coverage by mathematical family."""
        rows = self.nb.conn.execute("""
            SELECT stage1_passed, graph_json, arch_spec_json
            FROM program_results
            WHERE graph_json IS NOT NULL OR arch_spec_json IS NOT NULL
            ORDER BY timestamp DESC LIMIT 5000
        """).fetchall()

        stats = {
            fam: {"family": fam, "n_tested": 0, "n_survived": 0}
            for fam in _FAMILY_ORDER
        }

        total_tested = 0
        total_survived = 0
        for row in rows:
            family = _family_from_row(row["graph_json"], row["arch_spec_json"])
            bucket = stats.get(family, stats["euclidean"])
            bucket["n_tested"] += 1
            total_tested += 1

            if row["stage1_passed"]:
                bucket["n_survived"] += 1
                total_survived += 1

        families = []
        for fam in _FAMILY_ORDER:
            entry = stats[fam]
            n_tested = entry["n_tested"]
            n_survived = entry["n_survived"]
            families.append(
                {
                    "family": fam,
                    "n_tested": n_tested,
                    "n_survived": n_survived,
                    "survival_rate": round(n_survived / n_tested, 4)
                    if n_tested > 0
                    else 0.0,
                    "tested_share": round(n_tested / total_tested, 4)
                    if total_tested > 0
                    else 0.0,
                    "survivor_share": round(n_survived / total_survived, 4)
                    if total_survived > 0
                    else 0.0,
                }
            )

        return {
            "families": families,
            "totals": {
                "n_tested": total_tested,
                "n_survived": total_survived,
            },
        }

    def _resolve_impact_columns(self) -> tuple[str, str]:
        """Resolve optional column names for mathspace impact queries."""
        cached = getattr(self.nb, "_program_results_columns", None)
        if cached is None:
            cached = {
                row["name"]
                for row in self.nb.conn.execute(
                    "PRAGMA table_info(program_results)"
                ).fetchall()
                if row and row["name"]
            }
            self.nb._program_results_columns = cached
        columns = cached
        validation_col = (
            "validation_passed"
            if "validation_passed" in columns
            else "NULL AS validation_passed"
        )
        baseline_col = (
            "validation_baseline_ratio"
            if "validation_baseline_ratio" in columns
            else "NULL AS validation_baseline_ratio"
        )
        return validation_col, baseline_col

    def _accumulate_mathspace_buckets(
        self, rows: list, tracked_ops: set[str]
    ) -> tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], int]:
        """Accumulate per-op and per-family buckets from rows.

        Returns (by_operator, by_family, programs_with_mathspace).
        """
        by_operator: Dict[str, Dict[str, float]] = {}
        by_family: Dict[str, Dict[str, float]] = {}
        programs_with_mathspace = 0

        for row in rows:
            graph_json = row["graph_json"]
            if not graph_json:
                continue

            ops = self._extract_ops_fast(graph_json)
            if ops is None:
                ops = self._extract_ops_fallback(graph_json)
            if not ops:
                continue

            used_ops = sorted(tracked_ops.intersection(set(ops)))
            if not used_ops:
                continue

            programs_with_mathspace += 1
            used_families = sorted({_OP_FAMILY_MAP[op] for op in used_ops})
            stage1_passed = bool(row["stage1_passed"])
            validation_passed = bool(row["validation_passed"])
            novelty = self._as_float(row["novelty_score"])
            baseline_ratio = self._as_float(row["validation_baseline_ratio"])
            baseline_win = baseline_ratio is not None and baseline_ratio < 1.0

            for target_key, store in [
                (used_ops, by_operator),
                (used_families, by_family),
            ]:
                for name in target_key:
                    bucket = _ensure_bucket(store, name)
                    bucket["n_tested"] += 1
                    if stage1_passed:
                        bucket["n_stage1_passed"] += 1
                    if validation_passed:
                        bucket["n_validation_passed"] += 1
                    if baseline_win:
                        bucket["n_baseline_wins"] += 1
                    if novelty is not None:
                        bucket["novelty_sum"] += novelty
                        bucket["novelty_count"] += 1

        return by_operator, by_family, programs_with_mathspace

    def mathspace_operator_impact(self) -> Dict:
        """Impact summary for math-space operators and families.

        Reports tested counts, S1/validation pass rates, novelty signal,
        and baseline-win rates for each math-space operator family.
        """
        validation_col, baseline_col = self._resolve_impact_columns()

        rows = self.nb.conn.execute(f"""
            SELECT graph_json, stage1_passed, {validation_col}, novelty_score, {baseline_col}
            FROM program_results
            WHERE graph_json IS NOT NULL
        """).fetchall()

        tracked_ops = set(_OP_FAMILY_MAP.keys())

        if not rows:
            return {
                "available": False,
                "totals": {
                    "n_programs_with_graph": 0,
                    "n_programs_with_mathspace": 0,
                    "n_mathspace_ops_observed": 0,
                },
                "by_operator": [],
                "by_family": [],
                "explanation": "No graph-level program data available for math-space impact analysis.",
            }

        by_operator, by_family, programs_with_mathspace = (
            self._accumulate_mathspace_buckets(rows, tracked_ops)
        )

        by_operator_rows = _finalize_buckets(by_operator, "op_name")
        by_family_rows = _finalize_buckets(by_family, "family")
        top_trustworthy_ops = sorted(
            by_operator_rows,
            key=lambda row: (
                -(row.get("trust_score") or 0.0),
                -(row.get("n_tested") or 0),
                row.get("op_name") or "",
            ),
        )[:3]

        top_op = by_operator_rows[0]["op_name"] if by_operator_rows else None
        explanation = (
            f"Observed {len(by_operator_rows)} math-space ops across {programs_with_mathspace}/{len(rows)} programs with graph traces. "
            f"Most common op: {top_op}."
            if top_op
            else "No math-space operators were observed in current graph traces."
        )

        return {
            "available": len(by_operator_rows) > 0,
            "totals": {
                "n_programs_with_graph": len(rows),
                "n_programs_with_mathspace": programs_with_mathspace,
                "n_mathspace_ops_observed": len(by_operator_rows),
            },
            "by_operator": by_operator_rows,
            "by_family": by_family_rows,
            "top_trustworthy_operators": top_trustworthy_ops,
            "explanation": explanation,
        }
