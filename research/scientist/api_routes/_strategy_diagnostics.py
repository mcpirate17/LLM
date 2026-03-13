"""Diagnostics helpers for strategy routes."""
from __future__ import annotations

import time
from typing import Any, Dict, List

from ..notebook import LabNotebook


def diagnose_research_issues(
    analytics_data: Dict[str, Any],
    nb: LabNotebook,
) -> List[Dict[str, Any]]:
    """Diagnose common research pipeline issues from analytics data.

    Returns list of issue dicts with 'issue', 'action_type', and optional 'config_fix'.
    """
    issues: List[Dict[str, Any]] = []

    op_rates = analytics_data.get("op_success_rates") or {}
    if isinstance(op_rates, dict):
        total_uses = sum(v.get("total_uses", 0) for v in op_rates.values() if isinstance(v, dict))
        total_passes = sum(v.get("s1_passes", 0) for v in op_rates.values() if isinstance(v, dict))
        if total_uses > 50 and total_passes == 0:
            issues.append({
                "issue": "Zero S1 passes across all ops — grammar may be misconfigured",
                "action_type": "info",
            })

    grammar = analytics_data.get("grammar_weights") or {}
    if isinstance(grammar, dict):
        learned = grammar.get("learned") or {}
        if not learned:
            issues.append({
                "issue": "No learned grammar weights — consider running more experiments",
                "action_type": "info",
            })

    try:
        stuck = nb.conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE status = 'running' "
            "AND timestamp < ?",
            (time.time() - 7200,),
        ).fetchone()[0]
        if stuck > 0:
            issues.append({
                "issue": f"{stuck} experiment(s) stuck in 'running' for >2 hours",
                "action_type": "info",
            })
    except Exception:
        pass

    return issues
