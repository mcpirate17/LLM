from __future__ import annotations

from collections import Counter
from typing import Dict, List

from .models import HistoricalInsightsResponse
from .research_signals import fetch_leaderboard_top_entries, fetch_research_recommendation_signals


def _top_components_from_entries(entries: List[Dict[str, object]]) -> List[Dict[str, object]]:
    comp_counter: Counter[str] = Counter()
    for entry in entries:
        component_ids = entry.get("_component_ids")
        if not isinstance(component_ids, list):
            continue
        for token in component_ids:
            normalized = str(token).strip().lower()
            if normalized:
                comp_counter[normalized] += 1
    return [
        {"component_id": name, "count": count}
        for name, count in comp_counter.most_common(10)
    ]


def _extract_signal_patterns(signals: object) -> tuple[List[str], List[str]]:
    success_patterns: List[str] = []
    failure_patterns: List[str] = []
    if isinstance(signals, dict):
        insights = signals.get("insights")
        if isinstance(insights, list):
            for ins in insights[:50]:
                if not isinstance(ins, dict):
                    continue
                cat = str(ins.get("category") or "").lower()
                content = str(ins.get("content") or "").strip()
                if not content:
                    continue
                if cat == "success_factor":
                    success_patterns.append(content)
                elif cat == "failure_mode":
                    failure_patterns.append(content)
        toxic_ops = signals.get("toxic_ops")
        if isinstance(toxic_ops, list):
            for op in toxic_ops:
                failure_patterns.append(f"Toxic operator: {op}")
    return success_patterns[:15], failure_patterns[:15]


def build_historical_insights_response() -> HistoricalInsightsResponse:
    entries = fetch_leaderboard_top_entries(n=10, min_composite=50.0)
    success_patterns, failure_patterns = _extract_signal_patterns(
        fetch_research_recommendation_signals(force=False)
    )

    return HistoricalInsightsResponse(
        top_components=_top_components_from_entries(entries),
        success_patterns=success_patterns,
        failure_patterns=failure_patterns,
    )
