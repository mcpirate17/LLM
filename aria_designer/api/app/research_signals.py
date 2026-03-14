from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, Optional, Set

import requests

from .component_identity import canonicalize_component_id
from .config import settings

_RESEARCH_SIGNALS_CACHE_LOCK = threading.Lock()
_RESEARCH_SIGNALS_CACHE: Dict[str, Any] = {
    "fetched_at": 0.0,
    "payload": None,
}
_LEADERBOARD_CACHE: Dict[str, Any] = {}


def _extract_component_ids(entry: Dict[str, Any]) -> list[str]:
    graph_json = entry.get("graph_json") or entry.get("_graph_json")
    if isinstance(graph_json, str) and graph_json:
        try:
            graph = json.loads(graph_json)
        except Exception:
            graph = None
        if isinstance(graph, dict):
            nodes = graph.get("nodes")
            if isinstance(nodes, list):
                component_ids: Set[str] = set()
                for node in nodes:
                    if not isinstance(node, dict):
                        continue
                    raw = str(node.get("component_type", "")).strip()
                    if raw:
                        canonical = canonicalize_component_id(raw)
                        component_ids.add(canonical.split("/")[-1])
                return sorted(component_ids)
    return []


def _hydrate_leaderboard_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    hydrated = dict(entry)
    hydrated["_component_ids"] = _extract_component_ids(hydrated)
    return hydrated


def _get_json(url: str, *, timeout: float) -> Optional[Any]:
    try:
        resp = requests.get(url, timeout=timeout)
        if not resp.ok:
            return None
        return resp.json() if resp.content else None
    except Exception:
        return None


def fetch_research_recommendation_signals(force: bool = False) -> Optional[Dict[str, Any]]:
    """Fetch and cache recommendation signals from the research analytics API."""
    if not settings.RECOMMENDER_USE_RESEARCH_SIGNALS:
        return None

    now = time.time()
    with _RESEARCH_SIGNALS_CACHE_LOCK:
        cached_payload = _RESEARCH_SIGNALS_CACHE.get("payload")
        fetched_at = float(_RESEARCH_SIGNALS_CACHE.get("fetched_at") or 0.0)
        if not force and cached_payload and (now - fetched_at) <= max(1.0, settings.RECOMMENDER_SIGNALS_TTL_S):
            return cached_payload

    url = f"{settings.LINEAGE_SYNC_BASE.rstrip('/')}/api/analytics/recommendation-signals"
    payload = _get_json(url, timeout=max(0.2, settings.RECOMMENDER_SIGNALS_TIMEOUT))
    if not isinstance(payload, dict):
        return None
    with _RESEARCH_SIGNALS_CACHE_LOCK:
        _RESEARCH_SIGNALS_CACHE["fetched_at"] = now
        _RESEARCH_SIGNALS_CACHE["payload"] = payload
    return payload


def fetch_leaderboard_top_entries(
    n: int = 10,
    min_composite: float = 50.0,
    force: bool = False,
) -> list[Dict[str, Any]]:
    if not settings.RECOMMENDER_USE_RESEARCH_SIGNALS:
        return []

    now = time.time()
    cache_key = f"{int(n)}:{float(min_composite):.3f}"
    with _RESEARCH_SIGNALS_CACHE_LOCK:
        cached = _LEADERBOARD_CACHE.get(cache_key)
        if (
            not force
            and isinstance(cached, dict)
            and (now - float(cached.get("fetched_at") or 0.0)) <= 120.0
        ):
            payload = cached.get("payload")
            return payload if isinstance(payload, list) else []

    url = (
        f"{settings.LINEAGE_SYNC_BASE.rstrip('/')}/api/leaderboard"
        f"?limit={max(1, int(n))}&sort=composite_score&include_references=0"
    )
    payload = _get_json(url, timeout=max(0.2, settings.RECOMMENDER_SIGNALS_TIMEOUT))
    if isinstance(payload, dict):
        entries = payload.get("entries")
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = None
    if not isinstance(entries, list):
        return []

    filtered: list[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            composite = float(entry.get("composite_score") or 0.0)
        except Exception:
            composite = 0.0
        if composite >= float(min_composite):
            filtered.append(_hydrate_leaderboard_entry(entry))
    with _RESEARCH_SIGNALS_CACHE_LOCK:
        _LEADERBOARD_CACHE[cache_key] = {"fetched_at": now, "payload": filtered}
    return filtered
