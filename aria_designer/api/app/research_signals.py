from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import requests

from .config import settings

_RESEARCH_SIGNALS_CACHE_LOCK = threading.Lock()
_RESEARCH_SIGNALS_CACHE: Dict[str, Any] = {
    "fetched_at": 0.0,
    "payload": None,
}


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
    try:
        resp = requests.get(url, timeout=max(0.2, settings.RECOMMENDER_SIGNALS_TIMEOUT))
        if not resp.ok:
            return None
        payload = resp.json() if resp.content else {}
        if not isinstance(payload, dict):
            return None
        with _RESEARCH_SIGNALS_CACHE_LOCK:
            _RESEARCH_SIGNALS_CACHE["fetched_at"] = now
            _RESEARCH_SIGNALS_CACHE["payload"] = payload
        return payload
    except Exception:
        return None
