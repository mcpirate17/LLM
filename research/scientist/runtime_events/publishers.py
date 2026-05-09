from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional

from .schema import RuntimeEvent, RuntimeEventDurability, build_runtime_event

LIVE_FEED_EVENT_TYPE = "live_feed"


def build_lifecycle_event(
    *,
    event_type: str,
    producer: str,
    run_id: str,
    sequence: int,
    payload: Optional[Mapping[str, Any]] = None,
) -> RuntimeEvent:
    return build_runtime_event(
        event_type=event_type,
        producer=producer,
        run_id=run_id,
        sequence=sequence,
        durability=RuntimeEventDurability.CRITICAL,
        payload=payload,
    )


def live_feed_content(event_type: str, data: Dict[str, Any]) -> str:
    title = event_type.replace("_", " ")
    return str(
        data.get("aria_message")
        or data.get("status")
        or data.get("summary")
        or data.get("error")
        or data.get("hypothesis")
        or title
    )[:500]


def publish_live_feed_event(
    *,
    notebook_path: str | Path,
    event_type: str,
    data: Dict[str, Any],
    durability: str = RuntimeEventDurability.BEST_EFFORT,
) -> None:
    """Append a dashboard live-feed replay event to the runtime spool."""
    from .bootstrap import publish_runtime_event

    experiment_id = data.get("experiment_id")
    payload = {
        "live_feed_type": event_type,
        "event_type": event_type,
        "title": event_type.replace("_", " "),
        "content": live_feed_content(event_type, data),
        "payload": data,
    }
    publish_runtime_event(
        notebook_path=notebook_path,
        event_type=LIVE_FEED_EVENT_TYPE,
        producer="runner.live_feed",
        run_id=str(experiment_id) if experiment_id else None,
        payload=payload,
        durability=durability,
    )


def runtime_event_to_live_feed_event(event: RuntimeEvent) -> Optional[Dict[str, Any]]:
    if event.event_type != LIVE_FEED_EVENT_TYPE:
        return None
    metadata = dict(event.payload or {})
    payload = metadata.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    ret: Dict[str, Any] = {
        "type": metadata.get("event_type") or metadata.get("live_feed_type") or "info",
        "content": metadata.get("content") or "",
        "timestamp": event.created_at,
        "experiment_id": event.run_id or payload.get("experiment_id"),
        "metadata": metadata,
    }
    for key, value in payload.items():
        if key not in ret:
            ret[key] = value
    return ret


def read_live_feed_events(
    notebook_path: str | Path,
    *,
    experiment_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Read recent live-feed events from the runtime spool."""
    from .bootstrap import get_runtime_event_services

    maxlen = max(1, int(limit))
    events: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
    services = get_runtime_event_services(notebook_path)
    for record in services.spool.replay():
        evt = runtime_event_to_live_feed_event(record.event)
        if evt is None:
            continue
        if experiment_id and evt.get("experiment_id") != experiment_id:
            continue
        events.append(evt)
    return list(events)


def latest_experiment_id(events: Iterable[Dict[str, Any]]) -> Optional[str]:
    latest_id: Optional[str] = None
    latest_ts = float("-inf")
    for event in events:
        exp_id = event.get("experiment_id")
        if not exp_id:
            continue
        try:
            ts = float(event.get("timestamp") or 0.0)
        except (TypeError, ValueError):
            ts = 0.0
        if ts >= latest_ts:
            latest_ts = ts
            latest_id = str(exp_id)
    return latest_id
