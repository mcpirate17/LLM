from __future__ import annotations

from typing import Any, Mapping, Optional

from .schema import RuntimeEvent, RuntimeEventDurability, build_runtime_event


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
