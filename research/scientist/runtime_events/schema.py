from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional


def _uuid7_hex() -> str:
    """Generate a UUIDv7 hex string (time-sortable, globally unique)."""
    ts_ms = int(time.time() * 1000)
    rand = int.from_bytes(os.urandom(10), "big")
    rand_a = rand >> 62 & 0x0FFF
    rand_b = rand & 0x3FFFFFFFFFFFFFFF
    uuid_int = (
        (ts_ms & 0xFFFFFFFFFFFF) << 80 | 7 << 76 | rand_a << 64 | 2 << 62 | rand_b
    )
    return f"{uuid_int:032x}"


LIFECYCLE_EVENT_TYPES = frozenset(
    {
        "experiment_start_requested",
        "experiment_started",
        "experiment_start_failed",
        "experiment_completed",
        "experiment_failed",
    }
)


class RuntimeEventDurability:
    CRITICAL = "critical"
    BEST_EFFORT = "best_effort"


@dataclass(frozen=True)
class RuntimeEvent:
    event_id: str
    event_type: str
    schema_version: int
    created_at: float
    producer: str
    run_id: Optional[str]
    sequence: int
    durability: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeEvent":
        return cls(
            event_id=str(data["event_id"]),
            event_type=str(data["event_type"]),
            schema_version=int(data.get("schema_version", 1)),
            created_at=float(data["created_at"]),
            producer=str(data["producer"]),
            run_id=str(data["run_id"]) if data.get("run_id") is not None else None,
            sequence=int(data.get("sequence", 0)),
            durability=str(data.get("durability", RuntimeEventDurability.BEST_EFFORT)),
            payload=dict(data.get("payload") or {}),
        )

    def is_lifecycle(self) -> bool:
        return self.event_type in LIFECYCLE_EVENT_TYPES


def build_runtime_event(
    *,
    event_type: str,
    producer: str,
    run_id: Optional[str] = None,
    sequence: int = 0,
    durability: str = RuntimeEventDurability.BEST_EFFORT,
    payload: Optional[Mapping[str, Any]] = None,
    event_id: Optional[str] = None,
    created_at: Optional[float] = None,
    schema_version: int = 1,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=event_id or _uuid7_hex(),
        event_type=event_type,
        schema_version=schema_version,
        created_at=float(created_at if created_at is not None else time.time()),
        producer=producer,
        run_id=run_id,
        sequence=int(sequence),
        durability=durability,
        payload=dict(payload or {}),
    )
