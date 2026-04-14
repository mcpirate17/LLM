"""Runtime event bus scaffolding for lifecycle-first migration."""

from .bus import PublishResult, RuntimeEventBus
from .bootstrap import (
    RuntimeEventServices,
    get_runtime_event_services,
    publish_lifecycle_event,
    publish_runtime_event,
    runtime_events_root_for,
    start_runtime_event_projector,
    stop_all_runtime_event_services,
    stop_runtime_event_services,
)
from .publishers import build_lifecycle_event
from .schema import (
    LIFECYCLE_EVENT_TYPES,
    RuntimeEvent,
    RuntimeEventDurability,
    build_runtime_event,
)
from .state_machine import LifecycleConflictError, LifecycleStateMachine
from .state_registry import RuntimeLifecycleRegistry
from .workers import ProjectorWorker

__all__ = [
    "LIFECYCLE_EVENT_TYPES",
    "LifecycleConflictError",
    "LifecycleStateMachine",
    "PublishResult",
    "ProjectorWorker",
    "RuntimeEvent",
    "RuntimeEventBus",
    "RuntimeEventServices",
    "RuntimeEventDurability",
    "RuntimeLifecycleRegistry",
    "build_lifecycle_event",
    "build_runtime_event",
    "get_runtime_event_services",
    "publish_lifecycle_event",
    "publish_runtime_event",
    "runtime_events_root_for",
    "start_runtime_event_projector",
    "stop_all_runtime_event_services",
    "stop_runtime_event_services",
]
