from __future__ import annotations

import atexit
import logging
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .bus import PublishResult, RuntimeEventBus
from .projectors import LifecycleProjector
from .schema import RuntimeEventDurability, build_runtime_event
from .spool import NdjsonEventSpool
from .state_machine import LifecycleConflictError
from .state_registry import RuntimeLifecycleRegistry
from .workers import ProjectorWorker, ProjectorWorkerHealth

logger = logging.getLogger(__name__)

_SERVICES_LOCK = threading.Lock()
_SERVICES_BY_ROOT: dict[str, "RuntimeEventServices"] = {}
_ATEXIT_REGISTERED = False


@dataclass
class RuntimeEventServices:
    spool: NdjsonEventSpool
    bus: RuntimeEventBus
    registry: RuntimeLifecycleRegistry
    projector_conn: Optional[sqlite3.Connection]
    lifecycle_projector: Optional[LifecycleProjector]
    projector_worker: Optional[ProjectorWorker]

    def bus_health(self):
        return self.bus.health_snapshot()

    def projector_health(self):
        if self.projector_worker is None:
            return ProjectorWorkerHealth(
                running=False,
                iterations=0,
                last_run_at=None,
                last_error=None,
                degraded=False,
                last_applied_count=0,
            )
        return self.projector_worker.health_snapshot()


def runtime_events_root_for(notebook_path: str | Path) -> Path:
    raw = str(notebook_path).strip()
    if raw == ":memory:":
        raise ValueError("runtime events are not supported for in-memory notebook paths")
    if raw.startswith("<MagicMock ") or "MagicMock name='mock.db_path'" in raw:
        raise TypeError(
            f"runtime event services require a real notebook path, got {raw!r}"
        )
    return Path(notebook_path).resolve().parent / "runtime_events"


def get_runtime_event_services(
    notebook_path: str | Path, *, start_projector: bool = False
) -> RuntimeEventServices:
    root = runtime_events_root_for(notebook_path)
    cache_key = str(root)
    with _SERVICES_LOCK:
        _register_atexit_once()
        services = _SERVICES_BY_ROOT.get(cache_key)
        if services is not None:
            if start_projector:
                _ensure_projector_initialized(services, notebook_path)
                _prime_projector(services)
                if services.projector_worker is not None:
                    services.projector_worker.start()
            return services

        spool = NdjsonEventSpool(root)
        bus = RuntimeEventBus(spool=spool)
        registry = RuntimeLifecycleRegistry()
        _replay_registry_from_spool(registry, spool)
        bus.subscribe(registry.consume)
        services = RuntimeEventServices(
            spool=spool,
            bus=bus,
            registry=registry,
            projector_conn=None,
            lifecycle_projector=None,
            projector_worker=None,
        )
        _SERVICES_BY_ROOT[cache_key] = services
        if start_projector:
            _ensure_projector_initialized(services, notebook_path)
            _prime_projector(services)
            if services.projector_worker is not None:
                services.projector_worker.start()
        return services


def start_runtime_event_projector(notebook_path: str | Path) -> RuntimeEventServices:
    services = get_runtime_event_services(notebook_path, start_projector=True)
    return services


def stop_runtime_event_services(notebook_path: str | Path) -> None:
    root = runtime_events_root_for(notebook_path)
    cache_key = str(root)
    with _SERVICES_LOCK:
        services = _SERVICES_BY_ROOT.pop(cache_key, None)
    if services is not None:
        _shutdown_services(services)


def stop_all_runtime_event_services() -> None:
    with _SERVICES_LOCK:
        services = list(_SERVICES_BY_ROOT.values())
        _SERVICES_BY_ROOT.clear()
    for service in services:
        _shutdown_services(service)


def publish_lifecycle_event(
    *,
    notebook_path: str | Path,
    event_type: str,
    producer: str,
    run_id: Optional[str],
    payload: Optional[Mapping[str, Any]] = None,
    sequence: int = 0,
    durability: str = RuntimeEventDurability.CRITICAL,
) -> PublishResult:
    return publish_runtime_event(
        notebook_path=notebook_path,
        event_type=event_type,
        producer=producer,
        run_id=run_id,
        payload=payload,
        sequence=sequence,
        durability=durability,
    )


def publish_runtime_event(
    *,
    notebook_path: str | Path,
    event_type: str,
    producer: str,
    run_id: Optional[str],
    payload: Optional[Mapping[str, Any]] = None,
    sequence: int = 0,
    durability: str = RuntimeEventDurability.BEST_EFFORT,
) -> PublishResult:
    services = get_runtime_event_services(notebook_path)
    event = build_runtime_event(
        event_type=event_type,
        producer=producer,
        run_id=run_id,
        sequence=sequence,
        durability=durability,
        payload=payload,
    )
    return services.bus.publish(event)


def _replay_registry_from_spool(
    registry: RuntimeLifecycleRegistry, spool: NdjsonEventSpool
) -> None:
    for record in spool.replay():
        try:
            registry.consume(record.event, quiet=True)
        except LifecycleConflictError as exc:
            logger.debug(
                "Ignoring conflicting lifecycle event during registry replay: event_id=%s run_id=%s type=%s reason=%s",
                record.event.event_id,
                record.event.run_id,
                record.event.event_type,
                exc,
            )


def _register_atexit_once() -> None:
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return
    atexit.register(stop_all_runtime_event_services)
    _ATEXIT_REGISTERED = True


def _shutdown_services(services: RuntimeEventServices) -> None:
    try:
        if services.projector_worker is not None:
            services.projector_worker.stop(timeout=2.0)
    except Exception:
        logger.debug("Failed to stop projector worker cleanly", exc_info=True)
    try:
        if services.projector_conn is not None:
            services.projector_conn.close()  # No-op for NativeConnectionWrapper
    except Exception:
        logger.debug("Failed to close projector connection cleanly", exc_info=True)


def _ensure_projector_initialized(
    services: RuntimeEventServices, notebook_path: str | Path
) -> None:
    if (
        services.projector_conn is not None
        and services.lifecycle_projector is not None
        and services.projector_worker is not None
    ):
        return
    projector_conn = sqlite3.connect(
        str(Path(notebook_path).resolve()),
        timeout=10.0,
        check_same_thread=False,
    )
    projector_conn.execute("PRAGMA foreign_keys=ON")
    projector_conn.execute("PRAGMA busy_timeout=15000")
    lifecycle_projector = LifecycleProjector(projector_conn, spool=services.spool)
    projector_worker = ProjectorWorker(lifecycle_projector.replay_once)
    services.projector_conn = projector_conn
    services.lifecycle_projector = lifecycle_projector
    services.projector_worker = projector_worker


def _prime_projector(services: RuntimeEventServices) -> None:
    if services.projector_worker is None:
        return
    try:
        status = services.projector_worker.run_once()
        services.registry.projector_unhealthy = bool(status.degraded)
    except Exception:
        services.registry.projector_unhealthy = True
        logger.warning("Runtime lifecycle projector priming failed", exc_info=True)
