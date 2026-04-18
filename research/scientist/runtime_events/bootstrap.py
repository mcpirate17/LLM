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
from .workers import ProjectorWorker

logger = logging.getLogger(__name__)

_SERVICES_LOCK = threading.Lock()
_SERVICES_BY_ROOT: dict[str, "RuntimeEventServices"] = {}
_ATEXIT_REGISTERED = False


@dataclass
class RuntimeEventServices:
    spool: NdjsonEventSpool
    bus: RuntimeEventBus
    registry: RuntimeLifecycleRegistry
    projector_conn: sqlite3.Connection
    lifecycle_projector: LifecycleProjector
    projector_worker: ProjectorWorker

    def bus_health(self):
        return self.bus.health_snapshot()

    def projector_health(self):
        return self.projector_worker.health_snapshot()


def runtime_events_root_for(notebook_path: str | Path) -> Path:
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
                _prime_projector(services)
                services.projector_worker.start()
            return services

        spool = NdjsonEventSpool(root)
        bus = RuntimeEventBus(spool=spool)
        registry = RuntimeLifecycleRegistry()
        _replay_registry_from_spool(registry, spool)
        bus.subscribe(registry.consume)
        from ..notebook.native_conn import NativeConnectionWrapper

        projector_conn = NativeConnectionWrapper(str(Path(notebook_path).resolve()))
        lifecycle_projector = LifecycleProjector(projector_conn, spool=spool)
        projector_worker = ProjectorWorker(lifecycle_projector.replay_once)
        services = RuntimeEventServices(
            spool=spool,
            bus=bus,
            registry=registry,
            projector_conn=projector_conn,
            lifecycle_projector=lifecycle_projector,
            projector_worker=projector_worker,
        )
        _SERVICES_BY_ROOT[cache_key] = services
        if start_projector:
            _prime_projector(services)
            projector_worker.start()
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
            registry.consume(record.event)
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
        services.projector_worker.stop(timeout=2.0)
    except Exception:
        logger.debug("Failed to stop projector worker cleanly", exc_info=True)
    try:
        services.projector_conn.close()  # No-op for NativeConnectionWrapper
    except Exception:
        logger.debug("Failed to close projector connection cleanly", exc_info=True)


def _prime_projector(services: RuntimeEventServices) -> None:
    try:
        status = services.projector_worker.run_once()
        services.registry.projector_unhealthy = bool(status.degraded)
    except Exception:
        services.registry.projector_unhealthy = True
        logger.warning("Runtime lifecycle projector priming failed", exc_info=True)
