import atexit
import json
import queue
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from research.scientist.api_routes._helpers import resolve_runner_status
from research.scientist.notebook import LabNotebook
from research.scientist.persona import get_aria
from research.scientist.runner.control_actions import _ControlActionsMixin
from research.scientist.runner.core import _CoreMixin
from research.scientist.runner.control_start import _ControlStartMixin
from research.scientist.runner.continuous_investigation import _ContinuousInvestigationMixin
from research.scientist.runner.continuous_loop import _ContinuousLoopMixin
from research.scientist.runner.continuous_modes import _ContinuousModesMixin
from research.scientist.runner.continuous_validation import _ContinuousValidationMixin
from research.scientist.runner.cycle import _CycleMixin
from research.scientist.runner.execution_investigation import _ExecutionInvestigationMixin
from research.scientist.runner.execution_screening import _ExecutionScreeningMixin
from research.scientist.runner.execution_search import _ExecutionSearchMixin
from research.scientist.runner.execution_validation import _ExecutionValidationMixin
from research.scientist.runner._types import LiveProgress, RunConfig
from research.scientist.runtime_events import (
    LifecycleConflictError,
    LifecycleStateMachine,
    RuntimeEventBus,
    RuntimeLifecycleRegistry,
    RuntimeEventDurability,
    ProjectorWorker,
    build_lifecycle_event,
    build_runtime_event,
    get_runtime_event_services,
    publish_lifecycle_event,
    publish_runtime_event,
    start_runtime_event_projector,
    stop_runtime_event_services,
)
from research.scientist.runtime_events.projectors import LifecycleProjector
from research.scientist.runtime_events.spool import NdjsonEventSpool

pytestmark = pytest.mark.unit


def _make_event(event_type: str, *, run_id: str = "exp-1", sequence: int = 0, **payload):
    return build_runtime_event(
        event_type=event_type,
        producer="test",
        run_id=run_id,
        sequence=sequence,
        durability=RuntimeEventDurability.CRITICAL,
        payload=payload,
    )


def _make_projector_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE experiments (
            experiment_id TEXT PRIMARY KEY,
            timestamp REAL NOT NULL,
            experiment_type TEXT NOT NULL,
            status TEXT NOT NULL,
            hypothesis TEXT,
            research_question TEXT,
            preregistration_id TEXT,
            config_json TEXT NOT NULL,
            results_json TEXT,
            n_programs_generated INTEGER DEFAULT 0,
            n_stage0_passed INTEGER DEFAULT 0,
            n_stage05_passed INTEGER DEFAULT 0,
            n_stage1_passed INTEGER DEFAULT 0,
            best_loss_ratio REAL,
            best_novelty_score REAL,
            aria_summary TEXT,
            aria_mood TEXT,
            insights_json TEXT,
            llm_analysis TEXT,
            started_at REAL,
            completed_at REAL,
            duration_seconds REAL
        )"""
    )
    conn.execute(
        """CREATE TABLE entries (
            entry_id TEXT PRIMARY KEY,
            experiment_id TEXT,
            timestamp REAL NOT NULL,
            entry_type TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata_json TEXT,
            tags TEXT
        )"""
    )
    return conn


def test_bus_publishes_to_spool_and_registry(tmp_path):
    spool = NdjsonEventSpool(tmp_path / "runtime_events")
    bus = RuntimeEventBus(spool=spool)
    registry = RuntimeLifecycleRegistry()
    bus.subscribe(registry.consume)

    result = bus.publish(
        _make_event(
            "experiment_started",
            sequence=1,
            experiment_type="screening",
            config={"batch_size": 8},
        )
    )

    assert result.spool_offset is not None
    assert result.spool_offset.line_number == 1
    active = registry.get("exp-1")
    assert active is not None
    assert active.status == "running"


def test_state_machine_rejects_conflicting_terminal_transition():
    machine = LifecycleStateMachine()
    started = _make_event("experiment_started", sequence=1)
    completed = _make_event("experiment_completed", sequence=2)
    failed = _make_event("experiment_failed", sequence=3, error="boom")

    accepted = machine.transition(None, _make_event("experiment_start_requested"))
    accepted = machine.transition(accepted, started)
    accepted = machine.transition(accepted, completed)
    with pytest.raises(LifecycleConflictError):
        machine.transition(accepted, failed)


def test_lifecycle_projector_replays_completed_run(tmp_path):
    spool = NdjsonEventSpool(tmp_path / "runtime_events")
    bus = RuntimeEventBus(spool=spool)
    conn = _make_projector_db()
    projector = LifecycleProjector(conn, spool=spool)

    bus.publish(_make_event("experiment_start_requested", sequence=1))
    bus.publish(
        _make_event(
            "experiment_started",
            sequence=2,
            experiment_type="screening",
            hypothesis="test hypothesis",
            config={"seed": 7},
            started_at=10.0,
        )
    )
    bus.publish(
        _make_event(
            "experiment_completed",
            sequence=3,
            completed_at=20.0,
            results={
                "total": 4,
                "stage0_passed": 2,
                "stage05_passed": 1,
                "stage1_passed": 1,
                "best_loss_ratio": 0.5,
            },
            insights=["one"],
        )
    )

    status = projector.replay_once()

    assert status.degraded is False
    row = conn.execute(
        "SELECT status, config_json, results_json, duration_seconds FROM experiments WHERE experiment_id = ?",
        ("exp-1",),
    ).fetchone()
    assert row is not None
    assert row[0] == "completed"
    assert json.loads(row[1]) == {"seed": 7}
    assert json.loads(row[2])["total"] == 4
    assert row[3] == 10.0


def test_lifecycle_projector_is_idempotent_on_replay(tmp_path):
    spool = NdjsonEventSpool(tmp_path / "runtime_events")
    bus = RuntimeEventBus(spool=spool)
    conn = _make_projector_db()
    projector = LifecycleProjector(conn, spool=spool)

    bus.publish(
        _make_event(
            "experiment_started",
            sequence=1,
            experiment_type="screening",
            config={"seed": 9},
        )
    )

    first = projector.replay_once()
    second = projector.replay_once()

    assert first.applied_count == 1
    assert second.applied_count == 0
    count = conn.execute("SELECT COUNT(*) FROM applied_runtime_events").fetchone()[0]
    assert count == 1


def test_bus_typed_subscriptions_only_receive_matching_events(tmp_path):
    spool = NdjsonEventSpool(tmp_path / "runtime_events")
    bus = RuntimeEventBus(spool=spool)
    seen = []

    def lifecycle_only(event):
        seen.append(event.event_type)

    bus.subscribe_to("experiment_started", lifecycle_only)
    bus.publish(_make_event("experiment_start_requested", sequence=1))
    result = bus.publish(_make_event("experiment_started", sequence=2))

    assert seen == ["experiment_started"]
    assert result.subscriber_count == 1
    assert result.subscriber_failures == 0


def test_bus_isolates_subscriber_failures(tmp_path):
    spool = NdjsonEventSpool(tmp_path / "runtime_events")
    bus = RuntimeEventBus(spool=spool)
    seen = []

    def broken(_event):
        raise RuntimeError("subscriber boom")

    def healthy(event):
        seen.append(event.event_id)

    bus.subscribe(broken)
    bus.subscribe(healthy)
    event = _make_event("experiment_started", sequence=1)
    result = bus.publish(event)
    health = bus.health_snapshot()

    assert seen == [event.event_id]
    assert result.subscriber_count == 2
    assert result.subscriber_failures == 1
    assert health.subscriber_failure_count == 1
    assert health.last_subscriber_failure is not None
    assert health.last_subscriber_failure.error_type == "RuntimeError"


def test_build_lifecycle_event_defaults_to_critical_durability():
    event = build_lifecycle_event(
        event_type="experiment_completed",
        producer="test.publisher",
        run_id="exp-42",
        sequence=5,
        payload={"results": {"total": 1}},
    )

    assert event.durability == RuntimeEventDurability.CRITICAL
    assert event.run_id == "exp-42"
    assert event.event_type == "experiment_completed"


def test_projector_worker_tracks_health_and_applies_events(tmp_path):
    spool = NdjsonEventSpool(tmp_path / "runtime_events")
    bus = RuntimeEventBus(spool=spool)
    conn = _make_projector_db()
    projector = LifecycleProjector(conn, spool=spool)
    worker = ProjectorWorker(projector.replay_once, interval_seconds=0.05)

    bus.publish(
        _make_event(
            "experiment_started",
            sequence=1,
            experiment_type="screening",
            config={"seed": 11},
        )
    )

    status = worker.run_once()
    health = worker.health_snapshot()

    assert status.applied_count == 1
    assert health.iterations == 1
    assert health.degraded is False
    assert health.last_applied_count == 1
    row = conn.execute(
        "SELECT status FROM experiments WHERE experiment_id = ?",
        ("exp-1",),
    ).fetchone()
    assert row is not None
    assert row[0] == "running"


class _IdleRunner:
    def __init__(self) -> None:
        self.is_running = False
        self.progress = LiveProgress()


def _make_status_notebook(tmp_path: Path) -> LabNotebook:
    return LabNotebook(str(tmp_path / "status.db"))


class _ResumeRunner(_ControlStartMixin):
    def __init__(self, notebook_path: str | Path) -> None:
        self.notebook_path = str(notebook_path)
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._progress = LiveProgress()
        self.events = []

    @property
    def is_running(self) -> bool:
        return False

    @property
    def progress(self) -> LiveProgress:
        return self._progress

    def _ensure_math_spaces(self) -> None:
        return

    def _make_notebook(self) -> LabNotebook:
        return LabNotebook(self.notebook_path, skip_migrate=True, check_same_thread=False)

    def _emit_event(self, event_type: str, data: dict) -> None:
        self.events.append((event_type, data))

    def _run_continuous_thread(self, config) -> None:
        return


class _ContinuousRunner(_ControlStartMixin, _ControlActionsMixin):
    def __init__(self, notebook_path: str | Path) -> None:
        self.notebook_path = str(notebook_path)
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._progress = LiveProgress()
        self._aria_cycle_paused = False
        self._aria_cycle_status = {
            "phase": "idle",
            "continuous_active": False,
            "cycle_index": 0,
        }
        self._live_loss_curve = []
        self._event_queue = queue.Queue(maxsize=10)
        self.aria = get_aria()

    @property
    def is_running(self) -> bool:
        return False

    @property
    def progress(self) -> LiveProgress:
        return self._progress

    def prescreen_run_config(self, config, mode: str, auto_harden: bool):
        return config, {}

    def _ensure_math_spaces(self) -> None:
        return

    def _set_aria_cycle_phase(
        self,
        phase: str,
        *,
        continuous_active: bool = False,
        cycle_index: int = 0,
        selected_mode=None,
        note: str = "",
        emit_event: bool = False,
    ) -> None:
        self._aria_cycle_status = {
            "phase": phase,
            "continuous_active": continuous_active,
            "cycle_index": cycle_index,
            "selected_mode": selected_mode,
            "last_note": note,
        }

    def _persist_live_feed_event(self, event_type: str, data):
        return

    def _run_continuous_thread(self, config) -> None:
        return


class _ContinuousLoopRunner(_ContinuousLoopMixin):
    def __init__(self, notebook_path: str | Path) -> None:
        self.notebook_path = str(notebook_path)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._progress = LiveProgress()
        self.aria = get_aria()
        self.events = []

    @property
    def progress(self) -> LiveProgress:
        return self._progress

    def _emit_event(self, event_type: str, data: dict) -> None:
        self.events.append((event_type, data))

    def _update_progress(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self._progress, key, value)

    def _set_aria_cycle_phase(self, *args, **kwargs) -> None:
        return

    def _end_of_session_automation(self, config, reason: str) -> None:
        return

    def _run_pending_scale_up(self) -> None:
        return


class _StartedModeRunner(_ControlStartMixin, _ControlActionsMixin):
    def __init__(self, notebook_path: str | Path) -> None:
        self.notebook_path = str(notebook_path)
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._progress = LiveProgress()
        self._event_queue = queue.Queue(maxsize=10)
        self._live_loss_curve = []
        self._aria_cycle_status = {"phase": "idle", "continuous_active": False}
        self.aria = get_aria()
        self.events = []

    @property
    def is_running(self) -> bool:
        return False

    @property
    def progress(self) -> LiveProgress:
        return self._progress

    def _ensure_math_spaces(self) -> None:
        return

    def _make_notebook(self) -> LabNotebook:
        return LabNotebook(self.notebook_path, check_same_thread=False)

    def _persist_live_feed_event(self, event_type: str, data):
        return

    def _build_hypothesis_metadata(self, **kwargs):
        return dict(kwargs)

    def _start_preregistered_experiment(
        self,
        *,
        nb: LabNotebook,
        experiment_type: str,
        config,
        hypothesis,
        hypothesis_metadata,
        preregistration,
        exploratory,
        created_by: str,
    ) -> str:
        return nb.start_experiment(experiment_type, config, hypothesis=hypothesis)

    def _validation_config_with_result_ids(
        self, config, result_ids: list[str], trigger: str
    ):
        payload = config.to_dict()
        payload["result_ids"] = list(result_ids)
        payload["trigger"] = trigger
        return payload

    def _emit_event(self, event_type: str, data: dict) -> None:
        self.events.append((event_type, data))
        self._publish_runtime_lifecycle_event(event_type, data)

    def _run_investigation_thread(self, *args) -> None:
        return

    def _run_validation_thread(self, *args) -> None:
        return

    def _run_scale_up_thread(self, *args) -> None:
        return

    def _run_evolution_thread(self, *args) -> None:
        return

    def _run_novelty_thread(self, *args) -> None:
        return


class _ModeStartRunner(_ControlStartMixin):
    def __init__(self, notebook_path: str | Path) -> None:
        self.notebook_path = str(notebook_path)
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._progress = LiveProgress()
        self.aria = get_aria()
        self.events = []

    @property
    def is_running(self) -> bool:
        return False

    def _ensure_math_spaces(self) -> None:
        return

    def _make_notebook(self):
        class _FakeNotebook:
            def get_tiers_for_result_ids(self, result_ids):
                return {rid: "screening" for rid in result_ids}

            def close(self):
                return

        return _FakeNotebook()

    def _build_hypothesis_metadata(self, **kwargs):
        return dict(kwargs)

    def _start_preregistered_experiment(self, **kwargs):
        return "mode-exp"

    def _emit_event(self, event_type: str, data: dict) -> None:
        self.events.append((event_type, data))

    def _run_investigation_thread(self, *args, **kwargs) -> None:
        return


class _ValidationPublisher(_ExecutionValidationMixin):
    def __init__(self, notebook_path: str | Path) -> None:
        self.notebook_path = str(notebook_path)


class _SearchPublisher(_ExecutionSearchMixin):
    def __init__(self, notebook_path: str | Path) -> None:
        self.notebook_path = str(notebook_path)


class _InvestigationPublisher(_ExecutionInvestigationMixin):
    def __init__(self, notebook_path: str | Path) -> None:
        self.notebook_path = str(notebook_path)


class _ScreeningPublisher(_ExecutionScreeningMixin):
    def __init__(self, notebook_path: str | Path) -> None:
        self.notebook_path = str(notebook_path)


class _ContinuousValidationPublisher(_ContinuousValidationMixin):
    def __init__(self, notebook_path: str | Path) -> None:
        self.notebook_path = str(notebook_path)


class _ContinuousInvestigationPublisher(_ContinuousInvestigationMixin):
    def __init__(self, notebook_path: str | Path) -> None:
        self.notebook_path = str(notebook_path)


class _ContinuousModesPublisher(_ContinuousModesMixin):
    def __init__(self, notebook_path: str | Path) -> None:
        self.notebook_path = str(notebook_path)
        self.aria = get_aria()


class _CycleAbortRunner(_CycleMixin):
    def __init__(self, notebook_path: str | Path) -> None:
        self.notebook_path = str(notebook_path)
        self._lock = threading.Lock()
        self._progress = LiveProgress()
        self.aria = get_aria()

    @property
    def progress(self) -> LiveProgress:
        return self._progress


class _ShutdownRunner(_CoreMixin):
    def __init__(self, notebook_path: str | Path) -> None:
        self.notebook_path = str(notebook_path)
        self._lock = threading.Lock()
        self._progress = LiveProgress()
        self._shutdown_handler_registered = False

    @property
    def progress(self) -> LiveProgress:
        return self._progress


def test_status_prefers_registry_state_over_notebook_fallback(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        notebook_exp_id = nb.start_experiment(
            "synthesis",
            {"mode": "single", "n_programs": 2},
            "notebook fallback",
        )
        publish_lifecycle_event(
            notebook_path=nb.db_path,
            event_type="experiment_started",
            producer="test",
            run_id="registry-exp",
            sequence=1,
            payload={
                "experiment_type": "screening",
                "hypothesis": "registry truth",
                "config": {"mode": "registry"},
                "started_at": 10.0,
            },
        )

        status = resolve_runner_status(nb, _IdleRunner())

        assert notebook_exp_id != "registry-exp"
        assert status["is_running"] is True
        assert status["progress"]["experiment_id"] == "registry-exp"
        assert status["progress"]["run_source"] == "runtime_lifecycle_registry"
        assert status["external_snapshot"]["source"] == "runtime_lifecycle_registry"
    finally:
        nb.close()


def test_started_event_sets_running_without_notebook_fallback(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        publish_lifecycle_event(
            notebook_path=nb.db_path,
            event_type="experiment_started",
            producer="test",
            run_id="exp-started",
            sequence=1,
            payload={
                "experiment_type": "screening",
                "hypothesis": "registry only",
                "config": {"mode": "screening"},
            },
        )

        status = resolve_runner_status(nb, _IdleRunner())

        assert status["is_running"] is True
        assert status["progress"]["experiment_id"] == "exp-started"
        assert status["progress"]["current_stage"] == "runtime_events"
        assert status["external_snapshot"]["source"] == "runtime_lifecycle_registry"
    finally:
        nb.close()


def test_start_failed_event_does_not_leave_phantom_running_state(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        publish_lifecycle_event(
            notebook_path=nb.db_path,
            event_type="experiment_start_requested",
            producer="test",
            run_id="launch-1",
            sequence=1,
            payload={"mode": "single"},
        )
        publish_lifecycle_event(
            notebook_path=nb.db_path,
            event_type="experiment_start_failed",
            producer="test",
            run_id="launch-1",
            sequence=2,
            payload={"mode": "single", "error": "boom"},
        )

        status = resolve_runner_status(nb, _IdleRunner())
        registry = get_runtime_event_services(nb.db_path).registry

        assert registry.get("launch-1") is not None
        assert registry.get("launch-1").status == "failed"
        assert status["is_running"] is False
        assert status["external_snapshot"] is None
    finally:
        nb.close()


def test_cleanup_stale_experiments_clears_registry_running_state(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        exp_id = nb.start_experiment(
            "synthesis",
            {"mode": "single", "n_programs": 1},
            "stale cleanup registry",
        )
        started_at = time.time() - (2 * 60 * 60)
        nb.conn.execute(
            "UPDATE experiments SET started_at = ? WHERE experiment_id = ?",
            (started_at, exp_id),
        )
        nb.conn.commit()

        status_before = resolve_runner_status(nb, _IdleRunner())
        cleaned = nb.cleanup_stale_experiments(timeout_minutes=60)
        status_after = resolve_runner_status(nb, _IdleRunner())
        registry = get_runtime_event_services(nb.db_path).registry

        assert status_before["is_running"] is True
        assert cleaned == 1
        assert registry.get(exp_id) is not None
        assert registry.get(exp_id).status == "failed"
        assert status_after["is_running"] is False
        assert status_after["external_snapshot"] is None
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_projector_bootstrap_replays_spool_into_notebook(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        publish_lifecycle_event(
            notebook_path=nb.db_path,
            event_type="experiment_started",
            producer="test",
            run_id="projected-exp",
            sequence=1,
            payload={
                "experiment_type": "screening",
                "hypothesis": "project me",
                "config": {"mode": "screening", "seed": 7},
                "started_at": 12.0,
            },
        )

        services = start_runtime_event_projector(nb.db_path)
        row = None
        deadline = time.time() + 2.0
        while time.time() < deadline:
            row = nb.conn.execute(
                "SELECT status, config_json FROM experiments WHERE experiment_id = ?",
                ("projected-exp",),
            ).fetchone()
            if row is not None:
                break
            time.sleep(0.05)

        assert row is not None
        assert row["status"] == "running"
        assert json.loads(row["config_json"]) == {"mode": "screening", "seed": 7}
        assert services.projector_health().iterations >= 1
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_notebook_start_experiment_publishes_runtime_event(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        exp_id = nb.start_experiment(
            "synthesis",
            {"mode": "single", "n_programs": 4},
            "notebook started",
        )

        status = resolve_runner_status(nb, _IdleRunner())
        registry = get_runtime_event_services(nb.db_path).registry

        assert registry.get(exp_id) is not None
        assert registry.get(exp_id).status == "running"
        assert status["is_running"] is True
        assert status["progress"]["experiment_id"] == exp_id
        assert status["progress"]["run_source"] == "runtime_lifecycle_registry"
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_notebook_complete_experiment_clears_running_state(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        exp_id = nb.start_experiment(
            "synthesis",
            {"mode": "single", "n_programs": 2},
            "notebook complete",
        )

        nb.complete_experiment(
            exp_id,
            {
                "total": 2,
                "stage0_passed": 1,
                "stage05_passed": 1,
                "stage1_passed": 1,
                "best_loss_ratio": 0.75,
                "best_novelty_score": 0.5,
            },
            aria_summary="done",
            insights=["ok"],
        )

        status = resolve_runner_status(nb, _IdleRunner())
        registry = get_runtime_event_services(nb.db_path).registry
        row = nb.conn.execute(
            "SELECT status FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()

        assert registry.get(exp_id) is not None
        assert registry.get(exp_id).status == "completed"
        assert row is not None
        assert row["status"] == "completed"
        assert status["is_running"] is False
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_notebook_fail_experiment_clears_running_state(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        exp_id = nb.start_experiment(
            "synthesis",
            {"mode": "single", "n_programs": 1},
            "notebook fail",
        )

        nb.fail_experiment(exp_id, "boom", results={"total": 1})

        status = resolve_runner_status(nb, _IdleRunner())
        registry = get_runtime_event_services(nb.db_path).registry
        row = nb.conn.execute(
            "SELECT status, aria_summary FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()

        assert registry.get(exp_id) is not None
        assert registry.get(exp_id).status == "failed"
        assert row is not None
        assert row["status"] == "failed"
        assert "FAILED:" in str(row["aria_summary"] or "")
        assert status["is_running"] is False
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_explicit_terminal_publish_then_notebook_sink_does_not_double_append(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        exp_id = nb.start_experiment(
            "validation",
            {"mode": "validation", "n_programs": 1},
            "validation dedupe",
        )
        publisher = _ValidationPublisher(nb.db_path)

        publisher._publish_validation_terminal_event(
            event_type="experiment_failed",
            exp_id=exp_id,
            payload={
                "completed_at": time.time(),
                "error": "explicit boundary failure",
                "results": None,
                "phase": "validation",
            },
        )
        nb.fail_experiment(exp_id, "explicit boundary failure")

        records = [
            record.event
            for record in get_runtime_event_services(nb.db_path).spool.replay()
            if record.event.run_id == exp_id and record.event.event_type == "experiment_failed"
        ]

        assert len(records) == 1
        assert records[0].payload["error"] == "explicit boundary failure"
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_explicit_search_terminal_publish_then_notebook_sink_does_not_double_append(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        exp_id = nb.start_experiment(
            "novelty",
            {"mode": "novelty", "n_programs": 1},
            "search dedupe",
        )
        publisher = _SearchPublisher(nb.db_path)

        publisher._publish_search_terminal_event(
            event_type="experiment_failed",
            exp_id=exp_id,
            payload={
                "completed_at": time.time(),
                "error": "search boundary failure",
                "results": None,
                "mode": "novelty",
            },
        )
        nb.fail_experiment(exp_id, "search boundary failure")

        records = [
            record.event
            for record in get_runtime_event_services(nb.db_path).spool.replay()
            if record.event.run_id == exp_id and record.event.event_type == "experiment_failed"
        ]

        assert len(records) == 1
        assert records[0].payload["error"] == "search boundary failure"
        assert records[0].payload["mode"] == "novelty"
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_explicit_investigation_terminal_publish_then_notebook_sink_does_not_double_append(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        exp_id = nb.start_experiment(
            "investigation",
            {"mode": "investigation", "n_programs": 1},
            "investigation dedupe",
        )
        publisher = _InvestigationPublisher(nb.db_path)

        publisher._publish_investigation_terminal_event(
            event_type="experiment_failed",
            exp_id=exp_id,
            payload={
                "completed_at": time.time(),
                "error": "investigation boundary failure",
                "results": None,
                "mode": "investigation",
            },
        )
        nb.fail_experiment(exp_id, "investigation boundary failure")

        records = [
            record.event
            for record in get_runtime_event_services(nb.db_path).spool.replay()
            if record.event.run_id == exp_id and record.event.event_type == "experiment_failed"
        ]

        assert len(records) == 1
        assert records[0].payload["error"] == "investigation boundary failure"
        assert records[0].payload["mode"] == "investigation"
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_explicit_screening_terminal_publish_then_notebook_sink_does_not_double_append(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        exp_id = nb.start_experiment(
            "synthesis",
            {"mode": "single", "n_programs": 1},
            "screening dedupe",
        )
        publisher = _ScreeningPublisher(nb.db_path)

        publisher._publish_screening_terminal_event(
            event_type="experiment_failed",
            exp_id=exp_id,
            payload={
                "completed_at": time.time(),
                "error": "screening boundary failure",
                "results": None,
                "mode": "screening",
            },
        )
        nb.fail_experiment(exp_id, "screening boundary failure")

        records = [
            record.event
            for record in get_runtime_event_services(nb.db_path).spool.replay()
            if record.event.run_id == exp_id and record.event.event_type == "experiment_failed"
        ]

        assert len(records) == 1
        assert records[0].payload["error"] == "screening boundary failure"
        assert records[0].payload["mode"] == "screening"
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_explicit_continuous_validation_terminal_publish_then_notebook_sink_does_not_double_append(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        exp_id = nb.start_experiment(
            "validation",
            {"mode": "continuous_validation", "n_programs": 1},
            "continuous validation dedupe",
        )
        publisher = _ContinuousValidationPublisher(nb.db_path)

        publisher._publish_continuous_validation_terminal_event(
            event_type="experiment_failed",
            exp_id=exp_id,
            payload={
                "completed_at": time.time(),
                "error": "continuous validation boundary failure",
                "results": None,
                "mode": "continuous_validation",
            },
        )
        nb.fail_experiment(exp_id, "continuous validation boundary failure")

        records = [
            record.event
            for record in get_runtime_event_services(nb.db_path).spool.replay()
            if record.event.run_id == exp_id and record.event.event_type == "experiment_failed"
        ]

        assert len(records) == 1
        assert records[0].payload["error"] == "continuous validation boundary failure"
        assert records[0].payload["mode"] == "continuous_validation"
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_explicit_continuous_investigation_terminal_publish_then_notebook_sink_does_not_double_append(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        exp_id = nb.start_experiment(
            "investigation",
            {"mode": "continuous_investigation", "n_programs": 1},
            "continuous investigation dedupe",
        )
        publisher = _ContinuousInvestigationPublisher(nb.db_path)

        publisher._publish_continuous_investigation_terminal_event(
            event_type="experiment_failed",
            exp_id=exp_id,
            payload={
                "completed_at": time.time(),
                "error": "continuous investigation boundary failure",
                "results": None,
                "mode": "continuous_investigation",
            },
        )
        nb.fail_experiment(exp_id, "continuous investigation boundary failure")

        records = [
            record.event
            for record in get_runtime_event_services(nb.db_path).spool.replay()
            if record.event.run_id == exp_id and record.event.event_type == "experiment_failed"
        ]

        assert len(records) == 1
        assert records[0].payload["error"] == "continuous investigation boundary failure"
        assert records[0].payload["mode"] == "continuous_investigation"
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_explicit_continuous_modes_terminal_publish_uses_canonical_completed_event(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        exp_id = nb.start_experiment(
            "synthesis",
            {"mode": "continuous_synthesis", "n_programs": 1},
            "continuous modes canonical terminal",
        )
        publisher = _ContinuousModesPublisher(nb.db_path)

        publisher._publish_continuous_modes_terminal_event(
            event_type="experiment_completed",
            exp_id=exp_id,
            payload={
                "completed_at": time.time(),
                "results": {"total": 1, "stage1_passed": 1},
                "aria_summary": "continuous done",
                "aria_mood": "focused",
                "insights": ["ok"],
                "llm_analysis": "done",
                "mode": "synthesis",
            },
        )
        nb.complete_experiment(
            exp_id,
            {"total": 1, "stage1_passed": 1},
            aria_summary="continuous done",
            insights=["ok"],
        )

        records = [
            record.event
            for record in get_runtime_event_services(nb.db_path).spool.replay()
            if record.event.run_id == exp_id
            and record.event.event_type == "experiment_completed"
        ]
        registry = get_runtime_event_services(nb.db_path).registry

        assert len(records) == 1
        assert records[0].event_type == "experiment_completed"
        assert registry.get(exp_id) is not None
        assert registry.get(exp_id).status == "completed"
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_cycle_abort_compensation_publishes_failure_without_double_append(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        exp_id = nb.start_experiment(
            "continuous_investigation",
            {"mode": "continuous_investigation", "n_programs": 1},
            "cycle compensation",
        )
        runner = _CycleAbortRunner(nb.db_path)
        runner._progress.experiment_id = exp_id
        runner._progress.status = "running"

        failed_exp_id = runner._fail_active_cycle_experiment(
            nb,
            "cycle watchdog timeout",
        )
        records = [
            record.event
            for record in get_runtime_event_services(nb.db_path).spool.replay()
            if record.event.run_id == exp_id and record.event.event_type == "experiment_failed"
        ]

        assert failed_exp_id == exp_id
        assert len(records) == 1
        assert records[0].payload["error"] == "cycle watchdog timeout"
        assert records[0].payload["reason"] == "cycle_abort_compensation"
        assert records[0].payload["mode"] == "continuous_investigation"
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_shutdown_compensation_interrupts_without_direct_runner_sql(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        exp_id = nb.start_experiment(
            "continuous_investigation",
            {"mode": "continuous_investigation", "n_programs": 1},
            "shutdown compensation",
        )
        runner = _ShutdownRunner(nb.db_path)
        runner._progress.experiment_id = exp_id
        runner._progress.status = "running"

        runner._register_shutdown_handler()
        atexit._run_exitfuncs()

        row = nb.conn.execute(
            "SELECT status, aria_summary FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        records = [
            record.event
            for record in get_runtime_event_services(nb.db_path).spool.replay()
            if record.event.run_id == exp_id and record.event.event_type == "experiment_failed"
        ]

        assert row is not None
        assert row["status"] == "interrupted"
        assert row["aria_summary"] == "FAILED: Interrupted by shutdown"
        assert records
        assert records[-1].payload["error"] == "Interrupted by shutdown"
        assert records[-1].payload["interrupted"] is True
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_notebook_cancel_experiment_publishes_terminal_failure(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        exp_id = nb.start_experiment(
            "synthesis",
            {"mode": "single", "n_programs": 1},
            "notebook cancel",
        )

        assert nb.cancel_experiment(exp_id) is True

        status = resolve_runner_status(nb, _IdleRunner())
        registry = get_runtime_event_services(nb.db_path).registry
        row = nb.conn.execute(
            "SELECT status, aria_summary FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()

        assert registry.get(exp_id) is not None
        assert registry.get(exp_id).status == "failed"
        assert row is not None
        assert row["status"] == "failed"
        assert "Cancelled by user" in str(row["aria_summary"] or "")
        assert status["is_running"] is False
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_log_learning_event_mirrors_best_effort_runtime_event(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        nb.log_learning_event(
            "grammar_update",
            "adjusted grammar weights",
            old_weights={"a": 0.1},
            new_weights={"a": 0.2},
            experiment_id="exp-telemetry",
            reason="unit-test",
        )

        records = list(get_runtime_event_services(nb.db_path).spool.replay())
        event = records[-1].event

        assert event.event_type == "learning_event_logged"
        assert event.durability == "best_effort"
        assert event.run_id == "exp-telemetry"
        assert event.payload["log_event_type"] == "grammar_update"
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_start_continuous_publishes_session_event(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        runner = _ContinuousRunner(nb.db_path)
        from research.scientist.runner._types import RunConfig

        session_id = runner.start_continuous(RunConfig(n_programs=3))
        if runner._thread is not None:
            runner._thread.join(timeout=1.0)

        records = list(get_runtime_event_services(nb.db_path).spool.replay())
        session_events = [r.event for r in records if r.event.event_type == "continuous_session_started"]

        assert session_id == "continuous"
        assert session_events
        assert session_events[-1].durability == "best_effort"
        assert session_events[-1].payload["mode"] == "continuous"
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_continuous_limit_reached_publishes_completed_session_event(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        runner = _ContinuousLoopRunner(nb.db_path)
        config = RunConfig()

        runner._handle_continuous_limit_reached(
            config=config,
            stop_reason="max_experiments reached",
            n_experiments=3,
            t_start=time.time() - 120.0,
            distiller=None,
        )

        records = list(get_runtime_event_services(nb.db_path).spool.replay())
        session_events = [
            r.event for r in records if r.event.event_type == "continuous_session_completed"
        ]

        assert session_events
        assert session_events[-1].run_id == "continuous"
        assert session_events[-1].payload["reason"] == "max_experiments reached"
        assert session_events[-1].payload["experiments_completed"] == 3
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_continuous_stop_publishes_stopped_session_event(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        runner = _ContinuousLoopRunner(nb.db_path)
        runner._stop_event.set()
        config = RunConfig(keep_checkpoints=True)

        class _Checkpoint:
            def cleanup(self, _experiment_id):
                raise AssertionError("cleanup should not run when stopped")

        runner._finish_continuous_run(
            config=config,
            ckpt=_Checkpoint(),
            resume_id=None,
            n_experiments=2,
            t_start=time.time() - 90.0,
            distiller=None,
        )

        records = list(get_runtime_event_services(nb.db_path).spool.replay())
        session_events = [
            r.event for r in records if r.event.event_type == "continuous_session_stopped"
        ]

        assert session_events
        assert session_events[-1].run_id == "continuous"
        assert session_events[-1].payload["experiments_completed"] == 2
        assert session_events[-1].payload["mode"] == "continuous"
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_continuous_fatal_thread_publishes_failed_session_event(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        runner = _ContinuousLoopRunner(nb.db_path)

        def _boom(_config):
            raise RuntimeError("fatal continuous failure")

        runner._run_continuous_thread_inner = _boom
        runner._run_continuous_thread(RunConfig())

        records = list(get_runtime_event_services(nb.db_path).spool.replay())
        session_events = [
            r.event for r in records if r.event.event_type == "continuous_session_failed"
        ]

        assert session_events
        assert session_events[-1].run_id == "continuous"
        assert "fatal continuous failure" in session_events[-1].payload["error"]
        assert runner.progress.status == "failed"
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_start_investigation_emits_canonical_and_mode_specific_start_events(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        from research.scientist.runner._types import RunConfig

        runner = _ModeStartRunner(nb.db_path)
        exp_id = runner.start_investigation(["rid-1"], RunConfig(), hypothesis="inspect")
        event_types = [event_type for event_type, _data in runner.events]
        canonical = [data for event_type, data in runner.events if event_type == "experiment_started"]
        mode_specific = [data for event_type, data in runner.events if event_type == "investigation_started"]

        assert exp_id == "mode-exp"
        assert "experiment_started" in event_types
        assert "investigation_started" in event_types
        assert canonical[-1]["experiment_id"] == "mode-exp"
        assert canonical[-1]["mode"] == "investigation"
        assert mode_specific[-1]["experiment_id"] == "mode-exp"
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_resume_publishes_started_for_failed_run_id(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        exp_id = nb.start_experiment(
            "synthesis",
            {"mode": "single", "n_programs": 1},
            "resume me",
        )
        nb.fail_experiment(exp_id, "boom", results={"total": 1})

        runner = _ResumeRunner(nb.db_path)
        resumed_id = runner.start_resume(exp_id)
        if runner._thread is not None:
            runner._thread.join(timeout=1.0)

        status = resolve_runner_status(nb, _IdleRunner())
        registry = get_runtime_event_services(nb.db_path).registry
        row = nb.conn.execute(
            "SELECT status FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()

        assert resumed_id == exp_id
        assert registry.get(exp_id) is not None
        assert registry.get(exp_id).status == "running"
        assert row is not None
        assert row["status"] == "running"
        assert status["is_running"] is True
        assert status["progress"]["experiment_id"] == exp_id
        assert any(event[0] == "experiment_resuming" for event in runner.events)
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_resume_publishes_started_for_interrupted_run_id(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        exp_id = nb.start_experiment(
            "synthesis",
            {"mode": "single", "n_programs": 1},
            "resume interrupted",
        )
        nb.conn.execute(
            "UPDATE experiments SET status = 'interrupted' WHERE experiment_id = ?",
            (exp_id,),
        )
        nb.conn.commit()

        runner = _ResumeRunner(nb.db_path)
        resumed_id = runner.start_resume(exp_id)
        if runner._thread is not None:
            runner._thread.join(timeout=1.0)

        status = resolve_runner_status(nb, _IdleRunner())
        registry = get_runtime_event_services(nb.db_path).registry
        row = nb.conn.execute(
            "SELECT status FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        resume_events = [
            record.event
            for record in get_runtime_event_services(nb.db_path).spool.replay()
            if record.event.run_id == exp_id
            and record.event.event_type == "experiment_started"
            and record.event.payload.get("resume") is True
        ]

        assert resumed_id == exp_id
        assert registry.get(exp_id) is not None
        assert registry.get(exp_id).status == "running"
        assert resume_events
        assert resume_events[-1].payload["resumed_from_status"] == "interrupted"
        assert row is not None
        assert row["status"] == "running"
        assert status["is_running"] is True
        assert status["progress"]["experiment_id"] == exp_id
        assert any(event[0] == "experiment_resuming" for event in runner.events)
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


@pytest.mark.parametrize(
    ("method_name", "method_args", "expected_mode_event"),
    [
        ("start_investigation", (["rid-1"], RunConfig(), "investigate"), "investigation_started"),
        ("start_validation", (["rid-1"], RunConfig(), "validate"), "validation_started"),
        ("start_scale_up", (["rid-1"], RunConfig(), "scale"), "scale_up_started"),
        ("start_evolution", (RunConfig(), "evolve"), "evolution_started"),
        ("start_novelty_search", (RunConfig(), "novel"), "novelty_started"),
    ],
)
def test_non_synthesis_start_modes_publish_canonical_started(
    tmp_path, method_name, method_args, expected_mode_event
):
    nb = _make_status_notebook(tmp_path)
    path = nb.db_path
    try:
        runner = _StartedModeRunner(path)
        method = getattr(runner, method_name)

        exp_id = method(*method_args)
        if runner._thread is not None:
            runner._thread.join(timeout=1.0)

        registry = get_runtime_event_services(path).registry
        state = registry.get(exp_id)

        assert state is not None
        assert state.run_id == exp_id
        assert state.status == "running"
        assert state.last_event.event_type == "experiment_started"
        assert state.last_event.payload["hypothesis"] == method_args[-1]
        assert "config" in state.last_event.payload
        assert any(event[0] == "experiment_started" for event in runner.events)
        assert any(event[0] == expected_mode_event for event in runner.events)
    finally:
        stop_runtime_event_services(path)
        nb.close()


def test_status_prefers_projected_lifecycle_before_legacy_notebook_heuristic(tmp_path):
    nb = _make_status_notebook(tmp_path)
    try:
        start_runtime_event_projector(nb.db_path)
        now = time.time()
        nb.conn.execute(
            """
            INSERT INTO experiments (
                experiment_id,
                timestamp,
                experiment_type,
                status,
                hypothesis,
                config_json,
                started_at,
                n_programs_generated,
                n_stage0_passed,
                n_stage05_passed,
                n_stage1_passed
            ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "projected-running",
                now,
                "screening",
                "projected hypothesis",
                json.dumps({"mode": "projected", "n_programs": 6}),
                now,
                3,
                1,
                1,
                0,
            ),
        )
        nb.conn.execute(
            """
            INSERT INTO applied_runtime_events (event_id, event_type, run_id, applied_at)
            VALUES (?, ?, ?, ?)
            """,
            ("evt-1", "experiment_started", "projected-running", now),
        )
        nb.conn.commit()

        status = resolve_runner_status(nb, _IdleRunner())

        assert status["is_running"] is True
        assert status["progress"]["experiment_id"] == "projected-running"
        assert status["progress"]["run_source"] == "projected_runtime_lifecycle"
        assert status["progress"]["current_stage"] == "runtime_projector"
        assert status["external_snapshot"]["source"] == "projected_runtime_lifecycle"
    finally:
        stop_runtime_event_services(nb.db_path)
        nb.close()


def test_projector_worker_background_start_and_stop():
    calls = []

    def replay_once():
        calls.append(time.time())
        class Status:
            degraded = False
            applied_count = 0
        return Status()

    worker = ProjectorWorker(replay_once, interval_seconds=0.05)
    worker.start()
    time.sleep(0.12)
    worker.stop(timeout=1.0)
    health = worker.health_snapshot()

    assert len(calls) >= 1
    assert health.iterations >= 1


def test_runtime_event_services_are_singleton_per_notebook_root(tmp_path):
    nb = _make_status_notebook(tmp_path)
    path = nb.db_path
    try:
        first = get_runtime_event_services(path)
        second = get_runtime_event_services(path)

        assert first is second
        assert first.projector_worker.health_snapshot().running is False
    finally:
        nb.close()
        stop_runtime_event_services(path)


def test_start_runtime_event_projector_bootstraps_worker_and_projects(tmp_path):
    nb = _make_status_notebook(tmp_path)
    path = nb.db_path
    try:
        services = start_runtime_event_projector(path)
        publish_lifecycle_event(
            notebook_path=path,
            event_type="experiment_started",
            producer="test",
            run_id="worker-exp",
            sequence=1,
            payload={
                "experiment_type": "screening",
                "config": {"mode": "worker"},
            },
        )

        deadline = time.time() + 2.0
        row = None
        while time.time() < deadline:
            row = nb.conn.execute(
                "SELECT status FROM experiments WHERE experiment_id = ?",
                ("worker-exp",),
            ).fetchone()
            if row is not None:
                break
            time.sleep(0.05)

        assert row is not None
        assert row["status"] == "running"
        assert services.projector_health().running is True
    finally:
        nb.close()
        stop_runtime_event_services(path)
