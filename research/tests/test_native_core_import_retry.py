from __future__ import annotations

import sys
import types
from pathlib import Path

from research.scientist.native import core as native_core
from research.scientist import native_runner_adapter


def test_cython_bridge_negative_cache_does_not_block_retry(monkeypatch):
    dummy = types.ModuleType("aria_bridge")
    monkeypatch.setitem(sys.modules, "aria_bridge", dummy)
    native_core._cython_bridge_cache = None

    bridge = native_core._try_import_cython_bridge()

    assert bridge is dummy
    assert native_core._cython_bridge_cache is dummy


def test_rust_scheduler_negative_cache_does_not_block_retry(monkeypatch):
    dummy = types.ModuleType("aria_scheduler")
    monkeypatch.setitem(sys.modules, "aria_scheduler", dummy)
    native_core._rust_scheduler_cache = None
    monkeypatch.setattr(Path, "exists", lambda self: False)

    rust = native_core._try_import_rust_scheduler()

    assert rust is dummy
    assert native_core._rust_scheduler_cache is dummy


def test_rust_scheduler_prefers_repo_local_build(monkeypatch):
    local_so = (
        Path(native_core.__file__).resolve().parents[2]
        / "runtime"
        / "native"
        / "rust"
        / "aria-scheduler"
        / "target"
        / "release"
        / "libaria_scheduler.so"
    )
    if not local_so.exists():
        raise AssertionError("expected local aria_scheduler build artifact to exist")

    calls: list[tuple[str, str]] = []

    class FakeLoader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.loaded_from = "local-build"

    monkeypatch.setattr(native_core, "_rust_scheduler_cache", None)
    monkeypatch.setattr(
        native_core.importlib.util,
        "spec_from_file_location",
        lambda name, path: (
            calls.append((name, str(path))),
            types.SimpleNamespace(loader=FakeLoader()),
        )[1],
    )
    monkeypatch.setattr(
        native_core.importlib.util,
        "module_from_spec",
        lambda spec: types.ModuleType("aria_scheduler"),
    )

    rust = native_core._try_import_rust_scheduler()

    assert rust.loaded_from == "local-build"
    assert calls == [("aria_scheduler", str(local_so))]


def test_detect_adapter_state_accepts_repo_native_runtime_build(monkeypatch):
    native_build = (
        Path(native_runner_adapter.__file__).resolve().parents[2]
        / "research"
        / "runtime"
        / "native"
        / "build"
        / "libaria_native_runtime.so"
    )
    if not native_build.exists():
        raise AssertionError(
            "expected repo-local native runtime build artifact to exist"
        )

    state = native_runner_adapter.detect_adapter_state()

    assert state.enabled is True
    assert state.designer_runtime_available is True
    assert state.reason == f"ready:{native_build}"
