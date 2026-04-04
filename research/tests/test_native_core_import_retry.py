from __future__ import annotations

import sys
import types

from research.scientist.native import core as native_core


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

    rust = native_core._try_import_rust_scheduler()

    assert rust is dummy
    assert native_core._rust_scheduler_cache is dummy
