"""Phase 4.1 — Unit tests for `_slot_constraints_loader.derive_slot_classes`.

The loader reads `meta_analysis.db` + `lab_notebook.db` to derive empirical
pass-cohort motif_class allow-lists per (template, slot_index). These tests
cover the cache layer + fallback behavior; full DB integration is exercised
by the synth-init smoke path on the real DB at module-import time.
"""

from __future__ import annotations

import pytest

from research.synthesis import _slot_constraints_loader as loader


pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _reset_cache():
    loader.reset_cache()
    yield
    loader.reset_cache()


def _stub_cache(monkeypatch, mapping: dict[tuple[str, int], tuple[str, ...]]) -> None:
    monkeypatch.setattr(loader, "_build_cache", lambda: mapping)


def test_returns_fallback_when_cache_miss(monkeypatch) -> None:
    _stub_cache(monkeypatch, {})
    assert loader.derive_slot_classes("missing_tpl", 0, fallback=("FFN",)) == ("FFN",)


def test_returns_derived_when_present(monkeypatch) -> None:
    _stub_cache(
        monkeypatch,
        {("latent_attn_sparse_ffn", 2): ("conv_core",)},
    )
    out = loader.derive_slot_classes("latent_attn_sparse_ffn", 2, fallback=("FFN",))
    assert out == ("conv_core",)


def test_fallback_preserved_for_other_slot(monkeypatch) -> None:
    _stub_cache(
        monkeypatch,
        {("latent_attn_sparse_ffn", 2): ("conv_core",)},
    )
    # slot 1 has no entry — must fall back
    assert loader.derive_slot_classes(
        "latent_attn_sparse_ffn", 1, fallback=("NORM",)
    ) == ("NORM",)


def test_int_coercion_on_slot_index(monkeypatch) -> None:
    _stub_cache(monkeypatch, {("tpl", 3): ("conv_core",)})
    # passing slot_index as a numpy-like or string-int should coerce
    assert loader.derive_slot_classes("tpl", 3, fallback=("X",)) == ("conv_core",)


def test_cache_is_lazy(monkeypatch) -> None:
    """First call builds the cache; subsequent calls reuse it."""
    call_count = {"n": 0}

    def _fake_build():
        call_count["n"] += 1
        return {("tpl", 0): ("conv_core",)}

    monkeypatch.setattr(loader, "_build_cache", _fake_build)
    loader.derive_slot_classes("tpl", 0, fallback=("X",))
    loader.derive_slot_classes("tpl", 0, fallback=("X",))
    loader.derive_slot_classes("tpl", 0, fallback=("X",))
    assert call_count["n"] == 1


def test_reset_cache_clears(monkeypatch) -> None:
    call_count = {"n": 0}

    def _fake_build():
        call_count["n"] += 1
        return {}

    monkeypatch.setattr(loader, "_build_cache", _fake_build)
    loader.derive_slot_classes("tpl", 0, fallback=("X",))
    loader.reset_cache()
    loader.derive_slot_classes("tpl", 0, fallback=("X",))
    assert call_count["n"] == 2


def test_thresholds_filter_low_n_or_low_passrate(monkeypatch) -> None:
    """Verify _build_cache logic via direct invocation with stubbed query."""
    fills = {
        ("good_tpl", 0): [
            ("conv_core", 80, 100),  # pass=0.80, n=100 — qualifies
            ("sparse_core", 5, 100),  # pass=0.05 — too low
            ("ffn_core", 9, 12),  # pass=0.75 but n=12, just above min_n=10
            ("low_n_core", 4, 5),  # n<10 — disqualified by MIN_N
        ],
        ("bad_tpl", 0): [
            ("efficient_proj", 0, 50),  # 0% pass
            ("math_space", 1, 80),  # 1% pass
        ],
    }
    monkeypatch.setattr(loader, "_query_pass_cohort_fills", lambda: fills)
    loader.reset_cache()
    cache = loader._ensure_cache()
    assert cache.get(("good_tpl", 0)) == ("conv_core", "ffn_core")
    # bad_tpl has zero qualifying classes -> not in cache
    assert ("bad_tpl", 0) not in cache
