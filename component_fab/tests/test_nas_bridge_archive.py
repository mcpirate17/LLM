"""Tests for archive-guided NAS sampling in the grammar→fab bridge.

The measurement path (building modules from cached graphs) is monkeypatched out;
these exercise the steering logic and the fail-safe fallback to random sampling.
One test runs the real (pure, torch-free) archive→GrammarConfig path.
"""

from __future__ import annotations

import pytest

import component_fab.proposer.nas_bridge as nb


def test_exploration_config_from_real_archive_targets_empty_niches() -> None:
    from research.synthesis.archive_guided import exploration_config_from_archive
    from research.synthesis.quality_diversity import MapElitesArchive

    archive = MapElitesArchive()  # 27-niche default behaviour space
    archive.add(
        "only",
        {
            "long_range_reach": 0.8,
            "content_dependence": 0.5,
            "content_match_gating": 0.2,
        },
        1.0,
    )
    cfg, guidance = exploration_config_from_archive(archive, model_dim=32)

    assert guidance.reachable_empty > 0
    assert cfg is not None
    assert cfg.exploration_targets  # grammar is now biased toward the holes
    assert cfg.model_dim == 32


def test_nas_graph_specs_passes_exploration_cfg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(nb, "_exploration_config", lambda dim, mg: sentinel)

    def fake_fresh(n, dim, seed, seen, cfg=None):
        captured["cfg"] = cfg
        return []

    monkeypatch.setattr(nb, "_fresh_grammar_specs", fake_fresh)
    nb.nas_graph_specs(n_fresh=2, dim=32, archive_guided=True, include_db_winners=False)
    assert captured["cfg"] is sentinel


def test_nas_graph_specs_falls_back_when_archive_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    # No cached population yet → build_nas_archive returns None → cfg is None.
    monkeypatch.setattr(nb, "build_nas_archive", lambda dim, max_graphs=64: None)

    def fake_fresh(n, dim, seed, seen, cfg=None):
        captured["cfg"] = cfg
        return []

    monkeypatch.setattr(nb, "_fresh_grammar_specs", fake_fresh)
    nb.nas_graph_specs(n_fresh=2, dim=32, archive_guided=True, include_db_winners=False)
    assert captured["cfg"] is None


def test_nas_graph_specs_random_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_a, **_k):
        raise AssertionError("archive guidance must not run when disabled")

    monkeypatch.setattr(nb, "_exploration_config", boom)
    captured: dict[str, object] = {}

    def fake_fresh(n, dim, seed, seen, cfg=None):
        captured["cfg"] = cfg
        return []

    monkeypatch.setattr(nb, "_fresh_grammar_specs", fake_fresh)
    nb.nas_graph_specs(
        n_fresh=2, dim=32, archive_guided=False, include_db_winners=False
    )
    assert captured["cfg"] is None


def test_archive_guidance_failure_is_nonfatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(dim, mg):
        raise RuntimeError("measurement blew up")

    monkeypatch.setattr(nb, "_exploration_config", boom)
    captured: dict[str, object] = {}

    def fake_fresh(n, dim, seed, seen, cfg=None):
        captured["cfg"] = cfg
        return []

    monkeypatch.setattr(nb, "_fresh_grammar_specs", fake_fresh)
    # Must not raise — guidance is best-effort, falls back to random.
    nb.nas_graph_specs(n_fresh=2, dim=32, archive_guided=True, include_db_winners=False)
    assert captured["cfg"] is None
