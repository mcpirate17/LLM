from __future__ import annotations

import research.scientist.llm.context as context_facade
from research.scientist.llm.context_briefing import build_briefing_context
from research.scientist.llm.context_experiment import (
    build_experiment_context,
    build_rich_context,
)
from research.scientist.llm.context_hypothesis import build_hypothesis_context


def test_context_facade_reexports_primary_builders():
    assert context_facade.build_briefing_context is build_briefing_context
    assert context_facade.build_experiment_context is build_experiment_context
    assert context_facade.build_rich_context is build_rich_context
    assert context_facade.build_hypothesis_context is build_hypothesis_context


def test_context_facade_preserves_legacy_compatibility_names():
    exported = set(context_facade.__all__)

    assert "Dict" in exported
    assert "List" in exported
    assert "Optional" in exported
    assert "logging" in exported
    assert "re" in exported
    assert "grouped_primitive_registry" in exported
    assert "primitive_registry_size" in exported
    assert context_facade.grouped_primitive_registry()
    assert context_facade.primitive_registry_size() > 0
