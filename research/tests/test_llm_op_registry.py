from __future__ import annotations

from research.scientist.llm._op_registry import (
    grouped_primitive_registry,
    primitive_registry_size,
)
from research.scientist.llm.context_experiment import build_op_reference


def test_grouped_primitive_registry_is_populated():
    grouped = dict(grouped_primitive_registry())

    assert primitive_registry_size() > 0
    assert "elementwise_unary" in grouped
    assert "relu" in grouped["elementwise_unary"]
    assert any("matmul" in ops for ops in grouped.values())


def test_build_op_reference_includes_registry_entries():
    section = build_op_reference()

    assert "VALID OP NAMES" in section
    assert "elementwise_unary" in section
    assert "relu" in section
    assert "WARNING: Do NOT invent op names." in section
