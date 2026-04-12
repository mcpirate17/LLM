from research.scientist.runner import execution_screening


def test_execution_screening_imports_graph_screening_helpers():
    assert callable(execution_screening.analyze_graph_for_screening)
    assert callable(execution_screening.structural_gate_failure)
    assert callable(execution_screening.toxic_failure_ratio)
