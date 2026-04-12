from __future__ import annotations

from research.synthesis.graph import ComputationGraph
from research.tools.audit_failure_root_causes import _build_summary, FailureRow


def _graph_with_dim_mismatch() -> ComputationGraph:
    g = ComputationGraph(64)
    inp = g.add_input()
    g.set_output(inp)
    return g


def test_build_summary_normalizes_s1_labels_and_tracks_preflight_prevention():
    rows = [
        FailureRow(
            raw_error_type="s1_RuntimeError",
            error_type="RuntimeError",
            error_message="The size of tensor a (32) must match the size of tensor b (64)",
            graph=_graph_with_dim_mismatch(),
        )
    ]

    summary = _build_summary(rows)

    assert summary["before_counts"]["RuntimeError"] == 1
    assert summary["prevented_by_current_validation"]["RuntimeError"] == 1
    assert summary["after_counts_preflight_replay"]["RuntimeError"] == 0
    assert (
        summary["ranked_root_causes"][0]["root_cause"]
        == "residual_dominant_no_learning"
    )
