from __future__ import annotations

from research.scientist.runner.execution_screening import (
    _make_experiment_results as canonical_make_experiment_results,
)
from research.scientist.runner.execution_screening_pipeline import (
    _make_experiment_results as pipeline_make_experiment_results,
)


def test_split_pipeline_results_initializer_matches_canonical_initializer():
    canonical = canonical_make_experiment_results()
    pipeline = pipeline_make_experiment_results()

    assert pipeline.keys() == canonical.keys()
    assert pipeline["funnel_counts"].keys() == canonical["funnel_counts"].keys()
    assert pipeline["funnel_counts"]["rapid_screen_attempted"] == 0
    assert pipeline["funnel_counts"]["dropped_rapid_screening"] == 0
