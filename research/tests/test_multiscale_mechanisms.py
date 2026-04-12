from research.tools.multiscale_catalogue import build_multiscale_registry
from research.tools.confirm_multiscale_rich_lane_router_winner import _dedupe_candidates
from research.tools.multiscale_mechanisms import (
    build_mechanism_coverage,
    classify_hard_mechanism,
    classify_medium_mechanism,
)


def test_medium_mechanism_coverage_counts():
    registry = build_multiscale_registry()
    medium_rows, _ = _dedupe_candidates(registry["medium_candidates"])
    coverage = build_mechanism_coverage(medium_rows, "medium")
    assert coverage["family_count"] == 7
    assert coverage["largest_family_share"] < 0.4


def test_hard_mechanism_coverage_counts():
    registry = build_multiscale_registry()
    hard_rows, _ = _dedupe_candidates(registry["hard_candidates"])
    coverage = build_mechanism_coverage(hard_rows, "hard")
    assert coverage["family_count"] == 5
    assert coverage["largest_family_share"] < 0.4


def test_default_families_resolve_as_expected():
    assert (
        classify_medium_mechanism("conv_only")["family"] == "local_convolutional_mixing"
    )
    assert (
        classify_hard_mechanism("mixed_recursion_gate")["family"]
        == "recursion_adaptive_depth"
    )
