from research.tools.comparative_anatomy_routing_templates import (
    _build_candidates,
    _redesign_hypotheses,
    _structural_summary,
)


def test_structural_summary_orders_complexity_as_expected():
    candidates = {candidate.name: candidate for candidate in _build_candidates()}
    multiscale = _structural_summary(candidates["multiscale_locked_prod"])
    intelligent = _structural_summary(candidates["intelligent_multilane_locked"])
    recursive = _structural_summary(candidates["recursive_depth_locked"])
    assert multiscale["decision_points"] > recursive["decision_points"]
    assert multiscale["merge_complexity"] > recursive["merge_complexity"]
    assert intelligent["branch_count"] >= recursive["branch_count"]


def test_redesign_hypotheses_are_nonempty_and_target_structure():
    rows = _redesign_hypotheses()
    assert len(rows) >= 3
    assert all(row["targets_problem"] for row in rows)
