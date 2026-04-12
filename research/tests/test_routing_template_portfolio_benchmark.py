from research.tools.benchmark_routing_template_portfolio import (
    _make_locked_candidates,
    _rank_rows,
)


def test_make_locked_candidates_includes_expected_portfolio():
    preselection = {
        "intelligent": {
            "selected": {
                "builder_kwargs": {
                    "easy_op": "conv_only",
                    "medium_op": "adaptive_lane_mixer",
                    "hard_op": "moe_topk",
                }
            }
        },
        "recursive": {
            "selected": {"builder_kwargs": {"max_depth": 3, "post_op": "conv_only"}}
        },
    }
    candidates = _make_locked_candidates(preselection)
    names = [candidate.name for candidate in candidates]
    assert names == [
        "multiscale_locked_prod",
        "multiscale_locked_hq",
        "hybrid_sparse_triplet_locked",
        "multiscale_difficulty_locked",
        "intelligent_multilane_locked",
        "recursive_depth_locked",
    ]


def test_rank_rows_sorts_by_requested_key():
    rows = [
        {"candidate": "a", "long_eval_loss": 11.5, "quality_per_cost": 100.0},
        {"candidate": "b", "long_eval_loss": 10.5, "quality_per_cost": 90.0},
    ]
    assert _rank_rows(rows, key="long_eval_loss")[0]["candidate"] == "b"
    assert _rank_rows(rows, key="quality_per_cost", reverse=True)[0]["candidate"] == "a"
