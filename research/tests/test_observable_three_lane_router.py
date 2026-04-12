from research.tools.run_observable_three_lane_router import _build_markdown


def test_observable_three_lane_markdown_mentions_expected_lanes():
    payload = {
        "checkpoints": [
            {
                "step": 0,
                "train_loss": float("nan"),
                "eval_loss": 12.0,
                "observability": {
                    "aggregate_routing": {
                        "lane_entropy": 0.5,
                        "route_strength_mean": 1.0,
                        "sparse_span_coverage": 0.0,
                    },
                    "difficulty": {"entropy": 0.8, "mean_max_prob": 0.6},
                    "merges": [
                        {
                            "merge_index": 0,
                            "branch_weight_mean": [0.7, 0.3],
                            "branch_gain_values": [1.0, 1.1],
                            "branch_bias_values": [0.0, 0.0],
                            "branch_dominance_mean": 0.7,
                        }
                    ],
                },
            }
        ]
    }
    markdown = _build_markdown(payload)
    assert "cheap_verify_blend" in markdown
    assert "block_sparse_linear" in markdown
    assert "moe_topk" in markdown
