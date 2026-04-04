from research.tools.exact_graph_replay import _expand_replays


def test_expand_replays_repeats_each_source_in_order():
    rows = [
        {"result_id": "a"},
        {"result_id": "b"},
    ]

    expanded = _expand_replays(rows, 2)

    assert [row["result_id"] for row in expanded] == ["a", "a", "b", "b"]
    assert [row["replay_index"] for row in expanded] == [0, 1, 0, 1]
