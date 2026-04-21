from __future__ import annotations

from research.eval import hellaswag_eval


def test_native_subset_payload_is_cached(monkeypatch):
    monkeypatch.setattr(
        hellaswag_eval,
        "_download_hellaswag",
        lambda: [
            {
                "ctx": "ctx 1",
                "endings": ["a", "b", "c", "d"],
                "label": 1,
            },
            {
                "ctx": "ctx 2",
                "endings": ["e", "f", "g", "h"],
                "label": 2,
            },
        ],
    )
    hellaswag_eval._tokenized_examples_cache.clear()
    hellaswag_eval._tokenized_subset_cache.clear()
    hellaswag_eval._native_subset_cache.clear()

    payload_a = hellaswag_eval._get_native_subset_payload(2, vocab_size=257)
    payload_b = hellaswag_eval._get_native_subset_payload(2, vocab_size=257)

    assert payload_a is payload_b
    ctx_tokens, ending_tokens, labels = payload_a
    assert len(ctx_tokens) == 2
    assert len(ending_tokens) == 2
    assert labels == [1, 2]
