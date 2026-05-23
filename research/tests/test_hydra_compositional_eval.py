"""Smoke test for hydra_compositional_eval.

Verifies the end-to-end measurement path runs on a tiny CPU model with
synthetic held-out examples — no checkpoint, no HYDRA data file required.
"""

from __future__ import annotations

import torch

from research.eval.hydra_compositional_eval import (
    HydraExample,
    _agg,
    _extract_state_dict,
    _is_clean_example,
    _pick_distractors,
    evaluate,
)
from research.tests._probe_test_support import TinyLM


def _make_examples() -> list[HydraExample]:
    return [
        HydraExample(
            prompt="If a train travels 60 miles in 2 hours, what is its speed in mph?",
            expected_answer="30",
            category="math",
            type_="math",
            src_idx=i,
        )
        for i in range(6)
    ] + [
        HydraExample(
            prompt="A baker has 12 cupcakes and gives away 7. How many remain?",
            expected_answer="5",
            category="math",
            type_="math",
            src_idx=i + 100,
        )
        for i in range(6)
    ]


def test_is_clean_example_rejects_missing_or_mismatched_answer() -> None:
    assert _is_clean_example(
        {"expected_answer": "45", "teacher_completion": "the answer is 45 bars"}
    )
    # the smoking-gun case from distill_math.jsonl audit
    assert not _is_clean_example(
        {
            "expected_answer": "441111",
            "teacher_completion": "the boxed answer is 111111111111",
        }
    )
    assert not _is_clean_example(
        {"expected_answer": "", "teacher_completion": "anything"}
    )
    assert not _is_clean_example({"expected_answer": "45"})  # missing completion


def test_extract_state_dict_accepts_all_three_formats() -> None:
    """mixer_fingerprint uses {model_state_dict, step}; legacy used {model}; bare dict also works."""
    fake_params = {"embed.weight": torch.zeros(2, 2), "proj.bias": torch.zeros(2)}
    # mixer_fingerprint current format (the one the 165K ckpt actually uses)
    assert (
        _extract_state_dict({"model_state_dict": fake_params, "step": 165000})
        is fake_params
    )
    # legacy sketch format
    assert _extract_state_dict({"model": fake_params}) is fake_params
    # bare state_dict (no wrapper)
    assert _extract_state_dict(fake_params) is fake_params


def test_pick_distractors_prefers_same_kind() -> None:
    pool = ["30", "45", "62", "lemur", "platypus"]
    import random

    rng = random.Random(0)
    picks = _pick_distractors(pool, "30", n_distractors=2, rng=rng)
    assert len(picks) == 2
    # "30" is numeric; both picks should be numeric (same_kind has 2 numeric: "45","62")
    assert all(p.replace(".", "", 1).lstrip("-").isdigit() for p in picks)


def test_agg_handles_empty_and_populated() -> None:
    assert _agg([]) == {"n": 0}
    rows = [
        {"rank": 1, "rr": 1.0, "margin": 0.5},
        {"rank": 3, "rr": 1 / 3, "margin": -0.2},
        {"rank": 2, "rr": 0.5, "margin": 0.1},
    ]
    a = _agg(rows)
    assert a["n"] == 3
    assert a["top1"] == 1 / 3
    assert a["top3"] == 1.0
    assert abs(a["mrr"] - (1.0 + 1 / 3 + 0.5) / 3) < 1e-9


def test_evaluate_end_to_end_on_tinylm() -> None:
    torch.manual_seed(0)
    model = TinyLM(vocab_size=100277, dim=16)  # cl100k_base-sized vocab
    examples = _make_examples()

    result = evaluate(
        model,
        examples,
        device=torch.device("cpu"),
        n_distractors=2,
        max_context_tokens=64,
        seed=0,
    )

    summary = result["summary"]
    assert "overall" in summary
    overall = summary["overall"]
    assert overall["n"] == len(examples)
    assert 0.0 <= overall["top1"] <= 1.0
    assert 0.0 <= overall["top3"] <= 1.0
    # mrr and margin should be finite floats (untrained model — values themselves are uninformative)
    assert overall["mrr"] == overall["mrr"]  # not nan
    assert overall["mean_margin"] == overall["mean_margin"]

    # per-example rows have the contract the JSONL output expects
    per_ex = result["per_example"]
    assert len(per_ex) == len(examples)
    for row in per_ex:
        assert {
            "rank",
            "rr",
            "margin",
            "lp_correct",
            "lp_distractor_mean",
        } <= row.keys()
        assert row["rank"] >= 1
        assert 0.0 < row["rr"] <= 1.0

    # stratifications are present and bucket sums equal overall n
    for axis in ("by_category", "by_length_bucket", "by_kind"):
        assert axis in summary
        bucket_n = sum(b["n"] for b in summary[axis].values())
        assert bucket_n == overall["n"]
