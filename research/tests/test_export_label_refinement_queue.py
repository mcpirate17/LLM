from research.tools.export_label_refinement_queue import (
    _build_ambiguous_batches,
    _dedupe_near_misses,
)


def test_dedupe_near_misses_keeps_best_signature_candidate():
    items = [
        {
            "result_id": "r1",
            "signature": "sig-a",
            "ensemble_score": 0.80,
            "gbm_score": 0.70,
            "graph_score": 0.60,
            "dup_group_s1_rate": 0.5,
            "loss_ratio": 0.30,
        },
        {
            "result_id": "r2",
            "signature": "sig-a",
            "ensemble_score": 0.85,
            "gbm_score": 0.65,
            "graph_score": 0.55,
            "dup_group_s1_rate": 0.5,
            "loss_ratio": 0.35,
        },
        {
            "result_id": "r3",
            "signature": "sig-b",
            "ensemble_score": 0.60,
            "gbm_score": 0.90,
            "graph_score": 0.40,
            "dup_group_s1_rate": 0.0,
            "loss_ratio": 0.25,
        },
    ]

    deduped = _dedupe_near_misses(items)

    assert [item["result_id"] for item in deduped] == ["r2", "r3"]


def test_build_ambiguous_batches_prefers_stage1_sources():
    groups = [
        {
            "signature": "sig-a",
            "n_rows": 3,
            "s1_rate": 0.33,
            "result_ids": ["loser", "winner", "other"],
            "ensemble_score": 0.9,
            "gbm_score": 0.8,
            "graph_score": 0.7,
            "templates_used": ["t1"],
            "motifs_used": [],
            "ops": ["op1"],
        }
    ]
    row_by_id = {
        "loser": {
            "result_id": "loser",
            "graph_fingerprint": "fp-loser",
            "stage05_passed": 1,
            "stage1_passed": 0,
            "loss_ratio": 0.22,
            "timestamp": 10,
        },
        "winner": {
            "result_id": "winner",
            "graph_fingerprint": "fp-winner",
            "stage05_passed": 1,
            "stage1_passed": 1,
            "loss_ratio": 0.18,
            "timestamp": 9,
        },
        "other": {
            "result_id": "other",
            "graph_fingerprint": "fp-other",
            "stage05_passed": 0,
            "stage1_passed": 0,
            "loss_ratio": None,
            "timestamp": 12,
        },
    }

    batches = _build_ambiguous_batches(
        groups,
        row_by_id,
        max_sources_per_group=2,
        batch_size=1,
        n_programs=24,
        refine_mutations_per_source=4,
        refine_pool_multiplier=3,
    )

    assert len(batches) == 1
    payload = batches[0]["launch_payload"]
    assert payload["result_ids"] == ["winner", "loser"]
    assert payload["mode"] == "refine_fingerprint"
    assert payload["refine_intent"] == "recommended"
