from argparse import Namespace

from research.tools import ollama_reference_probe_audit as audit


def test_build_items_covers_all_reference_tasks():
    items = audit.build_items([0, 1], samples_per_task=2)

    assert len(items) == 2 * 2 * len(audit.BUILDERS)
    assert {item.task for item in items} == {
        "associative_recall",
        "induction_copy",
        "entity_binding",
        "binding_multislot",
        "blimp_choice",
    }
    assert all(item.prompt for item in items)
    assert all(item.expected for item in items)


def test_single_response_scores_first_candidate_only():
    item = audit.ProbeItem(
        task="x",
        seed=0,
        item_index=0,
        prompt="",
        expected=("amber",),
        candidates=("amber", "cobalt"),
    )

    assert audit.score_response(item, "The answer is amber.")[0] == 1.0
    assert audit.score_response(item, "cobalt, not amber")[0] == 0.0


def test_multi_blank_slot_scoring_counts_partial_completion():
    item = audit.ProbeItem(
        task="binding_multislot",
        seed=0,
        item_index=0,
        prompt="",
        expected=("amber", "mug", "teal", "key"),
        candidates=("amber", "mug", "teal", "key"),
        score_mode="slots",
    )

    correct, slot_acc = audit.score_response(item, '{"1":"amber","2":"mug"}')

    assert correct == 0.0
    assert slot_acc == 0.5


def test_resolve_models_prefers_explicit_models():
    args = Namespace(
        models=["qwen3.5:0.8b", "gemma2:2b", "qwen3.5:0.8b"],
        model=None,
        all_local_models=False,
        ollama_url="http://127.0.0.1:11434",
        timeout_s=1.0,
        max_model_size_gb=None,
    )

    assert audit.resolve_models(args) == ["qwen3.5:0.8b", "gemma2:2b"]


def test_summarize_results_by_model_keeps_models_separate():
    results = [
        audit.ProbeResult(
            "m1", "binding", 0, 0, "x", "x", 1.0, 1.0, 10.0, 1, 1, 10.0, "stop", ""
        ),
        audit.ProbeResult(
            "m2", "binding", 0, 0, "x", "y", 0.0, 0.0, 20.0, 1, 1, 20.0, "stop", ""
        ),
    ]

    rows = audit.summarize_results_by_model(results)

    assert rows == [
        {
            "model": "m1",
            "task": "binding",
            "n": 1,
            "mean_correct": 1.0,
            "mean_slot_acc": 1.0,
            "median_latency_ms": 10.0,
        },
        {
            "model": "m2",
            "task": "binding",
            "n": 1,
            "mean_correct": 0.0,
            "mean_slot_acc": 0.0,
            "median_latency_ms": 20.0,
        },
    ]
