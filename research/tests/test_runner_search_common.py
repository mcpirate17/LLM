from types import SimpleNamespace

import pytest

from research.scientist.runner import search_common


pytestmark = pytest.mark.unit


class _Graph:
    def __init__(self, fingerprint: str):
        self._fingerprint = fingerprint

    def fingerprint(self) -> str:
        return self._fingerprint


def test_structural_population_novelty_penalizes_duplicate_fingerprints(monkeypatch):
    monkeypatch.setattr(
        search_common,
        "novelty_score",
        lambda _graph: SimpleNamespace(structural_novelty=0.8),
    )

    score = search_common.structural_population_novelty(
        _Graph("same"),
        [_Graph("same"), _Graph("same"), _Graph("same"), _Graph("other")],
    )

    assert score == pytest.approx(0.32)


def test_evolution_population_summary_matches_runner_contract():
    population = [
        SimpleNamespace(fitness=0.7, novelty=0.9, fingerprint="best"),
        SimpleNamespace(fitness=0.15, novelty=0.1, fingerprint="weak"),
        SimpleNamespace(fitness=0.3, novelty=0.4, fingerprint="survivor"),
    ]

    summary = search_common.summarize_evolution_population(
        population,
        {"total": 5, "s0": 3, "s1": 2},
    )

    assert summary["total"] == 5
    assert summary["stage0_passed"] == 3
    assert summary["stage05_passed"] == 3
    assert summary["stage1_passed"] == 2
    assert summary["novel_count"] == 1
    assert summary["best_loss_ratio"] == pytest.approx(0.3)
    assert summary["best_novelty_score"] == pytest.approx(0.9)
    assert summary["survivors"] == [
        {"fingerprint": "best", "novelty": 0.9, "loss_ratio": pytest.approx(0.3)},
        {
            "fingerprint": "survivor",
            "novelty": 0.4,
            "loss_ratio": pytest.approx(0.7),
        },
    ]


def test_novelty_summary_can_score_best_metrics_from_all_individuals():
    ns_result = SimpleNamespace(
        archive_size=11,
        best_individuals=[
            SimpleNamespace(fitness=0.1, novelty=0.95, fingerprint="novel-but-weak"),
            SimpleNamespace(fitness=0.6, novelty=0.4, fingerprint="fit"),
        ],
    )

    summary = search_common.summarize_novelty_result(
        ns_result,
        {"total": 2, "s0": 2, "s1": 1},
        best_from_all_individuals=True,
    )

    assert summary["archive_size"] == 11
    assert summary["novel_count"] == 1
    assert summary["best_loss_ratio"] == pytest.approx(0.4)
    assert summary["best_novelty_score"] == pytest.approx(0.95)
    assert summary["survivors"] == [
        {"fingerprint": "fit", "novelty": 0.4, "loss_ratio": pytest.approx(0.4)}
    ]


def test_novelty_summary_can_score_best_metrics_from_survivors_only():
    ns_result = SimpleNamespace(
        archive_size=7,
        best_individuals=[
            SimpleNamespace(fitness=0.1, novelty=0.95, fingerprint="novel-but-weak"),
            SimpleNamespace(fitness=0.4, novelty=0.3, fingerprint="survivor-a"),
            SimpleNamespace(fitness=0.7, novelty=0.8, fingerprint="survivor-b"),
        ],
    )

    summary = search_common.summarize_novelty_result(
        ns_result,
        {"total": 3, "s0": 3, "s1": 2},
        best_from_all_individuals=False,
    )

    assert summary["best_loss_ratio"] == pytest.approx(0.3)
    assert summary["best_novelty_score"] == pytest.approx(0.8)
    assert summary["survivors"] == [
        {
            "fingerprint": "survivor-a",
            "novelty": 0.3,
            "loss_ratio": pytest.approx(0.6),
        },
        {
            "fingerprint": "survivor-b",
            "novelty": 0.8,
            "loss_ratio": pytest.approx(0.3),
        },
    ]


def test_program_evaluation_callback_forwards_cached_behavioral_fingerprint():
    calls = []
    runner = SimpleNamespace(
        _on_program_evaluated=lambda *args, **kwargs: calls.append((args, kwargs))
    )
    counters = {"total": 0, "s0": 0, "s1": 0}
    cached_fingerprint = SimpleNamespace(quality="ok")
    callback = search_common.make_program_evaluation_callback(
        runner,
        eval_counters=counters,
        nb=SimpleNamespace(),
        exp_id="exp",
        model_source="novelty",
        fingerprint_cache={"abc": cached_fingerprint},
        debug=True,
    )

    callback(_Graph("abc"), 0.5, "sandbox", {"passed": True})

    args, kwargs = calls[0]
    assert args[0].fingerprint() == "abc"
    assert args[1] == 0.5
    assert args[4] is counters
    assert args[6] == "exp"
    assert kwargs == {
        "model_source": "novelty",
        "behavioral_fingerprint": cached_fingerprint,
        "debug": True,
    }


def test_search_result_analysis_updates_notebook_and_returns_aria_outputs():
    calls = []
    nb = SimpleNamespace(
        update_op_success_rates=lambda exp_id: calls.append(("success", exp_id)),
        update_failure_signatures=lambda exp_id: calls.append(("failure", exp_id)),
    )
    aria = SimpleNamespace(
        experiment_summary=lambda results, context: ("summary", results, context),
        analyze_results=lambda results, context: ("analysis", results, context),
    )
    runner = SimpleNamespace(
        aria=aria,
        _build_rich_context_for_experiment=lambda results, config, hypothesis, nb: {
            "hypothesis": hypothesis,
            "total": results["total"],
        },
        _analyze_results=lambda results, exp_id, nb, context: {
            "exp_id": exp_id,
            "context": context,
        },
    )

    context, summary, llm_analysis, insights = search_common.analyze_search_results(
        runner,
        exp_id="exp-1",
        results={"total": 2},
        config=SimpleNamespace(),
        hypothesis="try a sparse route",
        nb=nb,
    )

    assert calls == [("success", "exp-1"), ("failure", "exp-1")]
    assert context == {"hypothesis": "try a sparse route", "total": 2}
    assert summary == ("summary", {"total": 2}, context)
    assert llm_analysis == ("analysis", {"total": 2}, context)
    assert insights == {"exp_id": "exp-1", "context": context}


def test_publish_search_completion_emits_terminal_event_and_compat_record(
    monkeypatch,
):
    monkeypatch.setattr(search_common.time, "time", lambda: 123.5)
    terminal_events = []
    compat_records = []
    runner = SimpleNamespace(
        aria=SimpleNamespace(state=SimpleNamespace(mood="focused")),
        _publish_terminal_event=lambda **kwargs: terminal_events.append(kwargs),
        _complete_experiment_compat=lambda **kwargs: compat_records.append(kwargs),
    )
    nb = SimpleNamespace()

    search_common.publish_search_completion(
        runner,
        nb=nb,
        exp_id="exp-1",
        results={"total": 1},
        summary="summary",
        insights={"a": 1},
        llm_analysis={"b": 2},
        producer="runner.test",
        mode="novelty",
    )

    assert terminal_events == [
        {
            "producer": "runner.test",
            "event_type": "experiment_completed",
            "exp_id": "exp-1",
            "payload": {
                "completed_at": 123.5,
                "results": {"total": 1},
                "aria_summary": "summary",
                "aria_mood": "focused",
                "insights": {"a": 1},
                "llm_analysis": {"b": 2},
                "mode": "novelty",
            },
        }
    ]
    assert compat_records == [
        {
            "nb": nb,
            "experiment_id": "exp-1",
            "results": {"total": 1},
            "aria_summary": "summary",
            "insights": {"a": 1},
            "llm_analysis": {"b": 2},
        }
    ]


def test_cached_fingerprint_fn_uses_graph_fingerprint_key():
    cached = SimpleNamespace(quality="ok")
    lookup = search_common.make_cached_fingerprint_fn({"abc": cached})

    assert lookup(_Graph("abc")) is cached
    assert lookup(_Graph("missing")) is None
