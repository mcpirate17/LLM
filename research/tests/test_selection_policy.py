import pytest
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

from research.scientist.notebook import LabNotebook
from research.scientist.runner import ExperimentRunner, RunConfig

pytestmark = pytest.mark.unit


def _graph_json(op_name: str = "linear_proj") -> str:
    return json.dumps(
        {
            "nodes": {
                "0": {"id": 0, "op_name": "input", "input_ids": []},
                "1": {"id": 1, "op_name": op_name, "input_ids": [0]},
            }
        }
    )


def _seed_candidates(nb: LabNotebook, exp_id: str):
    rows = [
        dict(
            graph_fingerprint="a1",
            graph_json=_graph_json("attention"),
            stage0_passed=1,
            stage05_passed=1,
            stage1_passed=1,
            loss_ratio=0.52,
            baseline_loss_ratio=0.86,
            novelty_score=0.33,
            throughput_tok_s=220.0,
            flops_per_token=1.3,
            peak_memory_mb=410.0,
            stability_score=0.80,
            has_nan_grad=0,
            has_zero_grad=0,
        ),
        dict(
            graph_fingerprint="b1",
            graph_json=_graph_json("conv1d_seq"),
            stage0_passed=1,
            stage05_passed=1,
            stage1_passed=1,
            loss_ratio=0.58,
            baseline_loss_ratio=0.90,
            novelty_score=0.77,
            throughput_tok_s=165.0,
            flops_per_token=1.1,
            peak_memory_mb=300.0,
            stability_score=0.92,
            has_nan_grad=0,
            has_zero_grad=0,
        ),
        dict(
            graph_fingerprint="c1",
            graph_json=_graph_json("gelu"),
            stage0_passed=1,
            stage05_passed=1,
            stage1_passed=0,
            loss_ratio=0.70,
            baseline_loss_ratio=0.96,
            novelty_score=0.60,
            throughput_tok_s=250.0,
            flops_per_token=0.9,
            peak_memory_mb=280.0,
            stability_score=0.60,
            has_nan_grad=0,
            has_zero_grad=1,
        ),
    ]
    out = []
    for item in rows:
        rid = nb.record_program_result(exp_id, **item)
        item = dict(item)
        item["result_id"] = rid
        out.append(item)
    return out


def test_candidate_scoring_is_deterministic_for_fixed_db():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "selection_det.db")
    runner = ExperimentRunner(db_path)
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment(
        "synthesis", {"n_programs": 3}, "selection determinism"
    )
    candidates = _seed_candidates(nb, exp_id)

    cfg = RunConfig(selection_policy="ucb", selection_epsilon=0.0)
    first = runner._score_candidate_pool(candidates, cfg, nb, "unit_test", exp_id)
    second = runner._score_candidate_pool(candidates, cfg, nb, "unit_test", exp_id)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["selected"][0]["result_id"] == second["selected"][0]["result_id"]
    nb.close()


def test_safety_valve_triggers_novelty_mode_when_plateaued():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "selection_plateau.db")
    runner = ExperimentRunner(db_path)
    nb = LabNotebook(db_path)

    for i in range(8):
        exp_id = nb.start_experiment("synthesis", {"n_programs": 2}, f"plateau-{i}")
        nb.complete_experiment(
            exp_id,
            {
                "total": 2,
                "stage0_passed": 2,
                "stage05_passed": 2,
                "stage1_passed": 0,
                "best_loss_ratio": 0.82,
                "best_novelty_score": 0.4,
            },
        )

    cfg = RunConfig(safety_plateau_window=8, safety_plateau_min_delta=0.01)
    trigger = runner._selection_safety_valve(nb, cfg)
    assert trigger is not None
    assert trigger["triggered"] is True
    assert trigger["mode"] == "novelty"
    nb.close()


def test_mode_selection_logs_decision_and_applies_safety_valve():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "selection_mode.db")
    runner = ExperimentRunner(db_path)
    nb = LabNotebook(db_path)

    for i in range(8):
        exp_id = nb.start_experiment("synthesis", {"n_programs": 2}, f"mode-{i}")
        nb.complete_experiment(
            exp_id,
            {
                "total": 2,
                "stage0_passed": 2,
                "stage05_passed": 2,
                "stage1_passed": 0,
                "best_loss_ratio": 0.90,
                "best_novelty_score": 0.2,
            },
        )

    runner.aria.recommend_next_mode = MagicMock(
        return_value={
            "mode": "synthesis",
            "reasoning": "default",
            "confidence": 0.4,
            "config": {},
        }
    )
    runner._invoke_code_healer = MagicMock()

    cfg = RunConfig(safety_plateau_window=8, safety_plateau_min_delta=0.01)
    rec = runner._select_next_mode(cfg, nb, n_experiments=9)
    assert rec["mode"] == "novelty"
    logs = nb.get_selection_decisions(context="mode_selection", limit=5)
    assert logs
    assert logs[0]["context"] == "mode_selection"
    assert logs[0]["trigger_json"]["triggered"] is True
    nb.close()


def test_selection_insight_interactions_learn_supported_vs_not_supported():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "selection_insight_interactions.db")
    nb = LabNotebook(db_path)

    exp_id = nb.start_experiment(
        "synthesis", {"n_programs": 2}, "insight interaction learning"
    )
    i1 = nb.record_insight(
        "success_factor", "Top op: attention", exp_id, confidence=0.9
    )
    i2 = nb.record_insight(
        "pattern", "Pair: attention + conv1d_seq", exp_id, confidence=0.8
    )

    d1 = nb.record_selection_decision(
        context="auto_investigate_screening",
        candidate_pool_summary={"candidate_count": 2},
        score_breakdown=[],
        policy={"name": "ucb"},
        reason="trial-1",
        chosen_experiments=[{"result_id": "r1"}, {"result_id": "r2"}],
        experiment_id=exp_id,
        trigger=None,
    )
    t1 = nb.record_selection_insight_trial(
        decision_id=d1,
        context="auto_investigate_screening",
        insight_ids=[i1, i2],
        chosen_result_ids=["r1", "r2"],
        source_experiment_id=exp_id,
    )
    nb.resolve_selection_insight_trial(t1, reward=0.8, outcome="supported")

    d2 = nb.record_selection_decision(
        context="auto_investigate_screening",
        candidate_pool_summary={"candidate_count": 2},
        score_breakdown=[],
        policy={"name": "ucb"},
        reason="trial-2",
        chosen_experiments=[{"result_id": "r3"}, {"result_id": "r4"}],
        experiment_id=exp_id,
        trigger=None,
    )
    t2 = nb.record_selection_insight_trial(
        decision_id=d2,
        context="auto_investigate_screening",
        insight_ids=[i1, i2],
        chosen_result_ids=["r3", "r4"],
        source_experiment_id=exp_id,
    )
    nb.resolve_selection_insight_trial(t2, reward=0.2, outcome="not_supported")

    rows = nb.get_selection_insight_interactions(limit=20)
    pair = next((r for r in rows if r["insight_a"] != r["insight_b"]), None)
    assert pair is not None
    assert pair["n_trials"] == 2
    assert pair["n_supported"] == 1
    assert pair["n_not_supported"] == 1
    assert abs(float(pair["mean_reward"]) - 0.5) < 1e-8
    nb.close()


def test_candidate_scoring_includes_supporting_insight_ids():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "selection_insight_scoring.db")
    runner = ExperimentRunner(db_path)
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment(
        "synthesis", {"n_programs": 3}, "selection insight scoring"
    )
    candidates = _seed_candidates(nb, exp_id)

    nb.record_insight(
        "success_factor",
        "attention composes well with residual paths",
        exp_id,
        confidence=0.9,
        subject_key="attention",
        insight_level="op",
    )
    nb.record_insight(
        "success_factor",
        "conv1d_seq is robust in stage1",
        exp_id,
        confidence=0.8,
        subject_key="conv1d_seq",
        insight_level="op",
    )

    cfg = RunConfig(selection_policy="ucb", selection_epsilon=0.0)
    scored = runner._score_candidate_pool(
        candidates, cfg, nb, "unit_test_insights", exp_id
    )

    assert "supporting_insight_ids" in scored
    assert isinstance(scored["supporting_insight_ids"], list)
    assert scored["supporting_insight_ids"]
    assert "supporting_insight_ids" in scored["scored"][0]
    assert "insight_interaction" in scored["scored"][0]["components"]
    nb.close()


def test_mode_selection_promotes_refinement_when_diverse_winners_exist():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "selection_refinement_mode.db")
    runner = ExperimentRunner(db_path)
    nb = LabNotebook(db_path)

    exp_id = nb.start_experiment("synthesis", {"n_programs": 6}, "seed refinement pool")
    for idx in range(6):
        nb.record_program_result(
            exp_id,
            graph_fingerprint=f"seed_ref_{idx}",
            graph_json=_graph_json("attention" if idx % 2 == 0 else "conv1d_seq"),
            stage0_passed=1,
            stage05_passed=1,
            stage1_passed=1,
            loss_ratio=0.45 + 0.02 * idx,
            novelty_score=0.25 + 0.08 * idx,
            graph_n_ops=4 + idx,
        )
    nb.complete_experiment(
        exp_id,
        {
            "total": 6,
            "stage0_passed": 6,
            "stage05_passed": 6,
            "stage1_passed": 6,
            "best_loss_ratio": 0.45,
            "best_novelty_score": 0.65,
        },
    )

    runner.aria.recommend_next_mode = MagicMock(
        return_value={
            "mode": "synthesis",
            "reasoning": "base recommendation",
            "confidence": 0.4,
            "config": {},
        }
    )
    runner._invoke_code_healer = MagicMock()

    cfg = RunConfig(
        refinement_top_k=3,
        refinement_min_stage1_survivors=2,
        refinement_min_distance=0.05,
        refinement_novelty_pressure=0.35,
    )
    rec = runner._select_next_mode(cfg, nb, n_experiments=2)
    assert rec["mode"] == "refinement"
    assert rec["config"]["model_source"] == "fingerprint_refine"
    assert rec["config"]["refine_source_result_ids"]
    nb.close()


def test_refinement_source_selection_is_deterministic_and_diverse():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "selection_refinement_deterministic.db")
    runner = ExperimentRunner(db_path)
    nb = LabNotebook(db_path)

    exp_id = nb.start_experiment(
        "synthesis", {"n_programs": 8}, "deterministic refinement seeds"
    )
    for idx in range(8):
        nb.record_program_result(
            exp_id,
            graph_fingerprint=f"det_seed_{idx}",
            graph_json=_graph_json("attention" if idx % 2 == 0 else "gelu"),
            stage0_passed=1,
            stage05_passed=1,
            stage1_passed=1,
            loss_ratio=0.40 + 0.03 * idx,
            novelty_score=0.15 + 0.09 * idx,
            graph_n_ops=3 + idx,
        )
    nb.complete_experiment(
        exp_id,
        {
            "total": 8,
            "stage0_passed": 8,
            "stage05_passed": 8,
            "stage1_passed": 8,
            "best_loss_ratio": 0.40,
            "best_novelty_score": 0.78,
        },
    )

    cfg = RunConfig(
        refinement_top_k=4,
        refinement_min_stage1_survivors=2,
        refinement_min_distance=0.12,
        refinement_novelty_pressure=0.4,
    )
    plan1 = runner._build_refinement_plan(nb, cfg)
    plan2 = runner._build_refinement_plan(nb, cfg)
    assert plan1 is not None and plan2 is not None
    assert plan1["source_result_ids"] == plan2["source_result_ids"]

    picked = [nb.get_program_detail(rid) for rid in plan1["source_result_ids"]]
    picked = [row for row in picked if row]
    assert len(picked) >= 2
    for i in range(len(picked)):
        for j in range(i + 1, len(picked)):
            dist = runner._refinement_candidate_distance(picked[i], picked[j])
            assert dist >= cfg.refinement_min_distance - 1e-9
    nb.close()


def test_auto_recommend_records_next_experiment_plan_decision():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "selection_next_plan.db")
    runner = ExperimentRunner(db_path)
    nb = LabNotebook(db_path)

    exp_id = nb.start_experiment(
        "synthesis", {"n_programs": 3}, "plan recommendation seed"
    )
    for idx in range(3):
        nb.record_program_result(
            exp_id,
            graph_fingerprint=f"plan_seed_{idx}",
            graph_json=_graph_json("attention"),
            stage0_passed=1,
            stage05_passed=1,
            stage1_passed=1,
            loss_ratio=0.52 + 0.02 * idx,
            novelty_score=0.40 + 0.05 * idx,
            graph_n_ops=5 + idx,
        )
    nb.complete_experiment(
        exp_id,
        {
            "total": 3,
            "stage0_passed": 3,
            "stage05_passed": 3,
            "stage1_passed": 3,
            "best_loss_ratio": 0.52,
            "best_novelty_score": 0.50,
        },
    )

    class _FakePlanner:
        def propose_plan(self, *_args, **_kwargs):
            return {
                "mode": "refinement",
                "reasoning": "Top survivors warrant local recursive tweaks.",
                "confidence": 0.78,
                "config": {
                    "model_source": "fingerprint_refine",
                    "refine_source_result_ids": "x1,x2",
                    "n_programs": 16,
                },
                "guardrails": {"diversity": "greedy max-distance selection"},
                "planner": {"source": "local", "backend": "ollama"},
            }

    with patch(
        "research.scientist.runner.results.NextExperimentDecisionPlanner.from_run_config",
        return_value=_FakePlanner(),
    ):
        runner._auto_recommend(
            {
                "experiment_id": exp_id,
                "total": 3,
                "stage0_passed": 3,
                "stage05_passed": 3,
                "stage1_passed": 3,
            },
            RunConfig(),
            "test hypothesis",
            nb,
        )

    decisions = nb.get_decisions(decision_type="next_experiment_plan")
    assert decisions, "Expected a next_experiment_plan decision to be recorded"
    assert decisions[0]["decision_type"] == "next_experiment_plan"
    evidence = decisions[0].get("evidence_pack") or {}
    assert evidence.get("mode") == "refinement"
    assert (evidence.get("planner") or {}).get("source") == "local"
    nb.close()


def test_recursive_refinement_runs_generation_and_records_decision():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "selection_recursive_refine.db")
    runner = ExperimentRunner(db_path)
    nb = LabNotebook(db_path)

    seed_exp = nb.start_experiment("synthesis", {"n_programs": 4}, "seed winners")
    seed_ids = []
    for idx in range(4):
        rid = nb.record_program_result(
            seed_exp,
            graph_fingerprint=f"seed_{idx}",
            graph_json=_graph_json("attention"),
            stage0_passed=1,
            stage05_passed=1,
            stage1_passed=1,
            loss_ratio=0.55 - 0.02 * idx,
            novelty_score=0.30 + 0.07 * idx,
            graph_n_ops=5 + idx,
        )
        seed_ids.append(rid)
    nb.complete_experiment(
        seed_exp,
        {
            "total": 4,
            "stage0_passed": 4,
            "stage05_passed": 4,
            "stage1_passed": 4,
            "best_loss_ratio": 0.49,
            "best_novelty_score": 0.51,
        },
    )

    calls = []

    def _fake_synthesis(cfg, fake_nb, n_exp, _limit, _reason):
        calls.append(
            {
                "model_source": cfg.model_source,
                "sources": cfg.refine_source_result_ids,
            }
        )
        exp = fake_nb.start_experiment("synthesis", cfg.to_dict(), f"gen-{len(calls)}")
        for i, source in enumerate(str(cfg.refine_source_result_ids or "").split(",")):
            fake_nb.record_program_result(
                exp,
                graph_fingerprint=f"child_{len(calls)}_{i}_{source[:4]}",
                graph_json=_graph_json("conv1d_seq"),
                stage0_passed=1,
                stage05_passed=1,
                stage1_passed=1,
                loss_ratio=0.48 - 0.01 * len(calls) - 0.001 * i,
                novelty_score=0.32 + 0.03 * i,
                graph_n_ops=6 + i,
            )
        fake_nb.complete_experiment(
            exp,
            {
                "total": max(
                    1, len(str(cfg.refine_source_result_ids or "").split(","))
                ),
                "stage0_passed": 1,
                "stage05_passed": 1,
                "stage1_passed": 1,
                "best_loss_ratio": 0.45,
                "best_novelty_score": 0.40,
            },
        )

    cfg = RunConfig(
        n_programs=12,
        refinement_top_k=2,
        refinement_generations=2,
        refinement_budget_programs=24,
        refinement_min_stage1_survivors=2,
    )

    with patch.object(runner, "_run_continuous_synthesis", side_effect=_fake_synthesis):
        runner._run_continuous_refinement(
            cfg, nb, n_experiments=2, limit_str="test", mode_reasoning="loop"
        )

    assert calls
    assert all(call["model_source"] == "fingerprint_refine" for call in calls)
    assert all(call["sources"] for call in calls)
    decisions = nb.get_decisions(decision_type="recursive_refinement")
    assert decisions
    nb.close()
