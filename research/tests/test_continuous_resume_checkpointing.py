from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace

from research.scientist.runner._types import RunConfig
from research.scientist.runner.continuous_investigation import (
    _ContinuousInvestigationMixin,
)
from research.scientist.runner.continuous_validation import _ContinuousValidationMixin
from research.training.checkpointing import CheckpointManager


class _FakeTrainingProgram:
    def __init__(self, name: str):
        self.name = name

    def to_dict(self):
        return {"name": self.name}


class _FakeAria:
    def __init__(self):
        self.total_cost = 0.0
        self.total_tokens = 0
        self.state = SimpleNamespace(mood="steady")

    def experiment_summary(self, results, context=None):
        del results, context
        return "summary"

    def analyze_results(self, results, context=None):
        del results, context
        return {"analysis": "ok"}

    def formulate_investigation_hypothesis(self, context=None):
        del context
        return "hypothesis"


class _FakeNotebook:
    def __init__(self):
        self.recorded = []
        self.completed = []
        self.failed = []

    def record_program_result(self, **kwargs):
        self.recorded.append(kwargs)

    def complete_experiment(self, **kwargs):
        self.completed.append(kwargs)

    def fail_experiment(self, exp_id, message):
        self.failed.append((exp_id, message))

    def flush_writes(self):
        pass


class _ValidationRunner(_ContinuousValidationMixin):
    def __init__(self):
        self._stop_event = threading.Event()
        self._live_training_context = None
        self._lock = threading.Lock()
        self.aria = _FakeAria()
        self.visited_candidates = []
        # Production mixin reads self.notebook_path inside _publish_terminal_event;
        # the test fixture has no real notebook so a placeholder string is enough
        # to satisfy the attribute access during finalization paths.
        self.notebook_path = ":memory:"

    def _publish_terminal_event(self, **kwargs):
        del kwargs

    def _fail_experiment_compat(self, *args, **kwargs):
        del args, kwargs

    def _complete_experiment_compat(self, *args, **kwargs):
        del args, kwargs

    def _inline_validation_candidate_ids(self, config, leaderboard):
        del config, leaderboard
        return ["skip-me", "resume-me"]

    def _inline_validation_bootstrap(
        self, config, nb, leaderboard, result_ids, limit_str
    ):
        del config, nb, leaderboard, result_ids, limit_str
        return "exp-validation", "hypothesis"

    def _inline_validation_prepare_runtime(self, config, nb, result_ids):
        del nb
        results = {
            "total": len(result_ids),
            "stage0_passed": 0,
            "stage05_passed": 0,
            "stage1_passed": 0,
            "best_loss_ratio": None,
            "best_novelty_score": None,
            "validation_results": [],
        }
        source_map = {
            rid: {
                "result_id": rid,
                "graph_json": "{}",
                "arch_spec_json": None,
                "model_source": "graph_synthesis",
                "novelty_score": 0.1,
                "novelty_confidence": 0.2,
            }
            for rid in result_ids
        }
        return results, "cpu", "cpu", config, source_map

    def _validation_run_seeds(self, *args, **kwargs):
        source_result_id = args[6]
        self.visited_candidates.append(source_result_id)
        del kwargs
        return [
            {
                "seed": 0,
                "passed": True,
                "loss_ratio": 0.25,
                "final_loss": 1.0,
                "n_train_steps": 4,
                "final_lr": 1e-3,
            }
        ]

    def _validation_compute_metrics(self, config, dev_str, source, seed_results):
        del config, dev_str, source, seed_results
        return SimpleNamespace(
            passed_seeds=[{"seed": 0}],
            val_loss_ratio=0.25,
            multi_seed_std=0.0,
            robustness_score=1.0,
            is_unstable=False,
            init_sensitivity_std=0.0,
            val_baseline_ratio=0.9,
            val_normalized_ratio=0.8,
            val_param_efficiency=1.1,
            loss_ratios=[0.25],
            best_seed={"final_loss": 1.0},
            source_params=128,
        )

    def _run_external_evals(self, **kwargs):
        del kwargs
        return SimpleNamespace(is_breakthrough=False)

    def _build_rich_context_for_experiment(self, results, config, hypothesis, nb):
        del results, config, hypothesis, nb
        return {}

    def _analyze_results(self, results, exp_id, nb, context=None):
        del results, exp_id, nb, context
        return {}

    def _build_experiment_perf_report(self, results):
        del results
        return {"summary": "ok"}

    def _maybe_extract_knowledge(self, config, nb, n_experiments):
        del config, nb, n_experiments

    def _emit_event(self, *args, **kwargs):
        del args, kwargs

    def _update_progress(self, **kwargs):
        del kwargs

    def _run_continuous_synthesis(self, *args, **kwargs):
        raise AssertionError("unexpected synthesis fallback")


class _ValidationSeedRunner(_ContinuousValidationMixin):
    def __init__(self):
        self._stop_event = threading.Event()
        self._live_training_context = None
        self.captured_contexts = []

    def _build_model_from_source(
        self,
        model_source,
        arch_spec_json_str,
        graph_json_str,
        config,
        seq_len_override=None,
    ):
        del model_source, arch_spec_json_str, graph_json_str, config, seq_len_override

        class _Model:
            def parameters(self):
                return []

        return _Model()

    def _micro_train(self, model, val_config, dev, seed=None):
        del model, val_config, dev, seed
        self.captured_contexts.append(dict(self._live_training_context))
        return {
            "passed": True,
            "loss_ratio": 0.2,
            "final_loss": 0.9,
            "n_train_steps": 3,
            "final_lr": 1e-3,
        }

    def _emit_event(self, *args, **kwargs):
        del args, kwargs

    def _stable_seed(self, *parts):
        del parts
        return 123


class _InvestigationRunner(_ContinuousInvestigationMixin):
    def __init__(self):
        self._stop_event = threading.Event()
        self._live_training_context = None
        self._lock = threading.Lock()
        self.aria = _FakeAria()
        self.captured = []
        # Production code paths inside _record_inline_investigation_candidate
        # call _register_investigation_eval_future (now indirected through
        # _submit_investigation_eval) and _finalize_inline_investigation calls
        # _wait_for_investigation_eval_futures + _publish_terminal_event.
        # The test mocks the eval submitters to noops, so capture-and-discard
        # implementations are sufficient.
        self.notebook_path = ":memory:"
        self._registered_futures: list = []

    def _register_investigation_eval_future(
        self, *, exp_id, future, kind, source_result_id
    ):
        del exp_id, future, kind, source_result_id

    def _wait_for_investigation_eval_futures(self, exp_id):
        del exp_id
        return None

    def _publish_terminal_event(self, **kwargs):
        del kwargs

    def _fail_experiment_compat(self, *args, **kwargs):
        del args, kwargs

    def _complete_experiment_compat(self, *args, **kwargs):
        del args, kwargs

    def _emit_event(self, *args, **kwargs):
        del args, kwargs

    def _update_progress(self, *args, **kwargs):
        del args, kwargs

    def _auto_escalate(self, *args, **kwargs):
        del args, kwargs

    def _maybe_extract_knowledge(self, *args, **kwargs):
        del args, kwargs

    def _pre_investigation_gate(self, config, nb, leaderboard):
        del config, nb, leaderboard
        return ["skip-me", "resume-me"]

    def _start_preregistered_experiment(
        self,
        nb,
        experiment_type,
        config,
        hypothesis,
        hypothesis_metadata,
        created_by,
    ):
        del nb, experiment_type, config, hypothesis, hypothesis_metadata, created_by
        return "exp-investigation"

    def _build_hypothesis_metadata(self, **kwargs):
        return dict(kwargs)

    def _build_model_from_source(
        self,
        model_source,
        arch_spec_json_str,
        graph_json_str,
        config,
        seq_len_override=None,
    ):
        del model_source, arch_spec_json_str, graph_json_str, config, seq_len_override
        return object()

    def _train_with_program(self, model, tp, inv_config, dev, seed=None):
        del model, tp, inv_config, dev, seed
        ctx = dict(self._live_training_context)
        resume_state = ctx.get("checkpoint_resume_state") or {}
        self.captured.append(
            (ctx["source_result_id"], int(resume_state.get("step", 0)))
        )
        return {"passed": True, "loss_ratio": 0.2, "final_loss": 0.7}

    def _investigation_loss_multiplier(self, screening_lr, best_lr):
        del screening_lr, best_lr
        return 1.0

    def _build_experiment_perf_report(self, results):
        del results
        return {"summary": "ok"}

    def _build_rich_context_for_experiment(self, results, config, hypothesis, nb):
        del results, config, hypothesis, nb
        return {}

    def _analyze_results(self, results, exp_id, nb, context=None):
        del results, exp_id, nb, context
        return {}

    def _auto_escalate(self, results, config, nb, phase):
        del results, config, nb, phase

    def _maybe_extract_knowledge(self, config, nb, n_experiments):
        del config, nb, n_experiments

    def _emit_event(self, *args, **kwargs):
        del args, kwargs

    def _update_progress(self, **kwargs):
        del kwargs

    def _run_continuous_synthesis(self, *args, **kwargs):
        raise AssertionError("unexpected synthesis fallback")

    def _stable_seed(self, *parts):
        del parts
        return 321

    def _cached_json_load(self, value):
        del value
        return {}


def test_continuous_validation_resumes_from_progress_marker(tmp_path, monkeypatch):
    from research.scientist.runner import continuous_validation as cv_mod

    cfg = replace(
        RunConfig(
            checkpoint_dir=str(tmp_path),
            validation_n_seeds=1,
            validation_steps=4,
            device="cpu",
        )
    )
    CheckpointManager(str(tmp_path)).save_phase(
        "exp-validation",
        "validation",
        -1,
        0,
        model_state_dict={},
        optimizer_state_dict={},
        step=0,
        metrics={"candidate_idx": 1},
    )

    monkeypatch.setattr(
        cv_mod,
        "build_validation_entry",
        lambda **kwargs: SimpleNamespace(to_dict=lambda: dict(kwargs)),
    )
    monkeypatch.setattr(cv_mod, "promote_validation_candidate", lambda **kwargs: None)
    monkeypatch.setattr(cv_mod, "run_trajectory_probe", lambda **kwargs: None)
    monkeypatch.setattr(cv_mod, "handle_breakthrough", lambda **kwargs: None)
    monkeypatch.setattr(cv_mod, "evaluate_perf_budget_gate", lambda report: report)

    runner = _ValidationRunner()
    nb = _FakeNotebook()
    leaderboard = [{"result_id": "resume-me"}]
    runner._run_inline_validation(cfg, nb, leaderboard, 1, "limit", "reasoning")

    assert runner.visited_candidates == ["resume-me"]
    state = CheckpointManager(str(tmp_path)).load_phase(
        "exp-validation", "validation", -1, 0
    )
    assert state is not None
    assert CheckpointManager.phase_resume_candidate_idx(state) == 2


def test_validation_seed_resume_state_threads_into_live_context(tmp_path):
    cfg = replace(
        RunConfig(
            checkpoint_dir=str(tmp_path),
            validation_n_seeds=1,
            validation_steps=4,
            phase_checkpoint_step_interval=8,
            device="cpu",
        )
    )
    ckpt = CheckpointManager(str(tmp_path))
    ckpt.save_phase(
        "exp-seed",
        "validation",
        0,
        0,
        model_state_dict={"weight": 1},
        optimizer_state_dict={"state": {}, "param_groups": []},
        step=3,
        metrics={"loss": 1.0},
    )

    runner = _ValidationSeedRunner()
    seed_results = runner._validation_run_seeds(
        cfg,
        cfg,
        "cpu",
        "exp-seed",
        0,
        1,
        "resume-me",
        {},
        "",
        "graph_synthesis",
        None,
        "{}",
        checkpoint_manager=ckpt,
    )

    assert len(seed_results) == 1
    assert runner.captured_contexts[0]["checkpoint_phase"] == "validation"
    assert runner.captured_contexts[0]["checkpoint_candidate_idx"] == 0
    assert runner.captured_contexts[0]["checkpoint_seed_idx"] == 0
    assert runner.captured_contexts[0]["checkpoint_interval_steps"] == 8
    assert runner.captured_contexts[0]["checkpoint_resume_state"]["step"] == 3


def test_continuous_investigation_resumes_and_loads_tp_checkpoint(
    tmp_path, monkeypatch
):
    from research.scientist.runner import continuous_investigation as ci_mod

    cfg = replace(
        RunConfig(
            checkpoint_dir=str(tmp_path),
            n_training_programs=1,
            investigation_steps=4,
            phase_checkpoint_step_interval=6,
            device="cpu",
        )
    )
    ckpt = CheckpointManager(str(tmp_path))
    ckpt.save_phase(
        "exp-investigation",
        "investigation",
        -1,
        0,
        model_state_dict={},
        optimizer_state_dict={},
        step=0,
        metrics={"candidate_idx": 1},
    )
    ckpt.save_phase(
        "exp-investigation",
        "investigation",
        1,
        0,
        model_state_dict={"weight": 1},
        optimizer_state_dict={"state": {}, "param_groups": []},
        step=5,
        metrics={"loss": 0.8},
    )

    monkeypatch.setattr(
        ci_mod,
        "_build_source_map",
        lambda nb, result_ids: {
            rid: {
                "result_id": rid,
                "graph_json": "{}",
                "arch_spec_json": None,
                "model_source": "graph_synthesis",
                "loss_ratio": 0.3,
                "novelty_score": 0.1,
            }
            for rid in result_ids
        },
    )
    monkeypatch.setattr(
        ci_mod, "build_investigation_context", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        ci_mod,
        "synthesize_training_program_batch",
        lambda **kwargs: (
            [_FakeTrainingProgram("tp0")],
            {"scheduling_avg_ms": 0.1, "scheduling_max_ms": 0.1},
        ),
    )
    monkeypatch.setattr(ci_mod, "_submit_benchmark_eval", lambda **kwargs: None)
    # _record_investigation_result lives in _helpers_benchmark, not ci_mod, and is
    # called only via _submit_benchmark_eval — patching that to a no-op above
    # already prevents the real record path from firing.  The previous monkeypatch
    # of ci_mod._record_investigation_result was a stale ref from an earlier
    # refactor that moved the function out.
    monkeypatch.setattr(ci_mod, "evaluate_perf_budget_gate", lambda report: report)
    monkeypatch.setattr(
        ci_mod, "resolve_device", lambda device: SimpleNamespace(type="cpu")
    )

    runner = _InvestigationRunner()
    nb = _FakeNotebook()
    runner._run_inline_investigation(cfg, nb, [], 1, "limit", "reasoning")

    assert runner.captured == [("resume-me", 5)]
    state = CheckpointManager(str(tmp_path)).load_phase(
        "exp-investigation", "investigation", -1, 0
    )
    assert state is not None
    assert CheckpointManager.phase_resume_candidate_idx(state) == 2
