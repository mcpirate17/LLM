from __future__ import annotations

import os

from research.tools import measured_rescue_ab as mab


class FakeNotebook:
    def __init__(self) -> None:
        self.started: list[dict] = []
        self.completed: list[dict] = []
        self.failed: list[dict] = []
        self.closed = False
        self.updated_ops: list[str] = []

    def start_experiment(self, **kwargs):
        exp_id = f"exp{len(self.started) + 1}"
        self.started.append({"experiment_id": exp_id, **kwargs})
        return exp_id

    def complete_experiment(self, **kwargs):
        self.completed.append(kwargs)

    def fail_experiment(self, experiment_id, error, results=None):
        self.failed.append(
            {"experiment_id": experiment_id, "error": error, "results": results}
        )

    def update_op_success_rates(self, exp_id):
        self.updated_ops.append(exp_id)

    def strip_graph_json_for_failures(self, exp_id):
        pass

    def update_failure_signatures(self, exp_id):
        pass

    def flush_writes(self):
        pass

    def close(self):
        self.closed = True


class FakeRunner:
    instances: list["FakeRunner"] = []

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.notebooks: list[FakeNotebook] = []
        self.execute_calls: list[dict] = []
        self.ensure_math_spaces_called = False
        FakeRunner.instances.append(self)

    def _ensure_math_spaces(self):
        self.ensure_math_spaces_called = True

    def _make_notebook(self):
        nb = FakeNotebook()
        self.notebooks.append(nb)
        return nb

    def _execute_experiment(
        self,
        exp_id,
        config,
        nb,
        use_learned_grammar=True,
    ):
        self.execute_calls.append(
            {
                "exp_id": exp_id,
                "config": config,
                "use_learned_grammar": use_learned_grammar,
                "env": {
                    "ARIA_MEASURED_RESCUE": os.environ.get("ARIA_MEASURED_RESCUE"),
                    "ARIA_MEASURED_RESCUE_TAU": os.environ.get(
                        "ARIA_MEASURED_RESCUE_TAU"
                    ),
                    "ARIA_MEASURED_RESCUE_MAX": os.environ.get(
                        "ARIA_MEASURED_RESCUE_MAX"
                    ),
                    "ARIA_MEASURED_RESCUE_PROBE_BUDGET": os.environ.get(
                        "ARIA_MEASURED_RESCUE_PROBE_BUDGET"
                    ),
                },
            }
        )
        enabled = os.environ.get("ARIA_MEASURED_RESCUE") == "1"
        records = [{"fingerprint": "fp1"}] if enabled else []
        return {
            "total": config.n_programs,
            "stage0_passed": 3,
            "stage05_passed": 2,
            "stage1_passed": 1 if enabled else 0,
            "best_loss_ratio": 0.7 if enabled else 0.8,
            "elapsed_seconds": 0.1,
            "funnel_counts": {
                "gbm_prescreener_skipped": 7,
                "post_gbm_prescreener": 5,
                "measured_rescued": len(records),
            },
            "measured_rescue_records": records,
        }


def test_bounded_config_disables_follow_on_automation():
    settings = mab.RescueABSettings(allow_unproven_screening_ensemble=True)
    config = mab.build_bounded_config(settings)

    assert config.mode == "single"
    assert config.continuous is False
    assert config.max_experiments == 1
    assert config.gbm_prescreener_enabled is True
    assert config.auto_scale_up is False
    assert config.auto_report is False
    assert config.auto_investigate is False
    assert config.auto_validate is False
    assert config.enable_campaigns is False
    assert config.auto_go_no_go is False
    assert config.enable_causal_ablation is False
    assert config.llm_decision_interval == 0
    assert config.allow_unproven_ml_influence is True


def test_run_bounded_ab_uses_direct_execute_and_scoped_env(tmp_path, monkeypatch):
    FakeRunner.instances.clear()
    monkeypatch.setattr(
        mab,
        "_prescreener_preflight",
        lambda config: {"active": True, "reason": "test"},
    )
    monkeypatch.setenv("ARIA_MEASURED_RESCUE", "preserve")
    monkeypatch.setenv("ARIA_MEASURED_RESCUE_TAU", "0.99")
    settings = mab.RescueABSettings(
        db_path="test.db",
        report_dir=tmp_path,
        n_programs=9,
        tau=0.02,
        max_rescue=3,
        probe_budget=11,
        dry_run=False,
        skip_backup_check=True,
        allow_unproven_screening_ensemble=True,
    )

    payload = mab.run_bounded_ab(settings, runner_factory=FakeRunner)

    runner = FakeRunner.instances[0]
    assert runner.ensure_math_spaces_called is True
    assert [call["exp_id"] for call in runner.execute_calls] == ["exp1", "exp1"]
    assert runner.execute_calls[0]["env"]["ARIA_MEASURED_RESCUE"] == "0"
    assert runner.execute_calls[1]["env"]["ARIA_MEASURED_RESCUE"] == "1"
    assert runner.execute_calls[1]["env"]["ARIA_MEASURED_RESCUE_TAU"] == "0.02"
    assert runner.execute_calls[1]["env"]["ARIA_MEASURED_RESCUE_MAX"] == "3"
    assert runner.execute_calls[1]["env"]["ARIA_MEASURED_RESCUE_PROBE_BUDGET"] == "11"
    assert os.environ["ARIA_MEASURED_RESCUE"] == "preserve"
    assert os.environ["ARIA_MEASURED_RESCUE_TAU"] == "0.99"
    assert len(runner.notebooks) == 2
    assert all(nb.completed for nb in runner.notebooks)
    assert all(not nb.failed for nb in runner.notebooks)
    assert all(nb.closed for nb in runner.notebooks)
    assert payload["comparison"]["delta_stage1_passed"] == 1
    assert payload["comparison"]["delta_measured_rescued"] == 1


def test_dry_run_writes_plan_without_runner(tmp_path):
    settings = mab.RescueABSettings(report_dir=tmp_path, dry_run=True)

    payload = mab.run_bounded_ab(settings, runner_factory=FakeRunner)

    assert payload["dry_run"] is True
    assert [arm["arm"] for arm in payload["arms"]] == ["off", "on"]
    assert payload["arms"][1]["env"]["ARIA_MEASURED_RESCUE"] == "1"
    assert list(tmp_path.glob("measured_rescue_ab_*.json"))
    assert list(tmp_path.glob("measured_rescue_ab_*.md"))


def test_run_bounded_ab_fails_fast_when_prescreener_inactive(tmp_path, monkeypatch):
    FakeRunner.instances.clear()
    monkeypatch.setattr(
        mab,
        "_prescreener_preflight",
        lambda config: {"active": False, "reason": "blocked_by_ml_trust_policy"},
    )
    settings = mab.RescueABSettings(
        report_dir=tmp_path,
        dry_run=False,
        skip_backup_check=True,
    )

    try:
        mab.run_bounded_ab(settings, runner_factory=FakeRunner)
    except RuntimeError as exc:
        assert "requires an active screening ensemble" in str(exc)
    else:
        raise AssertionError("expected inactive prescreener to fail fast")
    assert FakeRunner.instances == []
