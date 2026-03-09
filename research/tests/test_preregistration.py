import os
import tempfile

import pytest

from research.scientist.notebook import LabNotebook
from research.scientist.runner import ExperimentRunner, RunConfig

pytestmark = pytest.mark.unit


def test_runner_blocks_when_preregistration_required_but_auto_disabled():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "prereg_block.db")
    runner = ExperimentRunner(db_path)
    cfg = RunConfig(
        n_programs=1,
        stage1_steps=1,
        require_preregistration=True,
        auto_preregister=False,
    )

    with pytest.raises(Exception):
        runner.start_experiment(cfg, hypothesis="test hypothesis")


def test_notebook_preregistration_round_trip_and_experiment_link():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "prereg_link.db")
    nb = LabNotebook(db_path)
    try:
        prereg = {
            "hypothesis": {
                "statement": "A improves B",
                "variables": {"independent": ["A"], "dependent": ["B"], "controls": ["C"]},
                "expected_direction": {"B": "increase"},
                "success_criteria": {"B": ">0"},
            },
            "analysis_plan": {
                "primary_metrics": ["loss_ratio"],
                "secondary_metrics": ["novelty_score"],
                "thresholds": {"loss_ratio": {"operator": "<", "value": 1.0}},
                "baseline_comparison": {"method": "relative", "source": "baseline"},
            },
            "falsification_conditions": ["loss_ratio >= 1.0"],
            "confounders_checklist": [{"name": "seed_instability", "checked": False}],
            "exploratory": False,
        }
        prereg_id = nb.create_preregistration("synthesis", prereg, created_by="test")
        exp_id = nb.start_experiment(
            "synthesis",
            {"n_programs": 1},
            hypothesis="test",
            preregistration_id=prereg_id,
            require_preregistration=True,
        )

        linked = nb.get_preregistration_for_experiment(exp_id)
        assert linked is not None
        assert linked["preregistration_id"] == prereg_id
    finally:
        nb.close()
