import os
import tempfile

import pytest

from research.scientist.notebook import LabNotebook

pytestmark = pytest.mark.unit
from research.tools.repair_leaderboard_tier_data import repair_leaderboard_tier_data


def test_repair_leaderboard_tier_data_fills_missing_tier_fields():
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "repair.db")
        nb = LabNotebook(db_path)
        exp_id = nb.start_experiment("synthesis", {})

        screening_rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp-screen",
            graph_json="{}",
            stage1_passed=True,
            loss_ratio=0.4,
            novelty_score=0.7,
        )
        inv_rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp-inv",
            graph_json="{}",
            stage1_passed=True,
            loss_ratio=0.2,
            novelty_score=0.6,
        )
        val_rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp-val",
            graph_json="{}",
            stage1_passed=True,
            loss_ratio=0.1,
            novelty_score=0.8,
            validation_loss_ratio=0.12,
            baseline_loss_ratio=0.95,
            init_sensitivity_std=0.018,
        )
        nb.flush_writes()

        nb.upsert_leaderboard(
            result_id=screening_rid,
            model_source="graph_synthesis",
            tier="screening",
            screening_loss_ratio=0.4,
            screening_passed=True,
        )
        nb.upsert_leaderboard(
            result_id=inv_rid,
            model_source="graph_synthesis",
            tier="investigation",
            screening_loss_ratio=0.2,
            screening_passed=True,
        )
        nb.upsert_leaderboard(
            result_id=val_rid,
            model_source="graph_synthesis",
            tier="validation",
            screening_loss_ratio=0.1,
            screening_novelty=0.8,
            screening_passed=True,
            investigation_loss_ratio=0.11,
            investigation_robustness=1.0,
            investigation_passed=True,
            init_sensitivity_std=0.018,
        )
        nb.conn.execute("UPDATE leaderboard SET screening_novelty = NULL WHERE result_id = ?", (screening_rid,))
        nb.conn.execute(
            """
            UPDATE leaderboard
            SET screening_novelty = NULL,
                investigation_loss_ratio = NULL,
                investigation_robustness = NULL,
                investigation_passed = NULL
            WHERE result_id = ?
            """,
            (inv_rid,),
        )
        nb.conn.execute(
            """
            UPDATE leaderboard
            SET validation_loss_ratio = NULL,
                validation_baseline_ratio = NULL,
                validation_multi_seed_std = NULL,
                validation_passed = NULL
            WHERE result_id = ?
            """,
            (val_rid,),
        )
        nb.conn.commit()
        nb.close()

        counts = repair_leaderboard_tier_data(db_path)
        assert counts["repaired"] == 3

        nb2 = LabNotebook(db_path)
        screen = nb2.conn.execute(
            "SELECT screening_novelty FROM leaderboard WHERE result_id = ?",
            (screening_rid,),
        ).fetchone()
        inv = nb2.conn.execute(
            """
            SELECT screening_novelty, investigation_loss_ratio,
                   investigation_robustness, investigation_passed
            FROM leaderboard WHERE result_id = ?
            """,
            (inv_rid,),
        ).fetchone()
        val = nb2.conn.execute(
            """
            SELECT validation_loss_ratio, validation_baseline_ratio,
                   validation_multi_seed_std, validation_passed
            FROM leaderboard WHERE result_id = ?
            """,
            (val_rid,),
        ).fetchone()
        nb2.close()

        assert float(screen["screening_novelty"]) == 0.7
        assert float(inv["screening_novelty"]) == 0.6
        assert float(inv["investigation_loss_ratio"]) == 0.2
        assert float(inv["investigation_robustness"]) == 1.0
        assert int(inv["investigation_passed"]) == 1
        assert float(val["validation_loss_ratio"]) == 0.12
        assert float(val["validation_baseline_ratio"]) == 0.95
        assert float(val["validation_multi_seed_std"]) == 0.018
        assert int(val["validation_passed"]) == 1
