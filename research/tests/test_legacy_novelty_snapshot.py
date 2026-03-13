import os
import tempfile

import pytest

from research.scientist.notebook import LabNotebook

pytestmark = pytest.mark.unit
from research.tools.snapshot_legacy_novelty_scores import snapshot_legacy_novelty_scores


def test_record_program_result_persists_policy_and_full_flag():
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "test.db")
        nb = LabNotebook(db_path)
        exp_id = nb.start_experiment("synthesis", {})
        rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp",
            graph_json="{}",
            novelty_score=0.6,
            fingerprint_full_ran=True,
        )
        nb.flush_writes()
        row = nb.conn.execute(
            "SELECT novelty_scoring_policy_version, fingerprint_full_ran FROM program_results WHERE result_id = ?",
            (rid,),
        ).fetchone()
        nb.close()

        assert row is not None
        assert row["novelty_scoring_policy_version"] == "gated_lightning_v1"
        assert int(row["fingerprint_full_ran"]) == 1


def test_snapshot_legacy_novelty_scores_preserves_current_values():
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "test.db")
        nb = LabNotebook(db_path)
        exp_id = nb.start_experiment("synthesis", {})
        rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp",
            graph_json="{}",
            novelty_score=0.7,
            structural_novelty=0.4,
            behavioral_novelty=0.8,
            novelty_confidence=0.9,
            novelty_raw_score=0.68,
            novelty_reference_version="nv1",
            novelty_valid_for_promotion=1,
            novelty_validity_reason="artifact_reference",
            fingerprint_json='{"quality":"full"}',
            novelty_scoring_policy_version="full_fp_legacy_v1",
        )
        nb.flush_writes()
        nb.close()

        count = snapshot_legacy_novelty_scores(db_path)
        assert count == 1

        nb2 = LabNotebook(db_path)
        row = nb2.conn.execute(
            """
            SELECT novelty_score, novelty_score_legacy, structural_novelty_legacy,
                   behavioral_novelty_legacy, novelty_confidence_legacy,
                   novelty_raw_score_legacy, novelty_reference_version_legacy,
                   novelty_valid_for_promotion_legacy, novelty_validity_reason_legacy,
                   fingerprint_json_legacy
            FROM program_results WHERE result_id = ?
            """,
            (rid,),
        ).fetchone()
        nb2.close()

        assert row is not None
        assert float(row["novelty_score_legacy"]) == float(row["novelty_score"])
        assert float(row["structural_novelty_legacy"]) == 0.4
        assert float(row["behavioral_novelty_legacy"]) == 0.8
        assert float(row["novelty_confidence_legacy"]) == 0.9
        assert float(row["novelty_raw_score_legacy"]) == 0.68
        assert row["novelty_reference_version_legacy"] == "nv1"
        assert int(row["novelty_valid_for_promotion_legacy"]) == 1
        assert row["novelty_validity_reason_legacy"] == "artifact_reference"
        assert row["fingerprint_json_legacy"] == '{"quality":"full"}'
