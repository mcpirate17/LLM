import os
import tempfile

import pytest

from research.scientist.notebook import LabNotebook

pytestmark = pytest.mark.unit
from research.tools.backfill_novelty import _fetch_candidates


def _make_result(
    nb: LabNotebook,
    exp_id: str,
    graph_fingerprint: str,
    *,
    novelty_score: float | None,
    stage1_passed: bool = True,
) -> str:
    rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint=graph_fingerprint,
        graph_json="{}",
        stage1_passed=stage1_passed,
        novelty_score=novelty_score,
        structural_novelty=0.2 if novelty_score is not None else None,
        behavioral_novelty=0.8 if novelty_score is not None else None,
        novelty_confidence=0.9 if novelty_score is not None else None,
        fingerprint_json='{"quality":"full"}' if novelty_score is not None else None,
        novelty_scoring_policy_version="full_fp_legacy_v1" if novelty_score is not None else None,
    )
    nb.flush_writes()
    return rid


def test_fetch_candidates_recalculate_top_prefers_leaderboard_order():
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "test.db")
        nb = LabNotebook(db_path)
        exp_id = nb.start_experiment("synthesis", {})

        rid_low = _make_result(nb, exp_id, "fp-low", novelty_score=0.4)
        rid_high = _make_result(nb, exp_id, "fp-high", novelty_score=0.7)
        _make_result(nb, exp_id, "fp-missing", novelty_score=None)

        nb.upsert_leaderboard(
            result_id=rid_low,
            model_source="graph_synthesis",
            screening_novelty=0.4,
            screening_loss_ratio=1.1,
            composite_score=0.2,
        )
        nb.upsert_leaderboard(
            result_id=rid_high,
            model_source="graph_synthesis",
            screening_novelty=0.7,
            screening_loss_ratio=0.9,
            composite_score=0.9,
        )

        candidates = _fetch_candidates(
            nb,
            include_all=False,
            recalculate_top=True,
            limit=2,
            leaderboard_only=True,
        )
        nb.close()

        assert [row["result_id"] for row in candidates] == [rid_high, rid_low]
