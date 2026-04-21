from __future__ import annotations

from research.scientist.notebook import LabNotebook
from research.tools import backfill


def _insert_candidate(
    nb: LabNotebook,
    *,
    result_id: str,
    fingerprint: str,
    tier: str,
    score: float,
    hellaswag_acc: float | None = None,
    induction_auc: float | None = None,
    binding_auc: float | None = None,
    ar_auc: float | None = None,
    blimp_acc: float | None = None,
) -> None:
    nb.conn.execute(
        """
        INSERT INTO program_results(
            result_id,
            timestamp,
            graph_json,
            graph_fingerprint,
            hellaswag_acc,
            induction_auc,
            binding_auc,
            ar_auc,
            blimp_overall_accuracy,
            stage1_passed,
            intentional_rerun_reason
        ) VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, 1, ?)
        """,
        (
            result_id,
            '{"nodes":[],"edges":[]}',
            fingerprint,
            hellaswag_acc,
            induction_auc,
            binding_auc,
            ar_auc,
            blimp_acc,
            "unit_test_duplicate" if result_id == "weak" else None,
        ),
    )
    nb.conn.execute(
        """
        INSERT INTO leaderboard(
            entry_id, result_id, timestamp, tier, composite_score, is_reference, model_source
        ) VALUES (?, ?, datetime('now'), ?, ?, 0, 'unit')
        """,
        (f"e_{result_id}", result_id, tier, score),
    )


def test_query_candidates_limits_per_tier_and_preserves_requested_order(tmp_path):
    nb = LabNotebook(tmp_path / "lab_notebook.db")
    try:
        _insert_candidate(
            nb,
            result_id="screen_a",
            fingerprint="fp_screen_a",
            tier="screening",
            score=9.0,
        )
        _insert_candidate(
            nb,
            result_id="screen_b",
            fingerprint="fp_screen_b",
            tier="screening",
            score=8.0,
        )
        _insert_candidate(
            nb,
            result_id="screen_c",
            fingerprint="fp_screen_c",
            tier="screening",
            score=7.0,
        )
        _insert_candidate(
            nb, result_id="val_a", fingerprint="fp_val_a", tier="validation", score=20.0
        )
        _insert_candidate(
            nb, result_id="val_b", fingerprint="fp_val_b", tier="validation", score=19.0
        )
        _insert_candidate(
            nb, result_id="val_c", fingerprint="fp_val_c", tier="validation", score=18.0
        )
        nb.conn.commit()

        candidates = backfill.query_candidates(
            nb,
            tiers=["validation", "screening"],
            top_per_tier=2,
            null_column=None,
            force=False,
        )

        assert [c.result_id for c in candidates] == [
            "val_a",
            "val_b",
            "screen_a",
            "screen_b",
        ]
    finally:
        nb.close()


def test_query_signal_candidates_dedupes_by_fingerprint_before_sharding(tmp_path):
    nb = LabNotebook(tmp_path / "lab_notebook.db")
    try:
        _insert_candidate(
            nb,
            result_id="strong",
            fingerprint="fp_shared",
            tier="screening",
            score=10.0,
            induction_auc=0.25,
        )
        _insert_candidate(
            nb,
            result_id="weak",
            fingerprint="fp_shared",
            tier="screening",
            score=9.0,
            induction_auc=0.10,
        )
        _insert_candidate(
            nb,
            result_id="other",
            fingerprint="fp_other",
            tier="validation",
            score=8.0,
            hellaswag_acc=0.31,
        )
        nb.conn.commit()

        candidates = backfill.query_signal_candidates(
            nb,
            null_column=None,
            force=False,
            shard=(0, 2),
        )

        assert len(candidates) == 1
        assert candidates[0].result_id == "strong"
        assert candidates[0].graph_fingerprint == "fp_shared"
    finally:
        nb.close()
