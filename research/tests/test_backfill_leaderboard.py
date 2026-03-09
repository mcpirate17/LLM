"""Tests for research.tools.backfill_leaderboard."""
import json
import pytest

from research.scientist.notebook import LabNotebook

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path):
    """Create a LabNotebook with a small test dataset."""
    db_path = str(tmp_path / "test_backfill.db")
    nb = LabNotebook(db_path)

    exp_id = nb.start_experiment(
        experiment_type="backfill_test",
        config={"dim": 64, "n_layers": 2},
        hypothesis="test backfill",
    )

    # Entry 1: has final_loss and validation_loss but no generalization_gap
    r1 = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp_001",
        graph_json=json.dumps({"nodes": [], "model_dim": 64}),
        stage1_passed=True,
        loss_ratio=0.5,
        final_loss=2.0,
        validation_loss=2.5,
    )
    nb.flush_writes()  # record_program_result uses async write queue
    e1 = nb.upsert_leaderboard(
        result_id=r1,
        model_source="graph_synthesis",
        architecture_desc="test entry 1",
        screening_loss_ratio=0.5,
        tier="screening",
    )

    # Entry 2: has loss_ratio but leaderboard investigation_loss_ratio is NULL
    r2 = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp_002",
        graph_json=json.dumps({"nodes": [], "model_dim": 64}),
        stage1_passed=True,
        loss_ratio=0.42,
        discovery_loss_ratio=0.6,
    )
    nb.flush_writes()
    e2 = nb.upsert_leaderboard(
        result_id=r2,
        model_source="graph_synthesis",
        architecture_desc="test entry 2",
        tier="screening",
    )

    # Entry 3: reference entry (should be skipped by compile/train phases)
    r3 = "ref_gpt2_test"
    import time as _time
    nb.conn.execute(
        "INSERT INTO program_results (result_id, experiment_id, graph_fingerprint,"
        " graph_json, stage1_passed, loss_ratio, timestamp)"
        " VALUES (?, ?, ?, ?, 1, 0.45, ?)",
        (r3, exp_id, "fp_ref", json.dumps({"nodes": []}), _time.time()),
    )
    nb.conn.commit()
    nb.upsert_leaderboard(
        result_id=r3,
        model_source="reference",
        architecture_desc="GPT-2 ref",
        tier="screening",
        is_reference=True,
        reference_name="GPT-2",
    )

    return nb, exp_id, (r1, e1), (r2, e2), (r3,)


# ---------------------------------------------------------------------------
# Phase 1 tests
# ---------------------------------------------------------------------------

class TestPhaseSql:
    def test_generalization_gap_backfill(self, tmp_path):
        """generalization_gap should be computed from validation_loss - final_loss."""
        nb, _, (r1, e1), _, _ = _make_db(tmp_path)

        # Confirm gap is NULL before
        row = nb.conn.execute(
            "SELECT generalization_gap FROM program_results WHERE result_id = ?",
            (r1,),
        ).fetchone()
        assert row["generalization_gap"] is None

        from research.tools.backfill_leaderboard import phase_sql

        phase_sql(nb, dry_run=False)

        row = nb.conn.execute(
            "SELECT generalization_gap FROM program_results WHERE result_id = ?",
            (r1,),
        ).fetchone()
        assert row["generalization_gap"] is not None
        assert abs(row["generalization_gap"] - 0.5) < 1e-6  # 2.5 - 2.0

    def test_investigation_loss_ratio_copy(self, tmp_path):
        """investigation_loss_ratio should be copied from program_results.loss_ratio."""
        nb, _, _, (r2, e2), _ = _make_db(tmp_path)

        row = nb.conn.execute(
            "SELECT investigation_loss_ratio FROM leaderboard WHERE entry_id = ?",
            (e2,),
        ).fetchone()
        assert row["investigation_loss_ratio"] is None

        from research.tools.backfill_leaderboard import phase_sql

        phase_sql(nb, dry_run=False)

        row = nb.conn.execute(
            "SELECT investigation_loss_ratio FROM leaderboard WHERE entry_id = ?",
            (e2,),
        ).fetchone()
        assert row["investigation_loss_ratio"] is not None
        assert abs(row["investigation_loss_ratio"] - 0.42) < 1e-6

    def test_dry_run_no_writes(self, tmp_path):
        """Dry run should not modify the database."""
        nb, _, (r1, e1), _, _ = _make_db(tmp_path)

        from research.tools.backfill_leaderboard import phase_sql

        phase_sql(nb, dry_run=True)

        row = nb.conn.execute(
            "SELECT generalization_gap FROM program_results WHERE result_id = ?",
            (r1,),
        ).fetchone()
        assert row["generalization_gap"] is None

    def test_idempotency(self, tmp_path):
        """Running phase_sql twice should produce the same result."""
        nb, _, (r1, e1), _, _ = _make_db(tmp_path)

        from research.tools.backfill_leaderboard import phase_sql

        phase_sql(nb, dry_run=False)

        row1 = nb.conn.execute(
            "SELECT generalization_gap FROM program_results WHERE result_id = ?",
            (r1,),
        ).fetchone()

        # Run again
        phase_sql(nb, dry_run=False)

        row2 = nb.conn.execute(
            "SELECT generalization_gap FROM program_results WHERE result_id = ?",
            (r1,),
        ).fetchone()

        assert row1["generalization_gap"] == row2["generalization_gap"]


# ---------------------------------------------------------------------------
# promote_to_tier whitelist fix test
# ---------------------------------------------------------------------------

class TestPromoteToTierWhitelist:
    def test_routing_savings_ratio_persisted(self, tmp_path):
        """routing_savings_ratio should survive promote_to_tier."""
        nb, _, (r1, e1), _, _ = _make_db(tmp_path)

        nb.promote_to_tier(
            e1, "investigation",
            routing_savings_ratio=0.85,
            compression_ratio=0.72,
        )

        row = nb.conn.execute(
            "SELECT routing_savings_ratio, compression_ratio FROM leaderboard"
            " WHERE entry_id = ?",
            (e1,),
        ).fetchone()
        assert row["routing_savings_ratio"] is not None
        assert abs(row["routing_savings_ratio"] - 0.85) < 1e-6
        assert abs(row["compression_ratio"] - 0.72) < 1e-6

    def test_new_eval_columns_persisted(self, tmp_path):
        """New eval columns should survive promote_to_tier."""
        nb, _, (r1, e1), _, _ = _make_db(tmp_path)

        nb.promote_to_tier(
            e1, "screening",
            activation_sparsity_score=0.9,
            dead_neuron_ratio=0.05,
            routing_collapse_score=0.8,
            wikitext_perplexity=150.0,
            wikitext_score=0.6,
            tinystories_perplexity=120.0,
            tinystories_score=0.65,
            cross_task_score=0.7,
            efficiency_wall_score=0.75,
            max_viable_seq_len=512,
            scaling_regime="linear",
        )

        row = nb.conn.execute(
            "SELECT activation_sparsity_score, wikitext_perplexity, cross_task_score,"
            " efficiency_wall_score, scaling_regime"
            " FROM leaderboard WHERE entry_id = ?",
            (e1,),
        ).fetchone()

        assert abs(row["activation_sparsity_score"] - 0.9) < 1e-6
        assert abs(row["wikitext_perplexity"] - 150.0) < 1e-6
        assert abs(row["cross_task_score"] - 0.7) < 1e-6
        assert abs(row["efficiency_wall_score"] - 0.75) < 1e-6
        assert row["scaling_regime"] == "linear"


# ---------------------------------------------------------------------------
# Status report test
# ---------------------------------------------------------------------------

class TestStatus:
    def test_print_status_runs(self, tmp_path, capsys):
        """print_status should run without error on a test DB."""
        nb, *_ = _make_db(tmp_path)

        from research.tools.backfill_leaderboard import print_status

        print_status(nb)
        captured = capsys.readouterr()
        assert "Leaderboard Backfill Status" in captured.out
        assert "entries" in captured.out
