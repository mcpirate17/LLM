import json
import os
import sqlite3
import tempfile
import time

from research.scientist.notebook import LabNotebook
from research.synthesis.grammar_support import (
    DBOpWeightCache,
    DBTemplateWeightCache,
    SlotAdaptationCache,
    blend_template_weights_with_db,
)
from research.tools.backfill_stats import backfill


def _capability_graph_json() -> str:
    return json.dumps(
        {
            "nodes": {
                "0": {"op_name": "input"},
                "1": {"op_name": "cumsum"},
                "2": {"op_name": "gather_topk"},
            },
            "metadata": {
                "templates_used": ["cap_template"],
                "motifs_used": ["cap_motif"],
                "template_slot_usage": [
                    {
                        "template_name": "cap_template",
                        "slot_index": 0,
                        "selected_motif_class": "binding",
                        "wildcard": True,
                        "slot_classes": ["residual"],
                    }
                ],
            },
        }
    )


def _template_graph_json(template_name: str, op_name: str) -> str:
    return json.dumps(
        {
            "nodes": {
                "0": {"op_name": "input"},
                "1": {"op_name": op_name},
            },
            "metadata": {
                "templates_used": [template_name],
                "motifs_used": [f"{op_name}_motif"],
            },
        }
    )


def test_generation_stats_migration_adds_capability_columns():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "legacy.db")
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE template_stats (
                template_name TEXT PRIMARY KEY,
                eval_count INTEGER NOT NULL DEFAULT 0,
                s0_pass_count INTEGER NOT NULL DEFAULT 0,
                s1_pass_count INTEGER NOT NULL DEFAULT 0,
                mean_loss REAL,
                min_loss REAL,
                std_loss REAL,
                mean_novelty REAL,
                last_updated REAL NOT NULL
            );
            CREATE TABLE op_stats (
                op_name TEXT PRIMARY KEY,
                eval_count INTEGER NOT NULL DEFAULT 0,
                s0_pass_count INTEGER NOT NULL DEFAULT 0,
                s1_pass_count INTEGER NOT NULL DEFAULT 0,
                mean_loss REAL,
                min_loss REAL,
                std_loss REAL,
                mean_novelty REAL,
                co_occurrence_json TEXT,
                last_updated REAL NOT NULL
            );
            CREATE TABLE motif_stats (
                motif_name TEXT PRIMARY KEY,
                eval_count INTEGER NOT NULL DEFAULT 0,
                s0_pass_count INTEGER NOT NULL DEFAULT 0,
                s1_pass_count INTEGER NOT NULL DEFAULT 0,
                mean_loss REAL,
                min_loss REAL,
                std_loss REAL,
                mean_novelty REAL,
                best_template TEXT,
                last_updated REAL NOT NULL
            );
            CREATE TABLE slot_stats (
                slot_key TEXT PRIMARY KEY,
                template_name TEXT NOT NULL,
                slot_index INTEGER NOT NULL,
                slot_classes TEXT NOT NULL,
                eval_count INTEGER NOT NULL DEFAULT 0,
                s1_pass_count INTEGER NOT NULL DEFAULT 0,
                mean_loss REAL,
                min_loss REAL,
                class_outcomes TEXT,
                wildcard_count INTEGER NOT NULL DEFAULT 0,
                wildcard_s1_count INTEGER NOT NULL DEFAULT 0,
                wildcard_class_outcomes TEXT,
                last_updated REAL NOT NULL
            );
            """
        )
        conn.commit()
        conn.close()

        nb = LabNotebook(db_path, use_native=False)
        cols = {row[1] for row in nb.conn.execute("PRAGMA table_info(template_stats)")}
        nb.close()

        assert "avg_induction_auc" in cols
        assert "avg_binding_auc" in cols
        assert "avg_binding_composite" in cols
        assert "avg_ar_auc" in cols
        assert "avg_hellaswag_acc" in cols
        assert "avg_blimp_overall_accuracy" in cols
        assert "avg_induction_v2_investigation_auc" in cols
        assert "avg_binding_v2_investigation_auc" in cols
        assert "math_space_rate" in cols


def test_backfill_persists_capability_metrics():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "stats.db")
        nb = LabNotebook(db_path, use_native=False)
        exp_id = nb.start_experiment("synthesis", {}, "test")
        nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_cap_1",
            graph_json=_capability_graph_json(),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.4,
            novelty_score=0.3,
            induction_auc=0.02,
            binding_auc=0.01,
            binding_composite=0.012,
            ar_auc=0.06,
            hellaswag_acc=0.34,
            blimp_overall_accuracy=0.72,
            induction_v2_investigation_auc=0.09,
            binding_v2_investigation_auc=0.08,
            graph_uses_math_spaces=1,
        )
        nb.flush_writes()
        nb.close()

        counts = backfill(db_path)
        assert counts["template_stats"] == 1
        assert counts["slot_stats"] == 1

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            """
            SELECT avg_induction_auc, avg_binding_auc, avg_binding_composite,
                   avg_ar_auc, avg_hellaswag_acc, avg_blimp_overall_accuracy,
                   avg_induction_v2_investigation_auc, avg_binding_v2_investigation_auc,
                   math_space_rate
            FROM template_stats
            WHERE template_name = 'cap_template'
            """
        ).fetchone()
        assert row == (0.02, 0.01, 0.012, 0.06, 0.34, 0.72, 0.09, 0.08, 1.0)

        slot_row = conn.execute(
            "SELECT wildcard_class_outcomes FROM slot_stats WHERE slot_key = 'cap_template.slot0'"
        ).fetchone()
        payload = json.loads(slot_row[0])
        assert payload["binding"]["mean_induction_auc"] == 0.02
        assert payload["binding"]["mean_binding_auc"] == 0.01
        assert payload["binding"]["mean_binding_composite"] == 0.012
        assert payload["binding"]["mean_ar_auc"] == 0.06
        assert payload["binding"]["mean_hellaswag_acc"] == 0.34
        assert payload["binding"]["mean_blimp_overall_accuracy"] == 0.72
        assert payload["binding"]["mean_induction_v2_investigation_auc"] == 0.09
        assert payload["binding"]["mean_binding_v2_investigation_auc"] == 0.08
        assert payload["binding"]["math_space_rate"] == 1.0
        conn.close()


def test_backfill_recency_weights_metric_averages():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "recency.db")
        nb = LabNotebook(db_path, use_native=False)
        exp_id = nb.start_experiment("synthesis", {}, "test")
        old_id = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_old",
            graph_json=_template_graph_json("recency_template", "cumsum"),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.9,
            novelty_score=0.1,
            induction_auc=0.0,
        )
        recent_id = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_recent",
            graph_json=_template_graph_json("recency_template", "gather_topk"),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.1,
            novelty_score=0.1,
            induction_auc=1.0,
        )
        nb.flush_writes()
        now = time.time()
        nb.conn.execute(
            "UPDATE program_results SET timestamp = ? WHERE result_id = ?",
            (now - (60 * 24 * 3600), old_id),
        )
        nb.conn.execute(
            "UPDATE program_results SET timestamp = ? WHERE result_id = ?",
            (now, recent_id),
        )
        nb.conn.commit()
        nb.close()

        backfill(db_path)

        conn = sqlite3.connect(db_path)
        avg_induction = conn.execute(
            "SELECT avg_induction_auc FROM template_stats WHERE template_name = 'recency_template'"
        ).fetchone()[0]
        conn.close()

        assert avg_induction > 0.70
        assert avg_induction < 1.0


def test_db_caches_reward_capability_signals():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "weights.db")
        nb = LabNotebook(db_path, use_native=False)
        nb.close()

        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            INSERT INTO template_stats (
                template_name, eval_count, s0_pass_count, s1_pass_count,
                mean_loss, min_loss, std_loss, mean_novelty,
                avg_induction_auc, avg_binding_auc, avg_binding_composite,
                avg_ar_auc, avg_hellaswag_acc, avg_blimp_overall_accuracy,
                avg_induction_v2_investigation_auc, avg_binding_v2_investigation_auc,
                math_space_rate, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "tpl_cap",
                10,
                6,
                4,
                0.5,
                0.4,
                0.0,
                0.1,
                0.02,
                0.01,
                0.012,
                0.06,
                0.34,
                0.72,
                0.09,
                0.08,
                1.0,
                1.0,
            ),
        )
        conn.execute(
            """
            INSERT INTO template_stats (
                template_name, eval_count, s0_pass_count, s1_pass_count,
                mean_loss, min_loss, std_loss, mean_novelty,
                avg_induction_auc, avg_binding_auc, avg_binding_composite,
                avg_ar_auc, avg_hellaswag_acc, avg_blimp_overall_accuracy,
                avg_induction_v2_investigation_auc, avg_binding_v2_investigation_auc,
                math_space_rate, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "tpl_base",
                10,
                6,
                4,
                0.5,
                0.4,
                0.0,
                0.1,
                0.001,
                0.001,
                0.001,
                0.01,
                0.2,
                0.5,
                0.01,
                0.01,
                0.0,
                1.0,
            ),
        )
        conn.execute(
            """
            INSERT INTO op_stats (
                op_name, eval_count, s0_pass_count, s1_pass_count,
                mean_loss, min_loss, std_loss, mean_novelty,
                avg_induction_auc, avg_binding_auc, avg_binding_composite,
                avg_ar_auc, avg_hellaswag_acc, avg_blimp_overall_accuracy,
                avg_induction_v2_investigation_auc, avg_binding_v2_investigation_auc,
                math_space_rate, co_occurrence_json, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "cap_op",
                10,
                6,
                4,
                0.5,
                0.4,
                0.0,
                0.1,
                0.02,
                0.01,
                0.012,
                0.06,
                0.34,
                0.72,
                0.09,
                0.08,
                1.0,
                None,
                1.0,
            ),
        )
        conn.execute(
            """
            INSERT INTO op_stats (
                op_name, eval_count, s0_pass_count, s1_pass_count,
                mean_loss, min_loss, std_loss, mean_novelty,
                avg_induction_auc, avg_binding_auc, avg_binding_composite,
                avg_ar_auc, avg_hellaswag_acc, avg_blimp_overall_accuracy,
                avg_induction_v2_investigation_auc, avg_binding_v2_investigation_auc,
                math_space_rate, co_occurrence_json, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "base_op",
                10,
                6,
                4,
                0.5,
                0.4,
                0.0,
                0.1,
                0.001,
                0.001,
                0.001,
                0.01,
                0.2,
                0.5,
                0.01,
                0.01,
                0.0,
                None,
                1.0,
            ),
        )
        conn.commit()
        conn.close()

        tpl_weights = DBTemplateWeightCache(ttl=0.0).get(db_path)
        op_weights = DBOpWeightCache(ttl=0.0).get(db_path)

        assert tpl_weights["tpl_cap"] > tpl_weights["tpl_base"]
        assert op_weights["cap_op"] > op_weights["base_op"]


def test_slot_adaptation_promotes_capability_positive_wildcards():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "slots.db")
        nb = LabNotebook(db_path, use_native=False)
        nb.close()

        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            INSERT INTO slot_stats (
                slot_key, template_name, slot_index, slot_classes,
                eval_count, s1_pass_count, mean_loss, min_loss,
                avg_induction_auc, avg_binding_auc, avg_binding_composite,
                avg_ar_auc, avg_hellaswag_acc, avg_blimp_overall_accuracy,
                avg_induction_v2_investigation_auc, avg_binding_v2_investigation_auc,
                math_space_rate, class_outcomes, wildcard_count,
                wildcard_s1_count, wildcard_class_outcomes, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "cap_template.slot0",
                "cap_template",
                0,
                json.dumps(["residual"]),
                10,
                2,
                0.7,
                0.6,
                0.001,
                0.001,
                0.001,
                0.01,
                0.2,
                0.5,
                0.01,
                0.01,
                0.0,
                json.dumps({}),
                5,
                1,
                json.dumps(
                    {
                        "binding": {
                            "n": 5,
                            "s1": 1,
                            "mean_loss": 0.7,
                            "mean_induction_auc": 0.02,
                            "mean_binding_auc": 0.01,
                            "mean_binding_composite": 0.012,
                            "mean_ar_auc": 0.06,
                            "mean_hellaswag_acc": 0.34,
                            "mean_blimp_overall_accuracy": 0.72,
                            "mean_induction_v2_investigation_auc": 0.09,
                            "mean_binding_v2_investigation_auc": 0.08,
                            "math_space_rate": 1.0,
                        }
                    }
                ),
                1.0,
            ),
        )
        conn.commit()
        conn.close()

        adaptations = SlotAdaptationCache(ttl=0.0).get(db_path)
        assert adaptations["cap_template.slot0"] == ["binding"]


def test_template_cache_avoids_overblaming_salvageable_templates():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "salvage.db")
        nb = LabNotebook(db_path, use_native=False)
        nb.close()

        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            INSERT INTO template_stats (
                template_name, eval_count, s0_pass_count, s1_pass_count,
                mean_loss, min_loss, std_loss, mean_novelty,
                avg_induction_auc, avg_binding_auc, avg_binding_composite,
                avg_ar_auc, avg_hellaswag_acc, avg_blimp_overall_accuracy,
                avg_induction_v2_investigation_auc, avg_binding_v2_investigation_auc,
                math_space_rate, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "tpl_salvage",
                20,
                4,
                1,
                0.9,
                0.8,
                0.0,
                0.1,
                0.001,
                0.001,
                0.001,
                0.01,
                0.2,
                0.5,
                0.01,
                0.01,
                0.0,
                1.0,
            ),
        )
        conn.execute(
            """
            INSERT INTO template_stats (
                template_name, eval_count, s0_pass_count, s1_pass_count,
                mean_loss, min_loss, std_loss, mean_novelty,
                avg_induction_auc, avg_binding_auc, avg_binding_composite,
                avg_ar_auc, avg_hellaswag_acc, avg_blimp_overall_accuracy,
                avg_induction_v2_investigation_auc, avg_binding_v2_investigation_auc,
                math_space_rate, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "tpl_dead",
                20,
                4,
                1,
                0.9,
                0.8,
                0.0,
                0.1,
                0.001,
                0.001,
                0.001,
                0.01,
                0.2,
                0.5,
                0.01,
                0.01,
                0.0,
                1.0,
            ),
        )
        conn.execute(
            """
            INSERT INTO slot_stats (
                slot_key, template_name, slot_index, slot_classes,
                eval_count, s1_pass_count, mean_loss, min_loss,
                avg_induction_auc, avg_binding_auc, avg_binding_composite,
                avg_ar_auc, avg_hellaswag_acc, avg_blimp_overall_accuracy,
                avg_induction_v2_investigation_auc, avg_binding_v2_investigation_auc,
                math_space_rate, class_outcomes, wildcard_count,
                wildcard_s1_count, wildcard_class_outcomes, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "tpl_salvage.slot0",
                "tpl_salvage",
                0,
                json.dumps(["residual"]),
                10,
                1,
                0.9,
                0.8,
                0.001,
                0.001,
                0.001,
                0.01,
                0.2,
                0.5,
                0.01,
                0.01,
                0.0,
                json.dumps({}),
                5,
                1,
                json.dumps(
                    {
                        "binding": {
                            "n": 5,
                            "s1": 1,
                            "mean_loss": 0.7,
                            "mean_induction_auc": 0.02,
                            "mean_binding_auc": 0.01,
                            "mean_binding_composite": 0.012,
                            "mean_ar_auc": 0.06,
                            "mean_hellaswag_acc": 0.34,
                            "mean_blimp_overall_accuracy": 0.72,
                            "mean_induction_v2_investigation_auc": 0.09,
                            "mean_binding_v2_investigation_auc": 0.08,
                            "math_space_rate": 1.0,
                        }
                    }
                ),
                1.0,
            ),
        )
        conn.commit()
        conn.close()

        tpl_weights = DBTemplateWeightCache(ttl=0.0).get(db_path)
        assert tpl_weights["tpl_salvage"] > tpl_weights["tpl_dead"]


def test_template_rescue_requires_supported_slot_class_evidence():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "weak_salvage.db")
        nb = LabNotebook(db_path, use_native=False)
        nb.close()

        conn = sqlite3.connect(db_path)
        for tpl_name in ("tpl_lucky", "tpl_dead"):
            conn.execute(
                """
                INSERT INTO template_stats (
                    template_name, eval_count, s0_pass_count, s1_pass_count,
                    mean_loss, min_loss, std_loss, mean_novelty,
                    avg_induction_auc, avg_binding_auc, avg_binding_composite,
                    avg_ar_auc, avg_hellaswag_acc, avg_blimp_overall_accuracy,
                    avg_induction_v2_investigation_auc, avg_binding_v2_investigation_auc,
                    math_space_rate, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tpl_name,
                    20,
                    4,
                    1,
                    0.9,
                    0.8,
                    0.0,
                    0.1,
                    0.001,
                    0.001,
                    0.001,
                    0.01,
                    0.2,
                    0.5,
                    0.01,
                    0.01,
                    0.0,
                    1.0,
                ),
            )
        conn.execute(
            """
            INSERT INTO slot_stats (
                slot_key, template_name, slot_index, slot_classes,
                eval_count, s1_pass_count, mean_loss, min_loss,
                avg_induction_auc, avg_binding_auc, avg_binding_composite,
                avg_ar_auc, avg_hellaswag_acc, avg_blimp_overall_accuracy,
                avg_induction_v2_investigation_auc, avg_binding_v2_investigation_auc,
                math_space_rate, class_outcomes, wildcard_count,
                wildcard_s1_count, wildcard_class_outcomes, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "tpl_lucky.slot0",
                "tpl_lucky",
                0,
                json.dumps(["residual"]),
                10,
                1,
                0.9,
                0.8,
                0.001,
                0.001,
                0.001,
                0.01,
                0.2,
                0.5,
                0.01,
                0.01,
                0.0,
                json.dumps({}),
                1,
                1,
                json.dumps(
                    {
                        "binding": {
                            "n": 1,
                            "s1": 1,
                            "mean_loss": 0.1,
                            "mean_induction_auc": 0.02,
                            "mean_binding_auc": 0.01,
                            "mean_binding_composite": 0.012,
                            "mean_ar_auc": 0.06,
                            "mean_hellaswag_acc": 0.34,
                            "mean_blimp_overall_accuracy": 0.72,
                            "mean_induction_v2_investigation_auc": 0.09,
                            "mean_binding_v2_investigation_auc": 0.08,
                            "math_space_rate": 1.0,
                        }
                    }
                ),
                1.0,
            ),
        )
        conn.commit()
        conn.close()

        tpl_weights = DBTemplateWeightCache(ttl=0.0).get(db_path)
        assert abs(tpl_weights["tpl_lucky"] - tpl_weights["tpl_dead"]) < 1e-9


def test_existing_template_priors_are_blended_with_db_evidence():
    blended = blend_template_weights_with_db(
        {"tpl_cap": 9.0, "tpl_base": 9.0},
        {"tpl_cap": 4.0, "tpl_base": 0.25},
    )

    assert blended["tpl_cap"] > blended["tpl_base"]
    assert blended["tpl_cap"] > 9.0
    assert blended["tpl_base"] < 9.0
