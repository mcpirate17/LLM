"""Tests for the Bayesian insight system.

Covers: schema migration, Bayesian updates, display_only enforcement,
seed idempotency, grammar integration, and confidence filtering.
"""

from __future__ import annotations

import pytest

from research.scientist.notebook import LabNotebook


@pytest.fixture
def nb(tmp_path):
    """Fresh LabNotebook in a temp directory."""
    db = tmp_path / "test.db"
    notebook = LabNotebook(str(db))
    yield notebook
    notebook.close()


class TestBayesianUpdate:
    def test_update_converges(self, nb):
        """Alpha/beta updates converge confidence to true rate."""
        iid = nb.record_insight(
            category="pattern",
            content="test insight",
            semantic_key="test:converge",
            alpha=1.0,
            beta_=1.0,
            insight_level="structural",
            evidence_json={"test": "manual", "n": 10},
        )
        # Simulate 8 successes, 2 failures → true rate ~0.8
        for _ in range(8):
            nb.update_insight_bayesian(iid, success=True)
        for _ in range(2):
            nb.update_insight_bayesian(iid, success=False)

        row = nb.conn.execute(
            "SELECT alpha, beta_, confidence, n_predictions, n_correct FROM insights WHERE insight_id = ?",
            (iid,),
        ).fetchone()
        alpha = float(row["alpha"])
        beta_ = float(row["beta_"])
        confidence = alpha / (alpha + beta_)
        assert 0.65 < confidence < 0.85, f"Expected ~0.75, got {confidence:.3f}"
        assert int(row["n_predictions"]) == 10
        assert int(row["n_correct"]) == 8


class TestDisplayOnly:
    def test_failure_mode_always_display_only(self, nb):
        """display_only=1 is forced for failure_mode category."""
        iid = nb.record_insight(
            category="failure_mode",
            content="some failure",
            semantic_key="test:failure_display",
            display_only=False,  # Explicitly try to set False
            evidence_json={"test": "manual"},
        )
        row = nb.conn.execute(
            "SELECT display_only FROM insights WHERE insight_id = ?",
            (iid,),
        ).fetchone()
        assert int(row["display_only"]) == 1

    def test_exclude_display_only_filter(self, nb):
        """display_only=1 insights excluded when exclude_display_only=True."""
        nb.record_insight(
            category="failure_mode",
            content="hidden failure",
            semantic_key="test:hidden",
            evidence_json={"test": "manual"},
        )
        nb.record_insight(
            category="success_factor",
            content="visible success",
            semantic_key="test:visible",
            alpha=5.0,
            beta_=1.0,
            evidence_json={"test": "manual"},
        )
        all_insights = nb.get_insights(limit=100)
        filtered = nb.get_insights(limit=100, exclude_display_only=True)
        assert len(all_insights) == 2
        assert len(filtered) == 1
        assert filtered[0]["semantic_key"] == "test:visible"


class TestSeedIdempotency:
    def test_seed_twice_no_duplicates(self, nb):
        """Running seed logic twice doesn't create duplicate active insights."""
        for _ in range(2):
            nb.record_insight(
                category="structural_preference",
                content="7-9 ops optimal",
                semantic_key="structural:graph_size_optimal",
                alpha=940.0,
                beta_=1524.0,
                insight_level="structural",
                evidence_json={"test": "chi2", "n": 2464},
            )
        active = nb.get_insights(status="active", limit=100)
        matching = [
            i for i in active if i["semantic_key"] == "structural:graph_size_optimal"
        ]
        assert len(matching) == 1, f"Expected 1 active, got {len(matching)}"


class TestCodeFailureClassification:
    def test_code_failure_display_only(self, nb):
        """RuntimeError/TypeError failures get display_only=1 via failure_mode category."""
        iid = nb.record_insight(
            category="failure_mode",
            content="RuntimeError in compilation",
            semantic_key="test:runtime_error",
            evidence_json={"test": "frequency_count", "is_code_failure": True},
        )
        row = nb.conn.execute(
            "SELECT display_only FROM insights WHERE insight_id = ?",
            (iid,),
        ).fetchone()
        assert int(row["display_only"]) == 1


class TestGrammarIntegration:
    def test_structural_insight_reduces_max_ops(self, nb):
        """High-confidence structural insight reduces max_ops."""
        from research.synthesis.grammar import GrammarConfig
        from research.scientist.runner.execution_screening import (
            _apply_insight_adjustments,
        )

        nb.record_insight(
            category="structural_preference",
            content="13+ ops collapses",
            semantic_key="structural:graph_size_cap",
            subject_key="graph_size_cap",
            alpha=100.0,
            beta_=2.0,
            insight_level="structural",
            evidence_json={"test": "binomial", "recommended_max": 12},
        )
        grammar = GrammarConfig(max_ops=16)
        tw = dict(grammar.template_weights)
        mw = dict(grammar.motif_weights)
        _apply_insight_adjustments(nb, grammar, tw, mw)
        assert grammar.max_ops <= 12, f"Expected max_ops<=12, got {grammar.max_ops}"

    def test_low_confidence_insight_ignored(self, nb):
        """Insight with confidence < 0.6 doesn't affect grammar."""
        from research.synthesis.grammar import GrammarConfig
        from research.scientist.runner.execution_screening import (
            _apply_insight_adjustments,
        )

        nb.record_insight(
            category="structural_preference",
            content="Weak signal",
            semantic_key="structural:weak",
            subject_key="graph_size_cap",
            alpha=1.0,
            beta_=3.0,  # confidence = 0.25
            insight_level="structural",
            evidence_json={"test": "binomial", "recommended_max": 8},
        )
        grammar = GrammarConfig(max_ops=16)
        tw = dict(grammar.template_weights)
        mw = dict(grammar.motif_weights)
        _apply_insight_adjustments(nb, grammar, tw, mw)
        assert grammar.max_ops == 16, "Low-confidence insight should not change max_ops"


class TestConfidenceFiltering:
    def test_low_confidence_not_in_scoring(self, nb):
        """Insights with confidence <= 0.55 excluded from selection matching."""
        nb.record_insight(
            category="success_factor",
            content="weak op signal",
            semantic_key="test:weak_selection",
            alpha=1.0,
            beta_=1.0,  # confidence = 0.5
            insight_level="composition",
            evidence_json={"test": "manual"},
        )
        insights = nb.get_insights(limit=100, exclude_display_only=True)
        filtered = [
            i
            for i in insights
            if (
                float(i.get("alpha") or 1)
                / (float(i.get("alpha") or 1) + float(i.get("beta_") or 1))
            )
            > 0.55
        ]
        assert len(filtered) == 0


class TestEvidenceJson:
    def test_evidence_json_stored_and_parsed(self, nb):
        """evidence_json is stored as JSON text and parsed on retrieval."""
        evidence = {"test": "fisher_exact", "p_value": 0.001, "n": 500}
        iid = nb.record_insight(
            category="pattern",
            content="test evidence",
            semantic_key="test:evidence",
            evidence_json=evidence,
        )
        insights = nb.get_insights(limit=1)
        assert len(insights) == 1
        retrieved = insights[0].get("evidence_json")
        assert isinstance(retrieved, dict), f"Expected dict, got {type(retrieved)}"
        assert retrieved["test"] == "fisher_exact"
        assert retrieved["p_value"] == 0.001


class TestInsightLevelFiltering:
    def test_filter_by_level(self, nb):
        """get_insights(insight_level=...) filters correctly."""
        nb.record_insight(
            category="pattern",
            content="structural",
            semantic_key="test:s1",
            insight_level="structural",
            evidence_json={"test": "x"},
        )
        nb.record_insight(
            category="pattern",
            content="composition",
            semantic_key="test:c1",
            insight_level="composition",
            evidence_json={"test": "x"},
        )
        structural = nb.get_insights(insight_level="structural", limit=100)
        composition = nb.get_insights(insight_level="composition", limit=100)
        assert len(structural) == 1
        assert structural[0]["insight_level"] == "structural"
        assert len(composition) == 1
        assert composition[0]["insight_level"] == "composition"
