import json
import os
import tempfile

import pytest

from research.scientist.evidence import (
    EvidencePackError,
    build_evidence_pack,
    validate_evidence_pack,
    validate_learning_log_entry,
)
from research.scientist.notebook import LabNotebook
from research.scientist.runner import ExperimentRunner, RunConfig

pytestmark = pytest.mark.unit


def _make_notebook_with_novelty():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "evidence_pack.db")
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("synthesis", config={})
    fingerprint = {"similarity_path": "_compute_reference_cka"}
    nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp123",
        graph_json="{}",
        stage1_passed=True,
        novelty_score=0.7,
        novelty_confidence=0.8,
        cka_source="artifact",
        cka_artifact_version="v1",
        fingerprint_json=json.dumps(fingerprint),
    )
    nb.complete_experiment(
        exp_id,
        results={
            "total": 1,
            "stage0_passed": 1,
            "stage05_passed": 1,
            "stage1_passed": 1,
            "best_loss_ratio": 0.1,
            "best_novelty_score": 0.7,
        },
    )
    return nb


def test_validate_evidence_pack_rejects_missing_novelty_reference():
    pack = {
        "hypothesis": "test",
        "supporting_metrics": [
            {"name": "best_novelty_score", "value": 0.5, "baseline": 0.4, "delta_vs_baseline": 0.1}
        ],
        "uncertainty": {},
        "confounders": [],
        "falsification": [],
    }
    with pytest.raises(EvidencePackError):
        validate_evidence_pack(pack)


def test_build_evidence_pack_includes_reference_and_metrics():
    nb = _make_notebook_with_novelty()
    nb.flush_writes()
    pack = build_evidence_pack(nb)
    validate_evidence_pack(pack)
    assert "supporting_metrics" in pack
    assert pack["supporting_metrics"]
    assert pack.get("novelty_reference") is not None
    nb.close()


def test_log_grammar_weight_application_includes_audit_query():
    tmpdir = tempfile.mkdtemp()
    runner = ExperimentRunner(os.path.join(tmpdir, "runner.db"))

    class _FakeNB:
        def __init__(self):
            self.events = []

        def log_learning_event(self, event_type, description, **kwargs):
            self.events.append({
                "event_type": event_type,
                "evidence": kwargs.get("evidence"),
            })

    class _FakeAnalytics:
        def grammar_weight_audit_info(self):
            return {"query": "SELECT * FROM program_results", "params": []}

    nb = _FakeNB()
    runner._log_grammar_weight_application(
        nb,
        exp_id="exp123",
        old_weights={"a": 1.0},
        new_weights={"a": 1.2},
        analytics=_FakeAnalytics(),
    )
    assert nb.events, "Expected a grammar_weights_applied log event."
    validate_learning_log_entry(nb.events[0])


def test_mode_selection_entry_includes_evidence_pack():
    nb = _make_notebook_with_novelty()
    runner = ExperimentRunner(nb.db_path)

    def _fake_analytics(_nb):
        return {
            "compression_coverage": {"totals": {"n_tested": 10, "n_survived": 2}},
        }

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(runner, "_gather_analytics_data", _fake_analytics)
        mp.setattr(runner.aria, "recommend_next_mode", lambda **_kw: {
            "mode": "synthesis",
            "reasoning": "test",
            "confidence": 0.7,
            "config": {},
        })

        rec = runner._select_next_mode(RunConfig(device="cpu"), nb, n_experiments=1)

    assert rec.get("evidence_pack") is not None
    validate_evidence_pack(rec["evidence_pack"])
    row = nb.conn.execute(
        "SELECT metadata_json FROM entries WHERE entry_type='decision' ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    metadata = json.loads(row["metadata_json"])
    assert metadata.get("evidence_pack") is not None
    nb.close()
