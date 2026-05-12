from __future__ import annotations

import json
import os
import random
import sqlite3
from pathlib import Path

from research.scientist.runner._types import RunConfig
from research.synthesis._template_helpers import sample_routing_choice
from research.synthesis.generation_runtime import runtime_context_for_config
from research.synthesis.grammar import GrammarConfig
from research.synthesis.graph import ComputationGraph
from research.synthesis.routing_decision_priors import (
    build_routing_decision_prior_index,
    load_routing_decision_priors,
    routing_decision_prior_for,
    routing_decision_prior_weight,
)
from research.tools.build_routing_decision_priors import (
    build_routing_decision_prior,
    write_routing_decision_prior,
)


_COLUMNS = (
    "stage1_passed",
    "loss_ratio",
    "validation_loss_ratio",
    "ar_gate_score",
    "binding_intermediate_auc",
    "binding_screening_auc",
    "binding_screening_composite",
    "routing_utilization_entropy",
    "routing_drop_rate",
    "routing_savings_ratio",
    "routing_collapse_score",
)


def _create_runs_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE program_results (
                result_id TEXT PRIMARY KEY,
                graph_json TEXT,
                stage1_passed INTEGER,
                loss_ratio REAL,
                validation_loss_ratio REAL,
                ar_gate_score REAL,
                binding_intermediate_auc REAL,
                binding_screening_auc REAL,
                binding_screening_composite REAL,
                routing_utilization_entropy REAL,
                routing_drop_rate REAL,
                routing_savings_ratio REAL,
                routing_collapse_score REAL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _insert_result(
    path: Path,
    result_id: str,
    *,
    template: str = "unit_router",
    decision: str = "gate_threshold",
    value: object = 0.5,
    **outcomes: object,
) -> None:
    graph = {
        "metadata": {
            "routing_decisions": [
                {
                    "template_name": template,
                    "decision_key": decision,
                    "value": value,
                    "source": "test_fixture",
                }
            ]
        }
    }
    row = [outcomes.get(column) for column in _COLUMNS]
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            INSERT INTO program_results (
                result_id,
                graph_json,
                stage1_passed,
                loss_ratio,
                validation_loss_ratio,
                ar_gate_score,
                binding_intermediate_auc,
                binding_screening_auc,
                binding_screening_composite,
                routing_utilization_entropy,
                routing_drop_rate,
                routing_savings_ratio,
                routing_collapse_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (result_id, json.dumps(graph), *row),
        )
        conn.commit()
    finally:
        conn.close()


def test_routing_prior_shrinks_low_support_toward_neutral(tmp_path: Path) -> None:
    runs = tmp_path / "runs.db"
    _create_runs_db(runs)
    _insert_result(
        runs,
        "excellent_once",
        value="rare_good",
        stage1_passed=1,
        loss_ratio=0.2,
        validation_loss_ratio=0.2,
        ar_gate_score=0.9,
        binding_intermediate_auc=0.9,
        routing_drop_rate=0.0,
        routing_collapse_score=0.0,
    )
    for i in range(20):
        _insert_result(
            runs,
            f"baseline_{i}",
            value="baseline",
            stage1_passed=0,
            loss_ratio=0.8,
            validation_loss_ratio=0.8,
            ar_gate_score=0.2,
            binding_intermediate_auc=0.2,
            routing_drop_rate=0.1,
            routing_collapse_score=0.0,
        )

    prior = build_routing_decision_prior(runs, min_support=10, created_at=1.0)
    rare = routing_decision_prior_for(
        prior, "unit_router", "gate_threshold", "rare_good"
    )

    assert rare is not None
    assert rare["n"] == 1
    assert rare["support_confidence"] < 0.2
    assert 1.0 < rare["advisory_weight"] < 1.3


def test_routing_prior_penalizes_drop_and_collapse(tmp_path: Path) -> None:
    runs = tmp_path / "runs.db"
    _create_runs_db(runs)
    for i in range(8):
        _insert_result(
            runs,
            f"stable_{i}",
            value="stable",
            stage1_passed=1,
            loss_ratio=0.5,
            validation_loss_ratio=0.5,
            routing_drop_rate=0.0,
            routing_collapse_score=0.0,
        )
        _insert_result(
            runs,
            f"collapsed_{i}",
            value="collapsed",
            stage1_passed=1,
            loss_ratio=0.5,
            validation_loss_ratio=0.5,
            routing_drop_rate=0.95,
            routing_collapse_score=0.9,
        )

    prior = build_routing_decision_prior(runs, min_support=8, created_at=1.0)
    stable = routing_decision_prior_for(
        prior, "unit_router", "gate_threshold", "stable"
    )
    collapsed = routing_decision_prior_for(
        prior, "unit_router", "gate_threshold", "collapsed"
    )

    assert stable is not None
    assert collapsed is not None
    assert collapsed["contributions"]["routing_drop_penalty"] < 0
    assert collapsed["contributions"]["routing_collapse_penalty"] < 0
    assert collapsed["advisory_weight"] < stable["advisory_weight"]
    assert collapsed["advisory_weight"] < 1.0


def test_missing_ar_binding_data_does_not_zero_other_signal(tmp_path: Path) -> None:
    runs = tmp_path / "runs.db"
    _create_runs_db(runs)
    for i in range(10):
        _insert_result(
            runs,
            f"good_{i}",
            value="good",
            stage1_passed=1,
            loss_ratio=0.4,
            validation_loss_ratio=0.4,
        )
        _insert_result(
            runs,
            f"bad_{i}",
            value="bad",
            stage1_passed=0,
            loss_ratio=0.8,
            validation_loss_ratio=0.8,
        )

    prior = build_routing_decision_prior(runs, min_support=4, created_at=1.0)
    good = routing_decision_prior_for(prior, "unit_router", "gate_threshold", "good")

    assert good is not None
    assert good["metrics"]["n_ar_gate_score"] == 0
    assert good["metrics"]["n_binding_intermediate_auc"] == 0
    assert good["contributions"]["ar_lift"] == 0.0
    assert good["contributions"]["binding_lift"] == 0.0
    assert good["advisory_weight"] > 1.0


def test_routing_prior_write_and_loader_lookup(tmp_path: Path) -> None:
    runs = tmp_path / "runs.db"
    _create_runs_db(runs)
    for i in range(4):
        _insert_result(
            runs,
            f"good_{i}",
            value={"lanes": 2},
            stage1_passed=1,
            loss_ratio=0.4,
            validation_loss_ratio=0.4,
        )
        _insert_result(
            runs,
            f"bad_{i}",
            value={"lanes": 4},
            stage1_passed=0,
            loss_ratio=0.8,
            validation_loss_ratio=0.8,
        )

    prior = build_routing_decision_prior(runs, min_support=2, created_at=1.0)
    artifact = write_routing_decision_prior(prior, output_dir=tmp_path / "priors")
    loaded = load_routing_decision_priors(artifact.parent)

    assert artifact.exists()
    assert (artifact.parent / "latest.json").exists()
    assert loaded["loaded"] is True
    assert (
        routing_decision_prior_weight(
            loaded, "unit_router", "gate_threshold", {"lanes": 2}
        )
        > 1.0
    )


def test_routing_prior_loader_fails_closed(tmp_path: Path) -> None:
    missing = load_routing_decision_priors(tmp_path / "missing.json")
    assert missing["loaded"] is False
    assert missing["priors"] == {}
    assert (
        routing_decision_prior_weight(missing, "unit_router", "gate_threshold", 0.5)
        == 1.0
    )

    broken_path = tmp_path / "broken.json"
    broken_path.write_text("{not-json", encoding="utf-8")
    broken = load_routing_decision_priors(broken_path)
    assert broken["loaded"] is False
    assert broken["records"] == []

    stale_path = tmp_path / "stale.json"
    stale_path.write_text(
        json.dumps(
            {
                "schema_version": "routing_decision_prior_v1",
                "created_at": 1.0,
                "records": [],
                "priors": {},
            }
        ),
        encoding="utf-8",
    )
    stale = load_routing_decision_priors(stale_path, max_age_seconds=10.0, now=100.0)
    assert stale["loaded"] is False
    assert stale["load_reason"] == "stale"


def test_sample_routing_choice_preserves_rng_choice_when_prior_absent() -> None:
    graph = ComputationGraph(64)
    choices = ["low", "high"]
    seed = 17

    selected = sample_routing_choice(
        random.Random(seed),
        choices,
        graph=graph,
        template_name="unit_router",
        decision_key="gate_threshold",
    )

    assert selected == random.Random(seed).choice(choices)
    decision = graph.metadata["routing_decisions"][0]
    assert decision["source"] == "rng_choice"
    assert "prior" not in decision


def test_sample_routing_choice_uses_prior_and_records_attribution() -> None:
    prior = {
        "schema_version": "routing_decision_prior_v1",
        "version": "unit_prior",
        "created_at": 1.0,
        "loaded": True,
        "records": [
            {
                "template_name": "unit_router",
                "decision_key": "gate_threshold",
                "value": "low",
                "advisory_weight": 0.25,
                "n": 12,
            },
            {
                "template_name": "unit_router",
                "decision_key": "gate_threshold",
                "value": "high",
                "advisory_weight": 3.0,
                "n": 12,
            },
        ],
    }
    prior["priors"] = build_routing_decision_prior_index(prior["records"])
    graph = ComputationGraph(64)
    graph._routing_decision_prior_state = {"prior": prior, "strength": 1.0}

    selected = sample_routing_choice(
        random.Random(1),
        ["low", "high"],
        graph=graph,
        template_name="unit_router",
        decision_key="gate_threshold",
    )

    assert selected == "high"
    decision = graph.metadata["routing_decisions"][0]
    assert decision["source"] == "routing_prior_weighted"
    assert decision["prior"]["version"] == "unit_prior"
    assert decision["prior"]["matched_choices"] == 2
    assert decision["prior"]["selected_probability"] > 0.9


def test_routing_prior_config_fields_round_trip() -> None:
    config = RunConfig(
        use_routing_decision_priors=True,
        routing_decision_prior_path="/tmp/routing-priors.json",
        routing_decision_prior_strength=0.5,
    )

    reconstructed = RunConfig.from_dict(config.to_dict())

    assert reconstructed.use_routing_decision_priors is True
    assert reconstructed.routing_decision_prior_path == "/tmp/routing-priors.json"
    assert reconstructed.routing_decision_prior_strength == 0.5


def test_runtime_context_reloads_routing_prior_when_artifact_changes(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "latest.json"

    def write_prior(version: str, mtime_ns: int) -> None:
        artifact.write_text(
            json.dumps(
                {
                    "schema_version": "routing_decision_prior_v1",
                    "version": version,
                    "created_at": 1.0,
                    "records": [],
                    "priors": {},
                }
            ),
            encoding="utf-8",
        )
        os.utime(artifact, ns=(mtime_ns, mtime_ns))

    config = GrammarConfig(
        use_routing_decision_priors=True,
        routing_decision_prior_path=str(artifact),
    )

    write_prior("v1", 1_000_000_000)
    first = runtime_context_for_config(config)
    assert first.routing_decision_priors["version"] == "v1"

    write_prior("v2", 2_000_000_000)
    second = runtime_context_for_config(config)
    assert second.routing_decision_priors["version"] == "v2"
