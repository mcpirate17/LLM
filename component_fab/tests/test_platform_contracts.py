"""Fast platform contracts for component_fab.

These tests keep the research-platform scaffolding honest without running the
slow TinyLM or Wikitext-style probes.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from component_fab.state.ledger import Ledger, PROMOTION_PENDING, iter_jsonl_records
from component_fab.state.provenance import build_run_provenance
from component_fab.state.schema_versions import SCHEMA_VERSIONS
from component_fab.tests.conftest import make_spec
from component_fab.validator.grade import grade_candidate


_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_schema_versions_are_named_and_stable() -> None:
    assert SCHEMA_VERSIONS["ledger_grade"] == "component_fab.ledger.grade.v2"
    assert SCHEMA_VERSIONS["run_report"] == "component_fab.run_report.v1"


def test_policy_config_snapshots_load() -> None:
    expected = {
        "quality_v1.yml",
        "measured_screen_v1.yml",
        "invention_promotion_v1.yml",
    }
    found = {path.name for path in _CONFIG_DIR.glob("*.yml")}
    assert expected <= found
    for name in expected:
        payload = yaml.safe_load((_CONFIG_DIR / name).read_text(encoding="utf-8"))
        assert payload["version"] == name.removesuffix(".yml")


def test_run_provenance_has_replay_fields() -> None:
    payload = build_run_provenance(["--cycles", "1"], config_versions={"quality": "quality_v1"})
    assert payload["run_id"]
    assert payload["argv"] == ["--cycles", "1"]
    assert payload["schema_versions"]["proposal_spec"] == "component_fab.proposal_spec.v1"
    assert payload["config_versions"] == {"quality": "quality_v1"}


def test_ledger_write_replay_contract(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = Ledger(ledger_path)
    ledger.record_grade(
        "contract",
        name="contract",
        category="lane",
        synthesis_kind="semiring_swap",
        cycle=1,
        composite_score=0.25,
        smoke_pass=True,
        learned_signal=False,
        metadata={"math_axes": {"op_algebraic_space": "tropical"}},
    )
    ledger.close()

    replay = Ledger(ledger_path)
    entry = replay.entries["contract"]
    assert entry.promotion_status == PROMOTION_PENDING
    assert entry.best_composite() == 0.25
    records = list(iter_jsonl_records(ledger_path))
    assert records[0]["event"] == "grade"


def test_grade_candidate_fast_contract() -> None:
    spec = make_spec({"op_algebraic_space": "tropical"}, pid="contract_tropical")
    bundle = grade_candidate(
        spec,
        dim=8,
        seq_len=8,
        n_steps=1,
        run_range_probe=False,
        run_in_context=False,
    )
    assert bundle.capability
    assert bundle.eliminated_by is None
    assert bundle.solo is not None
    assert bundle.solo.smoke["forward_passed"]
    assert bundle.solo.smoke["backward_passed"]


def test_json_report_payload_can_embed_provenance() -> None:
    payload = {
        "run_metadata": build_run_provenance(["--dry-run"]),
        "results": [],
    }
    encoded = json.dumps(payload)
    assert "run_metadata" in encoded
    assert "schema_versions" in encoded
