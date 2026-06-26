"""Fast platform contracts for component_fab."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from component_fab.generator.code_generator import generate_module
from component_fab.generator.dispatch import UnknownBlockSlotError
from component_fab.state.ledger import Ledger, iter_jsonl_records
from component_fab.state.provenance import build_run_provenance
from component_fab.state.schema_versions import LEDGER_GRADE_SCHEMA_VERSION, SCHEMA_VERSIONS
from component_fab.tests.conftest import make_spec
from component_fab.tools._cli import write_report
from component_fab.validator.grade import grade_candidate

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_schema_configs_and_provenance_contract(tmp_path: Path) -> None:
    assert SCHEMA_VERSIONS["ledger_grade"] == "component_fab.ledger.grade.v2"
    assert SCHEMA_VERSIONS["run_report"] == "component_fab.run_report.v1"
    for name in ("quality_v1.yml", "measured_screen_v1.yml", "invention_promotion_v1.yml"):
        payload = yaml.safe_load((_CONFIG_DIR / name).read_text(encoding="utf-8"))
        assert payload["version"] == name.removesuffix(".yml")
    provenance = build_run_provenance(["--cycles", "1"])
    assert provenance["argv"] == ["--cycles", "1"]
    out = write_report({"results": []}, default_dir=tmp_path, prefix="contract", quiet=True)
    assert out is not None
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema_version"] == "component_fab.run_report.v1"
    assert report["run_metadata"]["schema_versions"]["proposal_spec"]


def test_ledger_schema_write_and_replay(tmp_path: Path) -> None:
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
    records = list(iter_jsonl_records(ledger_path))
    assert records[0]["schema_version"] == LEDGER_GRADE_SCHEMA_VERSION
    assert Ledger(ledger_path).entries["contract"].best_composite() == 0.25


def test_fast_grade_and_fail_loud_slot_contract() -> None:
    spec = make_spec({"op_algebraic_space": "tropical"}, pid="contract_tropical")
    bundle = grade_candidate(
        spec,
        dim=8,
        seq_len=16,
        n_steps=1,
        run_range_probe=False,
        run_in_context=False,
    )
    assert bundle.solo is not None
    assert bundle.solo.smoke["forward_passed"]
    with pytest.raises(UnknownBlockSlotError):
        generate_module(
            {
                "op_block_template": "gated_parallel",
                "op_algebraic_space": "tropical",
                "op_block_slot_b": "missing_slot",
            },
            dim=8,
        )
