"""CLI smoke tests for ``component_fab.tools.run_lm_probe``.

Build a tiny synthetic ledger with one promoted entry, run the CLI
against it at a 2-step budget, and check the JSON report is produced
with the expected schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from component_fab.improver.axis_variants import (
    DEFAULT_AXIS_VARIANT_TEMPLATES,
    anchor_axes_for_op,
    spec_for_variant,
)
from component_fab.proposer.property_miner import DEFAULT_META_DB
from component_fab.state.ledger import (
    PROMOTION_PROMOTED,
    Ledger,
)


def _seeded_ledger(tmp_path: Path) -> tuple[Path, str]:
    """Build a ledger with one promoted spec we know how to reconstruct."""
    if not DEFAULT_META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    anchor = anchor_axes_for_op("tropical_attention")
    if anchor is None:
        pytest.skip("tropical_attention not in op_property_catalog")
    variant = next(
        v for v in DEFAULT_AXIS_VARIANT_TEMPLATES if v.delta_name == "fourier_basis"
    )
    spec = spec_for_variant(anchor, variant)

    ledger_path = tmp_path / "ledger.jsonl"
    ledger = Ledger(ledger_path)
    ledger.record_grade(
        proposal_id=spec.proposal_id,
        name=spec.name,
        category=spec.category,
        synthesis_kind=spec.synthesis_kind,
        cycle=1,
        composite_score=0.75,
        smoke_pass=True,
        learned_signal=True,
        metadata={"math_knobs": []},
    )
    ledger.record_promotion(spec.proposal_id, PROMOTION_PROMOTED)
    return ledger_path, spec.proposal_id


def test_run_lm_probe_cli_resolves_proposal_id_and_writes_report(
    tmp_path: Path,
) -> None:
    from component_fab.tools.run_lm_probe import main

    ledger_path, pid = _seeded_ledger(tmp_path)
    out_path = tmp_path / "report.json"
    exit_code = main(
        [
            "--proposal-id",
            pid,
            "--ledger",
            str(ledger_path),
            "--n-train-steps",
            "2",
            "--batch-size",
            "2",
            "--dim",
            "16",
            "--n-blocks",
            "1",
            "--baseline-names",
            "softmax_attention",
            "--output",
            str(out_path),
        ]
    )
    assert exit_code == 0
    assert out_path.exists()
    payloads = json.loads(out_path.read_text())
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["candidate"]
    assert "multi_query_kv_recall" in payload["tasks"]
    rows = payload["tasks"]["multi_query_kv_recall"]
    # candidate + 1 baseline
    assert len(rows) == 2
    labels = {r["mixer_label"] for r in rows}
    assert "softmax_attention" in labels


def test_run_lm_probe_cli_unknown_proposal_returns_error(tmp_path: Path) -> None:
    from component_fab.tools.run_lm_probe import main

    ledger_path = tmp_path / "empty.jsonl"
    ledger_path.write_text("", encoding="utf-8")
    exit_code = main(
        [
            "--proposal-id",
            "does_not_exist_xyz",
            "--ledger",
            str(ledger_path),
            "--n-train-steps",
            "2",
        ]
    )
    assert exit_code == 2


def test_run_lm_probe_cli_top_n_unique_empty_ledger(tmp_path: Path) -> None:
    """Empty ledger -> top-n-unique returns 0 promoted -> exit 2."""
    from component_fab.tools.run_lm_probe import main

    ledger_path = tmp_path / "empty.jsonl"
    ledger_path.write_text("", encoding="utf-8")
    exit_code = main(
        [
            "--top-n-unique",
            "5",
            "--ledger",
            str(ledger_path),
            "--n-train-steps",
            "2",
        ]
    )
    assert exit_code == 2
