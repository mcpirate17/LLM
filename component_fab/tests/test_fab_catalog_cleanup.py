"""Cleanup policy tests for component_fab catalog artifacts."""

from __future__ import annotations

import os
import json
from pathlib import Path

from component_fab.proposer.proposal_catalog import load_proposals_by_id
from component_fab.state.ledger import (
    Ledger,
    _prune_rotations,
    iter_rotated_jsonl_paths,
)
from component_fab.tools.run_autonomous import (
    _parse_args,
    _prune_autonomous_run_summaries,
    _rotate_proposals,
)


def _touch_with_mtime(path: Path, mtime: int) -> None:
    path.write_text("x\n", encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_prune_rotations_keeps_three_newest_integer_suffixes(tmp_path: Path) -> None:
    base = tmp_path / "ledger.jsonl"
    base.write_text("", encoding="utf-8")
    for index in range(1, 6):
        _touch_with_mtime(tmp_path / f"ledger.jsonl.{index}", index)
    _touch_with_mtime(tmp_path / "ledger.jsonl.not-a-rotation", 100)

    assert _prune_rotations(base) == 2

    assert sorted(path.name for path in tmp_path.glob("ledger.jsonl.*")) == [
        "ledger.jsonl.3",
        "ledger.jsonl.4",
        "ledger.jsonl.5",
        "ledger.jsonl.not-a-rotation",
    ]


def test_iter_rotated_jsonl_paths_uses_mtime_order_then_active(tmp_path: Path) -> None:
    base = tmp_path / "ledger.jsonl"
    base.write_text("", encoding="utf-8")
    _touch_with_mtime(tmp_path / "ledger.jsonl.2", 10)
    _touch_with_mtime(tmp_path / "ledger.jsonl.1", 20)
    _touch_with_mtime(tmp_path / "ledger.jsonl.not-a-rotation", 1)

    paths = [path.name for path in iter_rotated_jsonl_paths(base)]

    assert paths == ["ledger.jsonl.2", "ledger.jsonl.1", "ledger.jsonl"]


def test_load_proposals_by_id_replays_rotations_by_mtime(tmp_path: Path) -> None:
    base = tmp_path / "proposals.jsonl"

    def row(name: str) -> str:
        return json.dumps({"proposal_id": "p", "name": name}) + "\n"

    (tmp_path / "proposals.jsonl.2").write_text(row("old"), encoding="utf-8")
    os.utime(tmp_path / "proposals.jsonl.2", (10, 10))
    (tmp_path / "proposals.jsonl.1").write_text(row("new"), encoding="utf-8")
    os.utime(tmp_path / "proposals.jsonl.1", (20, 20))
    base.write_text("", encoding="utf-8")

    loaded = load_proposals_by_id(base)

    assert loaded["p"].name == "new"


def test_ledger_rotation_prunes_old_rotations(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    for index in range(1, 6):
        _touch_with_mtime(tmp_path / f"ledger.jsonl.{index}", index)
    for i in range(20):
        ledger.record_grade(
            proposal_id=f"p{i}",
            name=f"p{i}",
            category="lane",
            synthesis_kind="x",
            cycle=1,
            composite_score=0.5,
            smoke_pass=True,
            learned_signal=False,
        )

    rotated = ledger.rotate_if_oversized(max_bytes=1)

    assert rotated is not None
    assert len(list(tmp_path.glob("ledger.jsonl.[0-9]*"))) == 3
    assert rotated.exists()


def test_rotate_proposals_prunes_old_rotations(tmp_path: Path) -> None:
    proposals_path = tmp_path / "proposals.jsonl"
    proposals_path.write_text("x" * 100, encoding="utf-8")
    for index in range(1, 6):
        _touch_with_mtime(tmp_path / f"proposals.jsonl.{index}", index)

    _rotate_proposals(proposals_path, rotate_bytes=1, quiet=True)

    assert proposals_path.exists()
    assert len(list(tmp_path.glob("proposals.jsonl.[0-9]*"))) == 3
    assert (tmp_path / "proposals.jsonl.6").exists()
    assert not (tmp_path / "proposals.jsonl.1").exists()


def test_autonomous_run_summary_emission_is_opt_in() -> None:
    default_args = _parse_args([])
    enabled_args = _parse_args(["--emit-run-summary"])

    assert default_args.emit_run_summary is False
    assert enabled_args.emit_run_summary is True


def test_prune_autonomous_run_summaries_keeps_three_newest(tmp_path: Path) -> None:
    for index in range(1, 6):
        _touch_with_mtime(
            tmp_path / f"autonomous_run_20260517_00000{index}.json", index
        )

    assert _prune_autonomous_run_summaries(tmp_path) == 2

    assert sorted(path.name for path in tmp_path.glob("autonomous_run_*.json")) == [
        "autonomous_run_20260517_000003.json",
        "autonomous_run_20260517_000004.json",
        "autonomous_run_20260517_000005.json",
    ]
