"""Tests for the dry-run promotion simulator.

We mock ``research.scientist.leaderboard_scoring`` so the simulator can
be exercised without pulling the full scoring stack into a unit-test
run. The deliverable under test is: does the simulator correctly join
current vs proposed scores, compute the promotion delta, and refuse to
write to the underlying DB?
"""

from __future__ import annotations

import sqlite3
import sys
import types
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE leaderboard (
            result_id TEXT PRIMARY KEY,
            tier TEXT,
            composite_score REAL,
            is_reference INTEGER,
            reference_name TEXT,
            template_name TEXT
        );
        CREATE TABLE program_results (
            result_id TEXT PRIMARY KEY,
            wikitext_perplexity REAL
        );
        """
    )
    rows = [
        ("attn1", "screening", 480.0, 1, "gpt2", "softmax_attention_block", 90.0),
        ("attn2", "screening", 460.0, 0, None, "softmax_attention_block", 100.0),
        ("ssm1", "screening", 430.0, 1, "mamba", "mamba_block_v2", 200.0),
        ("ssm2", "screening", 410.0, 0, None, "mamba_block_v2", 220.0),
    ]
    for rid, tier, comp, is_ref, ref_name, tmpl, ppl in rows:
        conn.execute(
            "INSERT INTO leaderboard VALUES (?, ?, ?, ?, ?, ?)",
            (rid, tier, comp, is_ref, ref_name, tmpl),
        )
        conn.execute("INSERT INTO program_results VALUES (?, ?)", (rid, ppl))
    conn.commit()
    conn.close()


def _stub_leaderboard_scoring(monkeypatch, *, score_map: dict[str, dict[str, float]]):
    """Install a fake research.scientist.leaderboard_scoring module.

    ``score_map`` maps result_id → {"current": float, "proposed": float}.
    The fake module reads back the active YAML to decide which side to
    return: when the YAML contains an "_simulator_marker: proposed" key
    we return the proposed score; otherwise the current score.
    """

    fake_lb = types.ModuleType("research.scientist.leaderboard_scoring")

    def prefetch_program_results(conn, result_ids):
        rows = conn.execute(
            f"SELECT result_id, wikitext_perplexity FROM program_results "
            f"WHERE result_id IN ({','.join('?' for _ in result_ids)})",
            list(result_ids),
        ).fetchall()
        return {r["result_id"]: dict(r) for r in rows}

    def build_score_kwargs_from_prefetch(lb_row, pr_row):
        return {"result_id": lb_row["result_id"]}

    def compute_composite(**kwargs):
        # Read the marker via the patched config module. The marker lives
        # inside ``base`` so it survives _merge_proposal.
        from research.scientist import scoring_config as _scfg

        base = (_scfg._PAYLOAD or {}).get("base") or {}
        side = "proposed" if base.get("_simulator_marker") == "proposed" else "current"
        rid = kwargs.get("result_id")
        return score_map[rid][side]

    fake_lb.prefetch_program_results = prefetch_program_results
    fake_lb.build_score_kwargs_from_prefetch = build_score_kwargs_from_prefetch
    fake_lb.compute_composite = compute_composite
    monkeypatch.setitem(sys.modules, "research.scientist.leaderboard_scoring", fake_lb)
    scientist_pkg = sys.modules.get("research.scientist")
    if scientist_pkg is not None:
        monkeypatch.setattr(scientist_pkg, "leaderboard_scoring", fake_lb, raising=False)


def _stub_scoring_config(monkeypatch, *, tmp_yaml: Path):
    """Provide a minimal stand-in for ``research.scientist.scoring_config``."""

    fake_scfg = types.ModuleType("research.scientist.scoring_config")
    fake_scfg._CONFIG_PATH = tmp_yaml
    fake_scfg._PAYLOAD = {}

    def reload_scoring_config():
        if fake_scfg._CONFIG_PATH.exists():
            fake_scfg._PAYLOAD = (
                yaml.safe_load(fake_scfg._CONFIG_PATH.read_text(encoding="utf-8")) or {}
            )
        else:
            fake_scfg._PAYLOAD = {}
        return ""

    fake_scfg.reload_scoring_config = reload_scoring_config
    monkeypatch.setitem(sys.modules, "research.scientist.scoring_config", fake_scfg)
    scientist_pkg = sys.modules.get("research.scientist")
    if scientist_pkg is not None:
        monkeypatch.setattr(scientist_pkg, "scoring_config", fake_scfg, raising=False)


def test_promotion_simulator_dry_run(tmp_path: Path, monkeypatch):
    db = tmp_path / "runs.db"
    scoring_yaml = tmp_path / "scoring_config.yaml"
    proposal = tmp_path / "proposal.yaml"
    _make_db(db)
    scoring_yaml.write_text(
        yaml.safe_dump({"base": {"w_cap_induction": 45.0, "w_cap_binding": 45.0}}),
        encoding="utf-8",
    )
    # The proposal injects a marker we can detect in the stubbed scorer.
    proposal.write_text(
        yaml.safe_dump(
            {
                "base": {
                    "_simulator_marker": "proposed",
                    "w_cap_induction": 30.0,
                    "w_cap_binding": 30.0,
                }
            }
        ),
        encoding="utf-8",
    )

    # Hand-picked scores: attn1 stays above floor under both; attn2 is
    # promoted under proposal; ssm1 stays passing; ssm2 stays failing.
    score_map = {
        "attn1": {"current": 480.0, "proposed": 480.0},
        "attn2": {"current": 440.0, "proposed": 470.0},
        "ssm1": {"current": 460.0, "proposed": 460.0},
        "ssm2": {"current": 410.0, "proposed": 415.0},
    }
    _stub_scoring_config(monkeypatch, tmp_yaml=scoring_yaml.with_suffix(".sim.yaml"))
    _stub_leaderboard_scoring(monkeypatch, score_map=score_map)

    # Import after stubs are in place so the simulator picks up the fakes.
    from research.tools import probe_promotion_simulator as sim

    report = sim.build_report(
        db_path=db,
        scoring_yaml_path=scoring_yaml,
        proposal_path=proposal,
        tiers=("screening",),
        current_floor=450.0,
        proposed_floor=450.0,
        limit=None,
    )

    promo = report["promotion_delta"]
    assert promo["promoted_n"] == 1
    assert "attn2" in promo["promoted_ids"]
    assert promo["demoted_n"] == 0
    assert promo["unchanged_passing"] >= 2

    # Reference rank sanity: gpt2 still > mamba under proposal.
    ref = report["reference_rank_check"]
    assert "gpt2" in ref["references_seen"]
    assert ref["current_scores"]["gpt2"] >= ref["current_scores"].get("mamba", 0)
    assert ref["proposed_scores"]["gpt2"] >= ref["proposed_scores"].get("mamba", 0)

    # Summary stats present.
    assert report["summary"]["n"] == 4


def test_promotion_simulator_does_not_write_db(tmp_path: Path, monkeypatch):
    db = tmp_path / "runs.db"
    scoring_yaml = tmp_path / "scoring_config.yaml"
    proposal = tmp_path / "proposal.yaml"
    _make_db(db)
    scoring_yaml.write_text(yaml.safe_dump({"base": {}}), encoding="utf-8")
    proposal.write_text(
        yaml.safe_dump({"base": {"_simulator_marker": "proposed"}}),
        encoding="utf-8",
    )

    score_map = {
        rid: {"current": 400.0 + i, "proposed": 400.0 + i}
        for i, rid in enumerate(("attn1", "attn2", "ssm1", "ssm2"))
    }
    _stub_scoring_config(monkeypatch, tmp_yaml=scoring_yaml.with_suffix(".sim.yaml"))
    _stub_leaderboard_scoring(monkeypatch, score_map=score_map)

    from research.tools import probe_promotion_simulator as sim

    mtime_before = db.stat().st_mtime
    sim.build_report(
        db_path=db,
        scoring_yaml_path=scoring_yaml,
        proposal_path=proposal,
        tiers=("screening",),
        current_floor=450.0,
        proposed_floor=450.0,
        limit=None,
    )
    mtime_after = db.stat().st_mtime
    assert mtime_before == mtime_after
