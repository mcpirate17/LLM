"""Tests for untapped-pair → motif auto-registration."""

from __future__ import annotations

import json
from pathlib import Path

from research.synthesis._motifs_mined import (
    _MINED_MOTIF_CLASS,
    register_mined_motifs,
)
from research.synthesis._motif_types import Motif


def _proposal(op_a: str, op_b: str, composition: str = "sequential") -> dict:
    return {
        "op_a": op_a,
        "op_b": op_b,
        "composition": composition,
        "signature": f"{op_a}->{op_b}",
        "stability_score": 0.1,
    }


def _write_proposals(path: Path, proposals: list[dict]) -> None:
    path.write_text(
        json.dumps({"count": len(proposals), "candidates": proposals}),
        encoding="utf-8",
    )


def test_register_disabled_by_default(tmp_path: Path):
    validated: dict = {}
    by_class: dict = {}
    proposals = tmp_path / "p.json"
    _write_proposals(proposals, [_proposal("linear_proj", "rmsnorm")])
    registered = register_mined_motifs(
        validated, by_class, json_path=proposals, enable=False
    )
    assert registered == []
    assert validated == {}


def test_register_adds_compatible_pairs(tmp_path: Path):
    validated: dict = {}
    by_class: dict = {}
    proposals = tmp_path / "p.json"
    _write_proposals(
        proposals,
        [
            _proposal("linear_proj", "rmsnorm"),
            _proposal("linear_proj_up", "linear_proj_down"),
        ],
    )
    registered = register_mined_motifs(
        validated, by_class, json_path=proposals, enable=True
    )
    assert "mined_linear_proj_then_rmsnorm" in registered
    assert "mined_linear_proj_up_then_linear_proj_down" in registered
    assert all(validated[name].motif_class == _MINED_MOTIF_CLASS for name in registered)
    bucket = by_class.get(_MINED_MOTIF_CLASS, [])
    assert len(bucket) == 2
    assert all(isinstance(m, Motif) for m in bucket)


def test_register_skips_unknown_ops(tmp_path: Path):
    validated: dict = {}
    by_class: dict = {}
    proposals = tmp_path / "p.json"
    _write_proposals(
        proposals,
        [
            _proposal("linear_proj", "rmsnorm"),
            _proposal("nope_ghost_op", "rmsnorm"),  # unknown → skipped
        ],
    )
    registered = register_mined_motifs(
        validated, by_class, json_path=proposals, enable=True
    )
    assert registered == ["mined_linear_proj_then_rmsnorm"]


def test_register_skips_residual_composition(tmp_path: Path):
    """Only sequential pairs flow through the grammar's pair signature set."""
    validated: dict = {}
    by_class: dict = {}
    proposals = tmp_path / "p.json"
    _write_proposals(
        proposals,
        [_proposal("linear_proj", "rmsnorm", composition="residual")],
    )
    registered = register_mined_motifs(
        validated, by_class, json_path=proposals, enable=True
    )
    assert registered == []


def test_register_skips_collisions(tmp_path: Path):
    validated: dict = {
        "mined_linear_proj_then_rmsnorm": Motif(
            name="mined_linear_proj_then_rmsnorm",
            motif_class="ffn_core",
            steps=(),
        )
    }
    by_class: dict = {"ffn_core": [validated["mined_linear_proj_then_rmsnorm"]]}
    proposals = tmp_path / "p.json"
    _write_proposals(proposals, [_proposal("linear_proj", "rmsnorm")])
    registered = register_mined_motifs(
        validated, by_class, json_path=proposals, enable=True
    )
    assert registered == []
    # original motif untouched
    assert validated["mined_linear_proj_then_rmsnorm"].motif_class == "ffn_core"


def test_register_handles_missing_file(tmp_path: Path):
    registered = register_mined_motifs(
        {}, {}, json_path=tmp_path / "missing.json", enable=True
    )
    assert registered == []
