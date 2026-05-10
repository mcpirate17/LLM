"""Tests for mined-template auto-registration into the TEMPLATES dict."""

from __future__ import annotations

import json
import random
from pathlib import Path

from research.synthesis._templates_mined import (
    _make_chain_template,
    register_mined_templates,
)
from research.synthesis.graph import ComputationGraph


def _candidate(
    name: str,
    chain: list[str],
    *,
    compile_passed: bool = True,
    forward_passed: bool = True,
    backward_passed: bool = True,
) -> dict:
    return {
        "proposed_template_name": name,
        "chain": chain,
        "validation": {
            "compile_passed": compile_passed,
            "validate_passed": True,
            "forward_passed": forward_passed,
            "backward_passed": backward_passed,
        },
    }


def _write_json(path: Path, candidates: list[dict]) -> None:
    payload = {
        "metadata": {},
        "candidates": candidates,
        "ready_for_registration": [
            c for c in candidates if c["validation"].get("backward_passed")
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_chain_template_callable_records_decisions(tmp_path: Path):
    """A constructed chain template emits ops + slot-usage entries."""
    tmpl = _make_chain_template("mined_test", ("linear_proj", "rmsnorm"))
    g = ComputationGraph(model_dim=64)
    inp = g.add_input()
    g.metadata["_active_template_instance"] = 0
    out = tmpl(g, inp, random.Random(0), None)
    g.set_output(out)
    op_names = [n.op_name for n in g.nodes.values() if not n.is_input]
    # Wrapper rmsnorm + linear_proj + rmsnorm (chain) = at least 3 nodes
    assert "linear_proj" in op_names
    slot_usage = g.metadata.get("template_slot_usage", [])
    assert any(s["template_name"] == "mined_test" for s in slot_usage)
    assert any(s["selected_motif_class"] == "mined_op" for s in slot_usage)


def test_register_disabled_by_default(tmp_path: Path):
    """No env flag, no enable kwarg → no registration."""
    target_templates: dict = {}
    target_weights: dict = {}
    candidates_path = tmp_path / "candidates.json"
    _write_json(candidates_path, [_candidate("x", ["linear_proj", "rmsnorm"])])
    registered = register_mined_templates(
        target_templates,
        target_weights,
        json_path=candidates_path,
        enable=False,
    )
    assert registered == []
    assert target_templates == {}


def test_register_adds_passing_candidates_when_enabled(tmp_path: Path):
    target_templates: dict = {}
    target_weights: dict = {}
    candidates_path = tmp_path / "candidates.json"
    _write_json(
        candidates_path,
        [
            _candidate("good", ["linear_proj", "rmsnorm"]),
            _candidate("smoke_failure", ["other"], backward_passed=False),
            _candidate("compile_failure", ["x"], compile_passed=False),
        ],
    )
    registered = register_mined_templates(
        target_templates,
        target_weights,
        json_path=candidates_path,
        enable=True,
    )
    assert registered == ["good"]
    assert "good" in target_templates
    assert target_weights["good"] == 0.5  # default mined weight


def test_register_skips_existing_template_names(tmp_path: Path):
    """Mined template names that collide with existing ones never overwrite."""
    target_templates: dict = {"existing_name": lambda *a, **k: 0}
    target_weights: dict = {"existing_name": 1.0}
    candidates_path = tmp_path / "candidates.json"
    _write_json(
        candidates_path,
        [_candidate("existing_name", ["linear_proj", "rmsnorm"])],
    )
    registered = register_mined_templates(
        target_templates,
        target_weights,
        json_path=candidates_path,
        enable=True,
    )
    assert registered == []
    # original entries unchanged
    assert target_weights["existing_name"] == 1.0


def test_register_caps_at_max(tmp_path: Path):
    target_templates: dict = {}
    target_weights: dict = {}
    candidates_path = tmp_path / "candidates.json"
    candidates = [
        _candidate(f"mined_{i}", ["linear_proj", "rmsnorm"]) for i in range(10)
    ]
    _write_json(candidates_path, candidates)
    registered = register_mined_templates(
        target_templates,
        target_weights,
        json_path=candidates_path,
        enable=True,
        max_register=3,
    )
    assert len(registered) == 3
    assert len(target_templates) == 3


def test_register_handles_missing_file(tmp_path: Path):
    """No JSON yet (mining never ran) → silent no-op."""
    registered = register_mined_templates(
        {}, {}, json_path=tmp_path / "missing.json", enable=True
    )
    assert registered == []
