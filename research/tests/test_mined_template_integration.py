"""End-to-end integration test: validated mined chain → registered template
→ grammar picks it → produces a valid graph.

Distinct from the unit tests for the auto-registrar in
``test_mined_template_registration.py``, which mock the JSON and verify
the registrar in isolation. This test exercises the actual import-time
registration path: write a fake validated-candidates JSON to the default
location, reload ``synthesis.templates``, and confirm:
  1. The mined name appears in TEMPLATES + DEFAULT_TEMPLATE_WEIGHTS.
  2. pick_template can return the mined name.
  3. apply_template(graph, ..., template_name=mined_name) runs without
     raising and emits a graph that compiles into a CompiledLayer.

The test isolates side effects with module reloads and restores process
state after teardown so the rest of the suite is unaffected.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Iterator

import pytest

from research.synthesis._templates_mined import (
    _ENV_FLAG,
    _ENV_PATH,
)


D = 64


def _candidate(name: str, chain: list[str]) -> dict:
    return {
        "proposed_template_name": name,
        "chain": chain,
        "validation": {
            "compile_passed": True,
            "validate_passed": True,
            "forward_passed": True,
            "backward_passed": True,
        },
    }


def _write_validated_json(path: Path, candidates: list[dict]) -> None:
    payload = {
        "metadata": {},
        "candidates": candidates,
        "ready_for_registration": candidates,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _purge_synthesis_modules() -> None:
    """Drop synthesis modules from sys.modules so they re-execute on import."""
    for name in list(sys.modules):
        if name.startswith("research.synthesis"):
            del sys.modules[name]


@pytest.fixture
def mined_template_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Stage a validated-candidates JSON at the default path + flip the env
    flag, then purge synthesis modules so registration re-runs at import."""
    json_path = tmp_path / "validated.json"
    _write_validated_json(
        json_path,
        [_candidate("mined_e2e_block", ["linear_proj", "rmsnorm"])],
    )
    # Env-path override survives sys.modules purge; setattr would be lost when
    # the module re-imports during the test's first `from synthesis.templates`.
    monkeypatch.setenv(_ENV_PATH, str(json_path))
    monkeypatch.setenv(_ENV_FLAG, "1")
    _purge_synthesis_modules()
    yield json_path
    _purge_synthesis_modules()


def test_validated_chain_registers_and_grammar_can_pick(mined_template_env):
    """Full path: import templates → mined name present → pick_template
    can return it → apply_template produces a valid graph."""
    from research.synthesis.templates import (
        DEFAULT_TEMPLATE_WEIGHTS,
        TEMPLATES,
        apply_template,
        pick_template,
    )
    from research.synthesis.graph import ComputationGraph

    assert "mined_e2e_block" in TEMPLATES
    assert DEFAULT_TEMPLATE_WEIGHTS["mined_e2e_block"] == 0.5

    # pick_template can return the mined name when restricted by
    # allowed_template_names — confirms the registrar made it visible to
    # the picker's index table.
    name, fn, _trial = pick_template(
        random.Random(0),
        allowed_template_names=["mined_e2e_block"],
    )
    assert name == "mined_e2e_block"

    g = ComputationGraph(model_dim=D)
    inp = g.add_input()
    out = apply_template(g, inp, random.Random(0), template_name="mined_e2e_block")
    g.set_output(out)

    op_names = [n.op_name for n in g.nodes.values() if not n.is_input]
    assert "linear_proj" in op_names
    assert any(name == "rmsnorm" for name in op_names)


def test_template_compiles_into_layer(mined_template_env):
    """Mined-registered template + grammar produces a graph that survives
    the full compile path (not just CompiledLayer construct)."""
    from research.synthesis.compiler import _compile_layer_module
    from research.synthesis.graph import ComputationGraph
    from research.synthesis.templates import apply_template

    g = ComputationGraph(model_dim=D)
    inp = g.add_input()
    out = apply_template(g, inp, random.Random(0), template_name="mined_e2e_block")
    g.set_output(out)
    # Should not raise — the registered chain template emits ops the
    # full compile pipeline (dispatch wiring + native handlers) recognises.
    layer = _compile_layer_module(g)
    assert layer is not None
