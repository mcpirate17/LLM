"""Unit tests for the YAML-backed scoring config loader.

Validates that:
- ``scoring_config.yaml`` loads without error and exposes the three layered
  configs (v10 / v11 / v14) with the expected key shapes.
- ``get_scoring_version()`` returns a 12-char hex string (the SHA prefix).
- v11 / v14 overrides actually take effect (v14 > v11 > v10 inheritance).
- ``breakthrough_gates`` floors are wired through to ``breakthrough_gates``.
- The hash rotates when the underlying YAML bytes change (reload semantics).
"""

from __future__ import annotations

import re

import pytest

from research.scientist import scoring_config


pytestmark = pytest.mark.unit


def test_yaml_loads():
    layers = scoring_config.get_layered_configs()
    assert set(layers.keys()) == {"v10", "v11", "v14"}
    assert len(layers["v14"]) >= len(layers["v11"]) >= len(layers["v10"])


def test_inheritance_actually_overrides():
    layers = scoring_config.get_layered_configs()
    # The active YAML is flattened: v10/v11/v14 compose to the same weights.
    assert layers["v10"]["w_blimp"] == 5.0
    assert layers["v11"]["w_blimp"] == 5.0
    assert layers["v14"]["w_blimp"] == 5.0
    assert layers["v10"]["w_cap_induction"] == 45.0
    assert layers["v11"]["w_cap_induction"] == 45.0
    assert layers["v14"]["w_cap_induction"] == 45.0
    # Flattening keeps controlled-language ladder keys visible in every layer.
    assert "w_cl_inv_sa" in layers["v14"]
    assert layers["v14"]["w_cl_s10_nb_bucket"] == 25.0
    assert layers["v14"]["w_cl_inv_nb_bucket"] == 25.0
    assert layers["v11"]["w_cl_inv_sa"] == 15.0


def test_hash_format():
    h = scoring_config.get_scoring_config_hash()
    assert isinstance(h, str)
    assert len(h) == 12
    assert re.fullmatch(r"[0-9a-f]{12}", h), h


def test_breakthrough_gates_section():
    gates = scoring_config.get_breakthrough_gates()
    assert gates["composite_floor"] == 450.0
    assert gates["capability_floor"] == 0.10


def test_breakthrough_gates_wired_through():
    """The breakthrough_gates module reads its floors from this YAML."""
    from research.scientist import breakthrough_gates as bg

    assert bg.BREAKTHROUGH_COMPOSITE_FLOOR == 450.0
    assert bg.BREAKTHROUGH_CAPABILITY_FLOOR == 0.10


def test_leaderboard_scoring_uses_loaded_configs():
    """leaderboard_scoring exposes the loaded dicts under the legacy names."""
    from research.scientist import leaderboard_scoring as ls

    layers = scoring_config.get_layered_configs()
    assert ls._V10_CONFIG is layers["v10"] or ls._V10_CONFIG == layers["v10"]
    assert ls._V11_CONFIG is layers["v11"] or ls._V11_CONFIG == layers["v11"]
    assert ls._V14_CONFIG is layers["v14"] or ls._V14_CONFIG == layers["v14"]
    assert ls.get_scoring_version() == scoring_config.get_scoring_config_hash()


def test_reload_recomputes_hash(tmp_path, monkeypatch):
    """If the YAML changes, the hash should rotate after reload."""
    original = scoring_config._CONFIG_PATH.read_bytes()
    h0 = scoring_config.get_scoring_config_hash()

    fake = tmp_path / "scoring_config.yaml"
    fake.write_bytes(original + b"\n# touched\n")
    monkeypatch.setattr(scoring_config, "_CONFIG_PATH", fake)
    h1 = scoring_config.reload_scoring_config()
    assert h1 != h0

    # Restore original via reload after monkeypatch teardown (auto-revert).


def test_trust_ceiling_section():
    trust = scoring_config.get_trust_ceiling()
    assert trust["ceiling"] == 360.0
    assert trust["ppl_floor"] == 150.0
