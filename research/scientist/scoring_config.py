"""Loader for the active scoring formula's tunable knobs.

Single source of truth: ``research/scoring_config.yaml``. Loaded once at
import time. Composes three layers (base + v11 overrides + v14 overrides)
into the three internal dicts that ``leaderboard_scoring`` expects.

Provides ``get_scoring_config_hash()`` — a SHA256 digest of the YAML bytes
truncated to 12 hex chars. Stamped on every rescored leaderboard row to
replace the meaningless ``scoring_version`` string column with real
provenance: "this row was scored under config matching hash X."

Hot-reload via ``reload_scoring_config()`` is exposed for the dashboard.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

import yaml

# YAML file lives at the research/ repo root, two parents up from this module.
_CONFIG_PATH: Path = Path(__file__).resolve().parents[1] / "scoring_config.yaml"


def _load_yaml(path: Path) -> tuple[Dict[str, Any], str]:
    raw_bytes = path.read_bytes()
    payload = yaml.safe_load(raw_bytes) or {}
    digest = hashlib.sha256(raw_bytes).hexdigest()[:12]
    return payload, digest


def _compose_layers(payload: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Build the three layered config dicts that the v10/v11/v14 functions consume."""
    base = dict(payload.get("base") or {})
    v11_overrides = dict(payload.get("v11_overrides") or {})
    v14_overrides = dict(payload.get("v14_overrides") or {})
    v11 = {**base, **v11_overrides}
    v14 = {**v11, **v14_overrides}
    return {"v10": base, "v11": v11, "v14": v14}


_PAYLOAD: Dict[str, Any]
_HASH: str
_LAYERS: Dict[str, Dict[str, float]]


def reload_scoring_config() -> str:
    """Re-read the YAML file and rebuild module state. Returns the new hash."""
    global _PAYLOAD, _HASH, _LAYERS
    _PAYLOAD, _HASH = _load_yaml(_CONFIG_PATH)
    _LAYERS = _compose_layers(_PAYLOAD)
    return _HASH


# Initialize at import.
reload_scoring_config()


def get_scoring_config_hash() -> str:
    """Return the SHA256(yaml_bytes)[:12] of the active config."""
    return _HASH


def get_config_path() -> Path:
    """Return the absolute path to the scoring config YAML."""
    return _CONFIG_PATH


def get_layered_configs() -> Dict[str, Dict[str, float]]:
    """Return the v10/v11/v14 layered configs for use by ``leaderboard_scoring``."""
    return _LAYERS


def get_section(name: str) -> Dict[str, Any]:
    """Return a top-level YAML section (e.g. ``trust_ceiling``, ``breakthrough_gates``)."""
    return dict(_PAYLOAD.get(name) or {})


def get_breakthrough_gates() -> Dict[str, float]:
    """Convenience accessor for ``breakthrough_gates`` section."""
    return get_section("breakthrough_gates")


def get_trust_ceiling() -> Dict[str, float]:
    """Convenience accessor for ``trust_ceiling`` section."""
    return get_section("trust_ceiling")
