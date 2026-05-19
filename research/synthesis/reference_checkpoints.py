"""Reference checkpoint registry.

Pre-trained model weights pinned as canonical baselines. Lives alongside
``reference_architectures.py`` (which builds blueprint graphs); this module
points at frozen ``.pt`` files that downstream tools can resolve by name.

Weights live under ``research/checkpoints/reference/`` — outside the
auto-pruned ``research/reports/`` tree so they survive the 14-day cleanup.
"""

from __future__ import annotations

import pathlib
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

REFERENCE_CHECKPOINTS: dict[str, dict[str, Any]] = {
    "mixer_interleaved_conv6_three_lane6_50m_100k": {
        "path": "research/checkpoints/reference/mixer_interleaved_conv6_three_lane6_50m_100k.pt",
        "params": 50_000_000,
        "dim": 320,
        "n_layers": 12,
        "mixer_pattern": "conv:6,three_lane:6",
        "trained_steps": 100_000,
        "trained_tokens": "~800M wikitext-103",
        "source_run": "scale60M_enriched_100k_interleaved_conv6_three_lane6",
        "model_config": {
            "use_position_embedding": True,
            "use_rope": False,
            "max_seq_len": 256,
        },
        "status": "legacy_abs_pos_embed",
        "notes": (
            "Pre-RoPE architecture (2026-05-17 training). pos_embed.weight shape "
            "(256, 320) caps inputs at seq_len=256, which is why the 4 hard probes "
            "(induction_validation, ar_validation, binding_multislot, binding_range) "
            "CUDA-asserted in the post-200K eval. Load only with "
            "use_position_embedding=True, use_rope=False, max_seq_len=256. "
            "Will be superseded by a RoPE-trained 100K reference once retraining "
            "completes; do not use the 200K resume weights."
        ),
    },
}


def resolve_reference_checkpoint(name: str) -> pathlib.Path:
    """Return the absolute ``Path`` to a registered reference checkpoint.

    Raises ``KeyError`` if the name is not registered. Raises
    ``FileNotFoundError`` if the registered path does not exist on disk.
    """
    if name not in REFERENCE_CHECKPOINTS:
        known = ", ".join(sorted(REFERENCE_CHECKPOINTS)) or "(none)"
        raise KeyError(f"unknown reference checkpoint {name!r}; known: {known}")
    rel = REFERENCE_CHECKPOINTS[name]["path"]
    abs_path = (_REPO_ROOT / rel).resolve()
    if not abs_path.exists():
        raise FileNotFoundError(
            f"reference checkpoint {name!r} registered at {abs_path} but file is missing"
        )
    return abs_path
