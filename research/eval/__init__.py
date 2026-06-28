"""
Safety and Novelty Evaluation

Sandbox execution, behavioral fingerprinting, and novelty metrics
for synthesized programs.
"""

import importlib
from typing import TYPE_CHECKING

# Lazy re-exports (PEP 562). Importing any ``research.eval.*`` submodule runs this
# package ``__init__``; eagerly importing ``.sandbox`` here forced torch+triton
# (~1.1s) onto every consumer of the package — including the notebook → leaderboard
# → dashboard glue chain, none of which touches torch. Resolve on first attribute
# access instead so torch loads only when a sandbox eval actually runs.
_LAZY = {
    "safe_eval": ".sandbox",
    "SandboxResult": ".sandbox",
    "novelty_score": ".metrics",
    "NoveltyMetrics": ".metrics",
    "BehavioralFingerprint": ".fingerprint_types",
    # Fingerprinting depends on optional native bindings (aria_core); resolves to
    # None if those bindings are unavailable (handled in __getattr__).
    "compute_fingerprint": ".fingerprint_runtime",
}

if TYPE_CHECKING:  # static-analysis / IDE visibility only — no runtime import cost
    from .fingerprint_runtime import compute_fingerprint as compute_fingerprint
    from .fingerprint_types import BehavioralFingerprint as BehavioralFingerprint
    from .metrics import (
        NoveltyMetrics as NoveltyMetrics,
        novelty_score as novelty_score,
    )
    from .sandbox import SandboxResult as SandboxResult, safe_eval as safe_eval

__all__ = [
    "safe_eval",
    "SandboxResult",
    "novelty_score",
    "NoveltyMetrics",
    "BehavioralFingerprint",
    "compute_fingerprint",
]


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        return getattr(importlib.import_module(module, __name__), name)
    except Exception:
        if name == "compute_fingerprint":  # optional native dependency guard
            return None
        raise
