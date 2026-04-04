"""
Safety and Novelty Evaluation

Sandbox execution, behavioral fingerprinting, and novelty metrics
for synthesized programs.
"""

from .sandbox import safe_eval as safe_eval, SandboxResult as SandboxResult
from .metrics import novelty_score as novelty_score, NoveltyMetrics as NoveltyMetrics
from .fingerprint_types import BehavioralFingerprint as BehavioralFingerprint

# Fingerprinting depends on optional native bindings (aria_core). Keep eval package
# importable for sandbox-only paths even when those bindings are unavailable.
try:
    from .fingerprint_runtime import compute_fingerprint
except Exception:  # pragma: no cover - optional dependency guard
    compute_fingerprint = None
