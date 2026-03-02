"""
Safety and Novelty Evaluation

Sandbox execution, behavioral fingerprinting, and novelty metrics
for synthesized programs.
"""

from .sandbox import safe_eval, SandboxResult
from .metrics import novelty_score, NoveltyMetrics

# Fingerprinting depends on optional native bindings (aria_core). Keep eval package
# importable for sandbox-only paths even when those bindings are unavailable.
try:
    from .fingerprint import compute_fingerprint, BehavioralFingerprint
except Exception:  # pragma: no cover - optional dependency guard
    compute_fingerprint = None
    BehavioralFingerprint = None
