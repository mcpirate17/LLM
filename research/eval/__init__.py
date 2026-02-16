"""
Safety and Novelty Evaluation

Sandbox execution, behavioral fingerprinting, and novelty metrics
for synthesized programs.
"""

from .sandbox import safe_eval, SandboxResult
from .fingerprint import compute_fingerprint, BehavioralFingerprint
from .metrics import novelty_score, NoveltyMetrics
